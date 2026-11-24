"""Single-configuration training and final-policy evaluation. No sweep JSON input."""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import torch
from gym.wrappers.time_limit import TimeLimit

from dtqn.networks.knn_memory import MemoryConfig
from utils import env_processing
from utils.agent_utils import get_agent
from utils.epsilon_anneal import LinearAnneal
from utils.experiment_config import RunConfig, normalize_saved_parameters
from utils.experiment_io import (
    ResourceMeter,
    code_identity,
    read_json,
    runtime_versions,
    write_json,
)
from utils.logging_utils import RecordCSV
from utils.random import RNG, make_deterministic, set_global_seed
from utils.trajectories import TrajectoryWriter, world_state

EPISODE_COLUMNS = [
    "phase",
    "episode",
    "phase_step",
    "episode_return",
    "episode_length",
    "success",
    "complete",
    "terminated",
    "truncated",
    "completion_reason",
]
# Called as hook(agent, training_step) after every training interaction
StepHook = Callable[[Any, int], None]
LOSS_ATTRIBUTES = [
    "td_errors",
    "grad_norms",
    "qvalue_max",
    "qvalue_mean",
    "qvalue_min",
    "target_max",
    "target_mean",
    "target_min",
]


def make_environment(config: RunConfig) -> Tuple[Any, int]:
    env = env_processing.make_env(
        config.env,
        config.corridor_length,
        noise_level=config.noise_level,
        noise_std=config.noise_std,
    )
    try:
        if config.max_episode_steps > 0:
            env = TimeLimit(  # type: ignore[no-untyped-call]
                env, max_episode_steps=config.max_episode_steps
            )
        horizon = env_processing.get_env_max_steps(env)
        if horizon is None or horizon <= 0:
            raise ValueError(
                "Environment has no finite horizon; supply --max-episode-steps"
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("The agents require a discrete action space")
        if env.action_space.n > 256:
            raise ValueError(
                "Replay actions use uint8: at most 256 discrete actions are supported"
            )
        return env, int(horizon)
    except Exception:
        env.close()
        raise


def build_agent(
    config: RunConfig, env: Any, horizon: int, device: str, *, evaluation: bool = False
) -> Any:
    context = 1 if config.model == "DQN" else config.context
    capacity = config.buf_size
    if evaluation:
        # Reuse the existing agent's inference/context logic without allocating
        # a full training replay buffer. Nothing is stored in evaluation mode.
        capacity = 2 * max(horizon, context)
    elif capacity // max(horizon, context) < config.batch + 2:
        raise ValueError("buf_size must hold at least batch + 2 padded episodes")
    return get_agent(
        config.model,
        (env,),
        config.obs_embed,
        config.a_embed,
        config.in_embed,
        capacity,
        torch.device(device),
        config.lr,
        config.batch,
        context,
        horizon,
        min(config.history, context),
        config.tuf,
        config.discount,
        config.heads,
        config.layers,
        config.dropout,
        config.identity,
        config.gate,
        config.pos,
        config.bag_size,
        memory_enabled=config.memory_enabled,
        memory_config=MemoryConfig(
            size=config.memory_size,
            top_k=config.memory_top_k,
            layer=config.memory_layer,
            gate=config.memory_gate,
            gate_bias_init=config.memory_gate_bias_init,
            stop_grad=config.memory_stop_grad,
        ),
    )


def effective_settings(config: RunConfig, horizon: int) -> Dict[str, Any]:
    params = config.to_dict()
    if config.model == "DQN":
        params["context"] = 1
    params["history"] = min(config.history, params["context"])
    params["max_episode_steps"] = horizon
    inactive: List[str] = []
    if config.model not in {"DTQN-kNN", "DTQN-kNN-cached"} or not config.memory_enabled:
        inactive.extend(key for key in params if key.startswith("memory_"))
    if config.env not in {"PassiveTMaze-v0", "NoisyTMaze-v0", "AssociativeCorridor-v0"}:
        inactive.append("corridor_length")
    if config.env != "NoisyTMaze-v0":
        inactive.extend(["noise_level", "noise_std"])
    if not config.model.startswith("DTQN"):
        inactive.extend(
            ["heads", "layers", "dropout", "gate", "identity", "pos", "bag_size"]
        )
    if config.model == "DQN":
        inactive.append("history")
    return {"parameters": params, "inactive_parameters": inactive}


def episode_success(env: Any, info: Dict[str, Any], reward: float) -> bool:
    return bool(
        env_processing.get_episode_success(
            info,
            reward,
            getattr(env.unwrapped, "requires_explicit_success", False),
        )
    )


def collect_training(
    agent: Any,
    env: Any,
    config: RunConfig,
    trajectories: TrajectoryWriter,
    episodes: RecordCSV,
    losses: RecordCSV,
    step_hook: Optional[StepHook] = None,
) -> Dict[str, int]:
    counts = {"prefill_steps": 0, "training_steps": 0, "training_episodes": 0}
    eps = LinearAnneal(1.0, 0.1, max(1, config.num_steps // 10))
    agent.eval_off()
    for phase, budget in (
        ("prefill", config.prefill_steps),
        ("train", config.num_steps),
    ):
        steps = 0
        episode = 0
        while steps < budget:
            observation = env.reset()
            agent.context_reset(observation)
            trajectories.write(
                "episode_start",
                phase=phase,
                episode=episode,
                phase_step=steps,
                observation=observation,
                world_state=world_state(env),
            )
            total_reward, length, done = 0.0, 0, False
            info: Dict[str, Any] = {}
            truncated = False
            while not done and (phase == "prefill" or steps < budget):
                epsilon = 1.0 if phase == "prefill" else float(eps.val)
                action = (
                    int(RNG.rng.integers(env.action_space.n))
                    if phase == "prefill"
                    else int(agent.get_action(epsilon=epsilon))
                )
                # Some legacy environments mutate observations in place.
                before = (
                    observation.copy() if hasattr(observation, "copy") else observation
                )
                observation, reward, done, info = env.step(action)
                truncated = bool(info.get("TimeLimit.truncated", False))
                agent.observe(observation, action, reward, bool(done and not truncated))
                steps += 1
                length += 1
                total_reward += float(reward)
                trajectories.write(
                    "step",
                    phase=phase,
                    episode=episode,
                    episode_step=length,
                    phase_step=steps,
                    observation=before,
                    next_observation=observation,
                    action=action,
                    reward=reward,
                    epsilon=epsilon,
                    terminated=bool(done and not truncated),
                    truncated=truncated,
                    info=info,
                    world_state=world_state(env),
                )
                if done:
                    agent.replay_buffer.flush()
                if phase == "train":
                    previous_updates = agent.num_train_steps
                    agent.train()
                    eps.anneal()
                    if agent.num_train_steps > previous_updates:
                        losses.write(
                            {
                                "training_step": steps,
                                "gradient_update": agent.num_train_steps,
                                **{
                                    key: getattr(agent, key).q[-1]
                                    for key in LOSS_ATTRIBUTES
                                },
                            }
                        )
                    if step_hook is not None:
                        step_hook(agent, steps)
            complete = bool(done)
            row = {
                "phase": phase,
                "episode": episode,
                "phase_step": steps,
                "episode_return": total_reward,
                "episode_length": length,
                "success": episode_success(env, info, total_reward)
                if complete
                else None,
                "complete": complete,
                "terminated": bool(done and not truncated),
                "truncated": truncated,
                "completion_reason": "time_limit"
                if truncated
                else "terminated"
                if done
                else "step_budget",
            }
            episodes.write(row)
            trajectories.write("episode_end", **row)
            episode += 1
        counts["prefill_steps" if phase == "prefill" else "training_steps"] = steps
        if phase == "train":
            counts["training_episodes"] = episode
    counts["gradient_updates"] = int(agent.num_train_steps)
    return counts


def train_run(
    config: RunConfig,
    output: Path,
    device: str = "cpu",
    verbose: bool = False,
    step_hook: Optional[StepHook] = None,
) -> Dict[str, Any]:
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Training output must be empty: {output}")
    make_deterministic()
    meter = ResourceMeter(device)
    env, horizon = make_environment(config)
    try:
        set_global_seed(config.training_seed, env)
        agent = build_agent(config, env, horizon, device)
        metadata = {
            "schema_version": 1,
            "requested": config.to_dict(),
            "effective": effective_settings(config, horizon),
            "code": code_identity(),
            "versions": runtime_versions(),
            "parameter_count": sum(
                p.numel() for p in agent.policy_network.parameters()
            ),
        }
        with TrajectoryWriter(
            output / "train_trajectories.jsonl.gz"
        ) as trace, RecordCSV(
            output / "train_episodes.csv",
            EPISODE_COLUMNS,
        ) as episodes, RecordCSV(
            output / "train_losses.csv",
            [
                "training_step",
                "gradient_update",
                *LOSS_ATTRIBUTES,
            ],
        ) as losses:
            counts = collect_training(
                agent, env, config, trace, episodes, losses, step_hook
            )
        # Always save the FINAL policy, including non-parameter state_dict buffers.
        torch.save(
            {
                "format_version": 1,
                "metadata": metadata,
                "state_dict": agent.policy_network.state_dict(),
            },
            output / "model.pt",
        )
        resources = meter.finish()
        result = {**metadata, **counts, **resources, "status": "complete"}
        write_json(output / "train_metadata.json", result)
        if verbose:
            print(
                f"Trained {config.env}: {counts['training_steps']} steps, "
                f"{counts['gradient_updates']} updates; saved {output / 'model.pt'}"
            )
        return result
    finally:
        env.close()


def load_policy(model_path: Path, device: str) -> Tuple[Any, Any, Dict[str, Any]]:
    # Bundles contain tensors and primitive metadata only, compatible with both
    # PyTorch 1.11 and the weights-only default in newer PyTorch.
    bundle = torch.load(model_path, map_location=device)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model bundle; expected a train.py model.pt")
    metadata = bundle["metadata"]
    metadata["requested"] = normalize_saved_parameters(metadata["requested"])
    config = RunConfig(**metadata["requested"])
    config.validate()
    env, horizon = make_environment(config)
    try:
        set_global_seed(config.training_seed, env)
        agent = build_agent(config, env, horizon, device, evaluation=True)
        agent.policy_network.load_state_dict(bundle["state_dict"])
        agent.eval_on()
        return agent, env, dict(metadata)
    except Exception:
        env.close()
        raise


def evaluate_episodes(
    agent: Any, env: Any, count: int, seed: int, output: Path
) -> Dict[str, Any]:
    # Reseed AFTER agent construction's observation-space probes. The first eval
    # episode then comes from exactly eval_seed for every independent train seed.
    set_global_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    agent.eval_on()
    successes, total_reward, total_length = 0, 0.0, 0
    with TrajectoryWriter(output / "eval_trajectories.jsonl.gz") as trace, RecordCSV(
        output / "eval_episodes.csv",
        EPISODE_COLUMNS,
    ) as episodes:
        for episode in range(count):
            observation = env.reset()
            agent.context_reset(observation)
            trace.write(
                "episode_start",
                phase="eval",
                episode=episode,
                phase_step=total_length,
                observation=observation,
                world_state=world_state(env),
            )
            done, length, reward_sum = False, 0, 0.0
            while not done:
                action = int(agent.get_action(epsilon=0.0))
                before = (
                    observation.copy() if hasattr(observation, "copy") else observation
                )
                observation, reward, done, info = env.step(action)
                truncated = bool(info.get("TimeLimit.truncated", False))
                agent.observe(observation, action, reward, bool(done and not truncated))
                length += 1
                total_length += 1
                reward_sum += float(reward)
                trace.write(
                    "step",
                    phase="eval",
                    episode=episode,
                    episode_step=length,
                    phase_step=total_length,
                    observation=before,
                    next_observation=observation,
                    action=action,
                    reward=reward,
                    epsilon=0.0,
                    terminated=bool(done and not truncated),
                    truncated=truncated,
                    info=info,
                    world_state=world_state(env),
                )
            success = episode_success(env, info, reward_sum)
            row = {
                "phase": "eval",
                "episode": episode,
                "phase_step": total_length,
                "episode_return": reward_sum,
                "episode_length": length,
                "success": success,
                "complete": True,
                "terminated": not truncated,
                "truncated": truncated,
                "completion_reason": "time_limit" if truncated else "terminated",
            }
            episodes.write(row)
            trace.write("episode_end", **row)
            successes += int(success)
            total_reward += reward_sum
    return {
        "eval_episodes": count,
        "eval_seed": seed,
        "successes": successes,
        "success_rate": successes / count,
        "mean_return": total_reward / count,
        "mean_episode_length": total_length / count,
    }


def eval_run(
    model_path: Path,
    output: Path,
    device: str = "cpu",
    eval_episodes: int = 100,
    eval_seed: int = 10_000,
    verbose: bool = False,
) -> Dict[str, Any]:
    if eval_episodes < 1 or eval_seed < 0:
        raise ValueError("eval_episodes must be positive and eval_seed nonnegative")
    make_deterministic()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Evaluation output must be empty: {output}")
    meter = ResourceMeter(device)
    agent, env, metadata = load_policy(model_path, device)
    try:
        with torch.no_grad():
            metrics = evaluate_episodes(agent, env, eval_episodes, eval_seed, output)
        result = {
            "schema_version": 1,
            **metrics,
            **meter.finish(),
            "training_seed": metadata["requested"]["training_seed"],
            "model_path": str(model_path.resolve()),
            "status": "complete",
            "code": code_identity(),
            "versions": runtime_versions(),
        }
        write_json(output / "eval_metadata.json", result)
        if verbose:
            print(
                f"Evaluated {eval_episodes} greedy episodes: success {result['success_rate']:.3f}"
            )
        return result
    finally:
        env.close()


def write_seed_result(train_dir: Path, eval_dir: Path, path: Path) -> Dict[str, Any]:
    training = read_json(train_dir / "train_metadata.json")
    evaluation = read_json(eval_dir / "eval_metadata.json")
    if evaluation["model_path"] != str((train_dir / "model.pt").resolve()):
        raise ValueError("Evaluation results do not belong to this training model")
    params = normalize_saved_parameters(training["requested"])
    result = {
        **params,
        **{
            key: evaluation[key]
            for key in (
                "eval_seed",
                "eval_episodes",
                "success_rate",
                "mean_return",
                "mean_episode_length",
            )
        },
        **{key: training[key] for key in ("training_steps", "gradient_updates")},
        "actual_prefill_steps": training["prefill_steps"],
        "train_wall_clock_seconds": training["wall_clock_seconds"],
        "eval_wall_clock_seconds": evaluation["wall_clock_seconds"],
        "total_wall_clock_seconds": training["wall_clock_seconds"]
        + evaluation["wall_clock_seconds"],
        "train_peak_gpu_memory_bytes": training["peak_gpu_memory_bytes"],
        "eval_peak_gpu_memory_bytes": evaluation["peak_gpu_memory_bytes"],
        "peak_gpu_memory_bytes": max(
            training["peak_gpu_memory_bytes"], evaluation["peak_gpu_memory_bytes"]
        ),
        "train_device": training["device"],
        "eval_device": evaluation["device"],
        "model_path": str((train_dir / "model.pt").resolve()),
        "train_artifacts": str(train_dir.resolve()),
        "eval_artifacts": str(eval_dir.resolve()),
        "git_commit": training["code"]["git_commit"],
        "source_sha256": training["code"]["source_sha256"],
    }
    with RecordCSV(path, list(result)) as file:
        file.write(result)
    return result


def read_seed_result(path: Path) -> Dict[str, str]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result in {path}")
    row = rows[0]
    if "seed" in row:
        if "training_seed" in row:
            raise ValueError("Saved result contains both seed and training_seed")
        row["training_seed"] = row.pop("seed")
    return row
