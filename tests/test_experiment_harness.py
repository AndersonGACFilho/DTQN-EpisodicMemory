"""Protocol tests: exact budgets, faithful traces, final weights and seed statistics."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import torch

from envs.associative_corridor import AssociativeCorridor
from envs.passive_tmaze import PassiveTMaze
from utils.experiment import (
    build_agent,
    effective_settings,
    eval_run,
    evaluate_episodes,
    load_policy,
    make_environment,
    read_seed_result,
    train_run,
    write_seed_result,
)
from utils.experiment_config import (
    RunConfig,
    config_cli,
    expand_grid,
    load_grid,
)
from utils.experiment_io import ResourceMeter, read_json, write_json
from utils.experiment_sweep import (
    METRICS,
    aggregate_batch,
    check_resume,
    describe,
    execute_batch,
    make_cases,
    new_batch,
)
from utils.logging_utils import RecordCSV
from utils.random import set_global_seed
from utils.trajectories import world_state


def tiny_config(**kwargs: Any) -> RunConfig:
    params: Dict[str, Any] = {
        "env": "PassiveTMaze-v0",
        "corridor_length": 3,
        "context": 2,
        "in_embed": 16,
        "heads": 2,
        "layers": 1,
        "batch": 2,
        "buf_size": 200,
        "num_steps": 7,
        "prefill_steps": 12,
        "eval_episodes": 3,
    }
    params.update(kwargs)
    return RunConfig(**params)


def tiny_grid() -> Dict[str, List[Any]]:
    return {key: [value] for key, value in tiny_config().to_dict().items()}


def read_trace(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as file:
        return [json.loads(line) for line in file]


class ConfigTests(unittest.TestCase):
    def test_outer_rejects_scientific_cli_overrides_and_dry_run_launches_nothing(
        self,
    ) -> None:
        import contextlib
        import io

        from train_and_eval import main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "grid.json"
            config.write_text(json.dumps(tiny_grid()))
            command = [
                "train_and_eval.py",
                "--config",
                str(config),
                "--output-root",
                str(root / "results"),
            ]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ), patch("sys.argv", command + ["--context", "8"]):
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 2)
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "sys.argv", command + ["--dry-run"]
            ), patch(
                "subprocess.run",
                side_effect=AssertionError("dry run must not launch jobs"),
            ):
                self.assertEqual(main(), 0)
            self.assertFalse((root / "results").exists())

    def test_cartesian_product_keeps_inactive_parameter_choices(self) -> None:
        grid = tiny_grid()
        grid.update(context=[2, 4], training_seed=[1, 2, 3], noise_level=[0.0, 0.5])
        cases = make_cases(grid)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["id"] for case in cases}), 12)
        self.assertEqual(len({case["config_id"] for case in cases}), 4)

    def test_invalid_combination_does_not_discard_other_cases(self) -> None:
        grid = tiny_grid()
        grid.update(in_embed=[16, 17])
        self.assertEqual(
            [case["status"] for case in make_cases(grid)], ["pending", "invalid"]
        )

    def test_malformed_configs_rejected_before_execution(self) -> None:
        invalid = [
            {**tiny_grid(), "training_seed": []},
            {**tiny_grid(), "training_seed": [1, 1]},
            {**tiny_grid(), "training_seed": [True]},
            {**tiny_grid(), "lr": [float("nan")]},
            {**tiny_grid(), "device": ["cpu"]},
            {**tiny_grid(), "context": 2},
            {key: value for key, value in tiny_grid().items() if key != "eval_seed"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for config in invalid:
                with self.subTest(config=config):
                    path.write_text(json.dumps(config))
                    with self.assertRaises(ValueError):
                        load_grid(path)
            path.write_text('{"training_seed": [1], "training_seed": [2]}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                load_grid(path)

    def test_singleton_and_boolean_cli(self) -> None:
        self.assertEqual(len(expand_grid(tiny_grid())), 1)
        self.assertEqual(
            config_cli({"identity": False, "num_steps": 7}), ["--num-steps", "7"]
        )
        self.assertEqual(config_cli({"identity": True}), ["--identity"])

    def test_requested_and_effective_settings(self) -> None:
        config = tiny_config(history=50)
        self.assertEqual(config.history, 50)
        effective = effective_settings(config, 4)
        self.assertEqual(effective["parameters"]["history"], 2)
        self.assertIn("noise_level", effective["inactive_parameters"])


class TrainingAndTraceTests(unittest.TestCase):
    def test_dqn_updates_its_current_observation_in_train_and_eval(self) -> None:
        config = tiny_config(model="DQN")
        env, horizon = make_environment(config)
        set_global_seed(1, env)
        agent = build_agent(config, env, horizon, "cpu")
        for mode in (agent.eval_off, agent.eval_on):
            mode()
            agent.context_reset(env.reset())
            obs, reward, done, _ = env.step(0)
            agent.observe(obs, 0, reward, done)
            self.assertTrue(np.array_equal(agent.context.obs[0], obs))
            self.assertEqual(agent.context.timestep, 1)
        env.close()

    def test_exact_budget_all_actions_final_weights_and_greedy_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = tiny_config()
            train_dir, eval_dir = root / "train", root / "eval"
            training = train_run(config, train_dir)
            self.assertEqual(training["training_steps"], 7)
            self.assertGreater(training["gradient_updates"], 0)
            self.assertEqual(training["prefill_steps"], 12)
            self.assertEqual(training["peak_gpu_memory_bytes"], 0)
            records = read_trace(train_dir / "train_trajectories.jsonl.gz")
            steps = [r for r in records if r["event"] == "step"]
            self.assertEqual(len(steps), 19)
            partial = [
                r for r in records if r["event"] == "episode_end" and not r["complete"]
            ]
            self.assertEqual(len(partial), 1)
            self.assertEqual(partial[0]["episode_length"], 3)
            self.assertEqual(partial[0]["completion_reason"], "step_budget")
            self.assertIsNone(partial[0]["success"])

            # Re-enact every action and compare observations/rewards/physical
            # state. Inferred initial goal matches the saved start snapshot.
            env = PassiveTMaze(corridor_length=3)
            for record in records:
                if record["event"] == "episode_start":
                    env.reset()
                    env.goal_y = record["world_state"]["goal_y"]
                    previous_obs = record["observation"]
                elif record["event"] == "step":
                    self.assertEqual(record["observation"], previous_obs)
                    obs, reward, done, _ = env.step(record["action"])
                    self.assertEqual(record["next_observation"], obs.tolist())
                    self.assertEqual(record["reward"], reward)
                    self.assertEqual(record["terminated"], done)
                    self.assertEqual(record["world_state"], world_state(env))
                    previous_obs = obs.tolist()

            bundle = torch.load(train_dir / "model.pt", map_location="cpu")
            agent, env, metadata = load_policy(train_dir / "model.pt", "cpu")
            self.assertEqual(metadata["requested"], config.to_dict())
            self.assertEqual(agent.replay_buffer.max_size, 2)
            for key, value in agent.policy_network.state_dict().items():
                self.assertTrue(
                    torch.equal(value.cpu(), bundle["state_dict"][key].cpu()), key
                )
            # Policies trained before the parameter rename remain usable.
            old_parameters = dict(bundle["metadata"]["requested"])
            old_parameters["seed"] = old_parameters.pop("training_seed")
            bundle["metadata"]["requested"] = old_parameters
            old_path = root / "old-model.pt"
            torch.save(bundle, old_path)
            old_agent, old_env, old_metadata = load_policy(old_path, "cpu")
            self.assertEqual(old_metadata["requested"], config.to_dict())
            for key, value in old_agent.policy_network.state_dict().items():
                self.assertTrue(
                    torch.equal(value, agent.policy_network.state_dict()[key])
                )
            old_env.close()
            saved_weights = {
                key: value.clone()
                for key, value in agent.policy_network.state_dict().items()
            }
            direct = root / "direct"
            direct.mkdir()
            with patch.object(
                agent, "train", side_effect=AssertionError("eval must not train")
            ):
                metrics = evaluate_episodes(agent, env, 3, 10000, direct)
            self.assertEqual(metrics["eval_episodes"], 3)
            self.assertEqual(agent.num_train_steps, 0)
            self.assertEqual(agent.replay_buffer.pos, [0, 0])
            self.assertFalse(agent.policy_network.training)
            for key, value in agent.policy_network.state_dict().items():
                self.assertTrue(torch.equal(value, saved_weights[key]), key)
            env.close()
            evaluation = eval_run(train_dir / "model.pt", eval_dir, eval_episodes=3)
            self.assertEqual(evaluation["success_rate"], metrics["success_rate"])
            self.assertEqual(
                read_trace(direct / "eval_trajectories.jsonl.gz"),
                read_trace(eval_dir / "eval_trajectories.jsonl.gz"),
            )
            result = write_seed_result(train_dir, eval_dir, root / "result.csv")
            self.assertEqual(result["training_steps"], 7)
            self.assertEqual(
                read_seed_result(root / "result.csv")["training_seed"], "1"
            )
            with (train_dir / "train_losses.csv").open() as file:
                self.assertEqual(
                    len(list(csv.DictReader(file))), training["gradient_updates"]
                )

    def test_prefill_finishes_episode_and_has_separate_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = train_run(
                tiny_config(prefill_steps=5, num_steps=1), Path(temp) / "train"
            )
            self.assertEqual(result["prefill_steps"], 8)
            self.assertEqual(result["training_steps"], 1)

    def test_time_limit_is_real_and_trace_marks_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = tiny_config(max_episode_steps=2, num_steps=3, prefill_steps=0)
            directory = Path(temp) / "train"
            train_run(config, directory)
            endings = [
                r
                for r in read_trace(directory / "train_trajectories.jsonl.gz")
                if r["event"] == "episode_end"
            ]
            self.assertTrue(endings[0]["truncated"])
            self.assertFalse(endings[0]["terminated"])
            self.assertEqual(endings[0]["episode_length"], 2)
            self.assertFalse(endings[1]["complete"])

    def test_recorder_does_not_change_randomness_or_policy_input(self) -> None:
        from envs.noisy_tmaze import NoisyTMaze

        first, second = NoisyTMaze(10), NoisyTMaze(10)
        first.seed(7)
        second.seed(7)
        self.assertTrue(np.array_equal(first.reset(), second.reset()))
        for action in [0] * 10 + [1]:
            snapshot = world_state(first)
            snapshot["x"] = -100
            obs_a, reward_a, done_a, info_a = first.step(action)
            obs_b, reward_b, done_b, info_b = second.step(action)
            self.assertTrue(np.array_equal(obs_a, obs_b))
            self.assertEqual((reward_a, done_a, info_a), (reward_b, done_b, info_b))

    def test_car_flag_and_memory_cards_start_with_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for env_id, adapter in [
                ("DiscreteCarFlag-v0", "CarFlag"),
                ("Memory-5-v0", "Memory"),
            ]:
                config = tiny_config(
                    env=env_id, prefill_steps=0, num_steps=2, buf_size=2000
                )
                directory = Path(temp) / env_id
                train_run(config, directory)
                trace = read_trace(directory / "train_trajectories.jsonl.gz")
                self.assertEqual(trace[0]["world_state"]["adapter"], adapter)
                self.assertEqual(len([r for r in trace if r["event"] == "step"]), 2)

    def test_positive_partial_associative_return_is_failure(self) -> None:
        class PartialOracle:
            def eval_on(self) -> None:
                pass

            def context_reset(self, observation: Any) -> None:
                pass

            def get_action(self, epsilon: float) -> int:
                self_outer.assertEqual(epsilon, 0.0)
                if env.position in env.queries:
                    if env.correct_answers:
                        return 2
                    return 1 if env.directions[env.queries[env.position]] > 0 else 3
                return 0

            def observe(self, *args: Any) -> None:
                pass

        self_outer = self
        env = AssociativeCorridor(12)
        with tempfile.TemporaryDirectory() as temp:
            metrics = evaluate_episodes(PartialOracle(), env, 5, 1, Path(temp))
            self.assertEqual(metrics["mean_return"], 1.0)
            self.assertEqual(metrics["success_rate"], 0.0)

    def test_horizon_required_and_capacity_checked(self) -> None:
        env = PassiveTMaze(10)
        with patch("utils.experiment.env_processing.make_env", return_value=env), patch(
            "utils.experiment.env_processing.get_env_max_steps", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "finite horizon"):
                make_environment(tiny_config())
        env, horizon = make_environment(tiny_config())
        set_global_seed(1, env)
        with self.assertRaisesRegex(ValueError, "batch \\+ 2"):
            build_agent(tiny_config(buf_size=2), env, horizon, "cpu")


class ResourceTests(unittest.TestCase):
    def test_cuda_peak_reset_before_allocation_and_synchronized(self) -> None:
        calls: List[str] = []
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.synchronize", side_effect=lambda device: calls.append("sync")
        ), patch(
            "torch.cuda.reset_peak_memory_stats",
            side_effect=lambda device: calls.append("reset"),
        ), patch("torch.cuda.max_memory_allocated", return_value=1234), patch(
            "utils.experiment_io.perf_counter", side_effect=[10.0, 12.0]
        ):
            meter = ResourceMeter("cuda:0")
            calls.append("model_allocation")
            metrics = meter.finish()
        self.assertEqual(calls, ["sync", "reset", "model_allocation", "sync"])
        self.assertEqual(metrics["wall_clock_seconds"], 2.0)
        self.assertEqual(metrics["peak_gpu_memory_bytes"], 1234)


class AggregationTests(unittest.TestCase):
    def test_failed_eval_reuses_trained_model_on_resume(self) -> None:
        import subprocess

        real_run = subprocess.run
        fail = True

        def run_with_eval_failure(command: List[str], **kwargs: Any) -> Any:
            if fail and command[1].endswith("/eval.py"):
                return subprocess.CompletedProcess(command, 1)
            return real_run(command, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            batch = new_batch(tiny_grid(), Path(temp), "resume")
            with patch(
                "utils.experiment_sweep.subprocess.run",
                side_effect=run_with_eval_failure,
            ):
                self.assertEqual(execute_batch(batch), 1)
                first = read_json(batch / "manifest.json")["cases"][0]
                self.assertEqual(first["status"], "eval_failed")
                path = batch / first["train_path"] / "model.pt"
                timestamp = path.stat().st_mtime_ns
                fail = False
                self.assertEqual(execute_batch(batch), 0)
                last = read_json(batch / "manifest.json")["cases"][0]
                self.assertEqual(last["status"], "complete")
                self.assertEqual(
                    [a["phase"] for a in last["attempts"]], ["train", "eval", "eval"]
                )
                self.assertEqual(path.stat().st_mtime_ns, timestamp)

    def test_student_t_ci_and_sample_std(self) -> None:
        stats = describe([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(float(stats["std"] or 0), 1.5811388300841898)
        self.assertAlmostEqual(float(stats["ci95_low"] or 0), 1.036756838522439)
        self.assertAlmostEqual(float(stats["ci95_high"] or 0), 4.963243161477561)
        self.assertIsNone(describe([0.5])["std"])
        self.assertIsNone(describe([0.5])["ci95_low"])

    def test_missing_seeds_withhold_statistics_and_attempts_not_counted_twice(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            grid = tiny_grid()
            grid["training_seed"] = [1, 2]
            batch = new_batch(grid, Path(temp), "test")
            manifest = read_json(batch / "manifest.json")
            case = manifest["cases"][0]
            path = batch / "runs" / case["id"]
            path.mkdir(parents=True)
            row = {
                **case["parameters"],
                **dict.fromkeys(METRICS, 0.5),
                **manifest["code"],
            }
            with RecordCSV(path / "result.csv", list(row)) as output:
                output.write(row)
            case.update(
                status="complete",
                attempts=[
                    {"phase": "train", "status": "failed"},
                    {"phase": "train", "status": "complete"},
                ],
            )
            write_json(batch / "manifest.json", manifest)
            summary = aggregate_batch(batch)[0]
            self.assertEqual(summary["completed_seeds"], 1)
            self.assertEqual(summary["missing_seeds"], "[2]")
            self.assertIsNone(summary["success_rate_mean"])
            self.assertEqual(summary["status"], "incomplete")

    def test_resume_rejects_changed_grid_or_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            batch = new_batch(tiny_grid(), Path(temp), "test")
            manifest = read_json(batch / "manifest.json")
            check_resume(manifest, tiny_grid())
            with self.assertRaisesRegex(ValueError, "grid"):
                check_resume(manifest, {**tiny_grid(), "training_seed": [2]})
            manifest["code"] = {}
            with self.assertRaisesRegex(ValueError, "code identity"):
                check_resume(manifest, tiny_grid())

    def test_tiny_process_sweep_resume_failure_and_saved_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            grid = tiny_grid()
            # Environment construction failure is discovered in its own process;
            # the later valid environment must still train and evaluate.
            grid["env"] = ["DoesNotExist-v0", "PassiveTMaze-v0"]
            batch = new_batch(grid, Path(temp), "integration")
            self.assertEqual(execute_batch(batch), 1)
            manifest = read_json(batch / "manifest.json")
            failed, passed = manifest["cases"]
            self.assertEqual(failed["status"], "train_failed")
            self.assertEqual(passed["status"], "complete")
            self.assertEqual(len(passed["attempts"]), 2)
            passed_artifact = batch / passed["train_path"] / "model.pt"
            stamp = passed_artifact.stat().st_mtime_ns
            self.assertEqual(execute_batch(batch), 1)
            manifest = read_json(batch / "manifest.json")
            self.assertEqual(len(manifest["cases"][0]["attempts"]), 2)
            self.assertEqual(passed_artifact.stat().st_mtime_ns, stamp)
            with patch(
                "subprocess.run",
                side_effect=AssertionError("aggregation must not launch jobs"),
            ), patch(
                "torch.load",
                side_effect=AssertionError("aggregation must not load models"),
            ):
                summaries = aggregate_batch(batch)
            complete = [row for row in summaries if row["status"] == "single_seed"]
            self.assertEqual(len(complete), 1)
            self.assertIsNotNone(complete[0]["success_rate_mean"])
            self.assertIsNone(complete[0]["success_rate_std"])


if __name__ == "__main__":
    unittest.main()
