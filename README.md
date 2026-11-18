# Pluggable kNN Episodic Memory for Deep Q-Learning Agents in Partially Observable Environments

Pluggable kNN episodic memory (from Memorizing Transformers) for DTQN and DRQN agents in partially observable RL. Tests whether agents can use information beyond the context window, evaluated on the Passive T-Maze, Car Flag and Memory Cards.

> **Status:** UFRGS Deep Learning final project (Group 7), due **11/30/2026**. This repository is a fork of [kevslinger/DTQN](https://github.com/kevslinger/DTQN); the original DTQN documentation is kept [below](#deep-transformer-q-networks-for-partially-observable-reinforcement-learning).
>
> **Paper (Overleaf):** _link to be added_ ([#42](https://github.com/AndersonGACFilho/DTQN-EpisodicMemory/issues/42))

## Overview

In partially observable problems (POMDPs), the current observation is not enough to act and the agent must remember something seen earlier in the episode. The Deep Transformer Q-Network (DTQN) applies a transformer decoder over the last K observations, so anything older than K steps falls out of context, and Cherepanov et al. (2024) show that DTQN then performs at chance.

We couple the external kNN memory from Memorizing Transformers (Wu et al., 2022) to DTQN: (key, value) pairs from earlier steps of the episode are stored and retrieved by similarity, giving access to information beyond the window.

## Research question

Can a pluggable kNN memory let Deep Q-Learning agents use information beyond the context window, without hurting performance when the information is inside it?

- **Beyond the window (dependency > K):** DTQN + memory reaches a quantitative success target where the original DTQN is at chance.
- **Within the window (dependency ≤ K):** DTQN + memory ties with the original DTQN.

The quantitative target and the tie criterion are fixed in the revised proposal. Every configuration uses at least 5 seeds and reports mean ± std with 95% confidence intervals.

## Environments and what each tests

| Environment | What it tests |
|---|---|
| **Passive T-Maze** (Ni et al., 2023) | Main experiment: the corridor length controls the dependency distance, below and above K |
| **Noisy T-Maze** | Whether retrieval still finds the cue when it is less distinct, since stored keys receive no gradient |
| **Car Flag** and **Memory Cards** | Already solved by DTQN with K = 50, so they check that the memory does not hurt within the window |

## Agents and baselines

| Agent | Role |
|---|---|
| DTQN (original) | Main baseline |
| DTQN with K covering the corridor | Answers "why not increase K?", with time and GPU-memory cost, where feasible |
| DTQN + kNN memory | Proposed method |
| DRQN (+ kNN memory) | **Optional**: only if the DTQN results are promising and time allows; needs its own coupling (what is key/value and how the read combines with the hidden state) and hypothesis |

## Design questions addressed

- **Memory during training:** DTQN trains on length-K subsequences sampled from the replay buffer, so the memory must hold the (key, value) pairs of the steps before the window.
- **Keys without gradient:** nothing forces the query to retrieve the cue step; the noisy T-Maze and the stop-gradient vs. differentiable ablation test this.
- **Retrieval diagnostic:** we log whether the cue step is in the top-k at decision time, to separate retrieval failure from usage failure.
- **Ablations:** gate, top-k, memory size, key layer, stop-gradient vs. differentiable memory.
- **Compute budget:** an early pilot estimates GPU-hours before the experiment grid is fixed; the desired but infeasible grid is discussed in the paper.

## Project management

Work is tracked in the [GitHub Project](https://github.com/users/AndersonGACFilho/projects/19) (Kanban, Roadmap, Goals, Epics, Priorities and one board per person).

| Milestone | Due | Goal |
|---|---|---|
| M1: Proposal revised & design | 11/17 | Proposal updated with the professor's feedback; memory design fixed |
| M2: Memory module & environments | 11/20 | Passive T-Maze, evaluation harness and kNN memory integrated into DTQN |
| M3: Pilot & experiment grid | 11/22 | GPU-hour pilot, large-K cost and the final experiment grid |
| M4: Experiments | 11/25 | Main experiments, within-window check, noisy T-Maze and ablations |
| M5: Analysis & figures | 11/26 | Results table, retrieval analysis and figures |
| M6: Paper draft | 11/28 | All sections written and reviewed |
| M7: Submission | 11/30 | Paper submitted |

| Role | Responsibility | Paper sections |
|---|---|---|
| **P1** (Estevan Küster) | Passive T-Maze, evaluation harness, main experiment, results, figures | Results |
| **P2** (Anderson Gonçalves) | Memory design and module, DTQN integration, retrieval diagnostic, ablations | Method |
| **P3** (Esthevan Ferretti) | DTQN baseline, GPU-hour pilot, large-K cost, within-window check, DRQN (optional) | Related work, setup |
| **P4** (João Augusto Laner) | Proposal revision, targets, noisy T-Maze, experiment grid, coordination, submission | Introduction, discussion, conclusion, abstract |

**Conventions:** issue types (Epic, Feature, Task, Experiment, Writing, Research, Bug); form templates in [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE); epics group tasks as sub-issues; blocked by / blocking links follow the critical path; labels `P1`–`P4` mark the people involved and `optional` marks the DRQN work.

## References

- Esslinger, Platt and Amato (2022). *Deep Transformer Q-Networks for Partially Observable Reinforcement Learning.* [arXiv:2206.01078](https://arxiv.org/abs/2206.01078)
- Wu, Rabe, Hutchins and Szegedy (2022). *Memorizing Transformers.* ICLR 2022. [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)
- Hausknecht and Stone (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs.* [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)
- Cherepanov et al. (2024). *Unraveling the Complexity of Memory in RL Agents.* [arXiv:2412.06531](https://arxiv.org/abs/2412.06531)
- Ni, Ma, Eysenbach and Bacon (2023). *When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment.* NeurIPS 2023. [arXiv:2307.03864](https://arxiv.org/abs/2307.03864)

---

# Deep Transformer Q-Networks for Partially Observable Reinforcement Learning

Deep Transformer Q-Network (DTQN) is an extension of [DQN](https://www.nature.com/articles/nature14236) and [DRQN](https://arxiv.org/abs/1507.06527) designed to encode an agent's history effectively for solving partially observable reinforcement learning tasks.
Our architecture is built from a Transformer Decoder (like [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)).
DTQN is a sequence-to-sequence model; that is, given a history of the agent's interactions (either observation or action-observation) with the environment, DTQN outputs a sequence of Q-values.
Each element in the output sequence represents the Q-values for each action, given the agent's history up to that point.
For instance, the 3rd vector of Q-values was generated based on only the first three interactions between the agent and the environment.
This method allows us to train much more efficiently than a Sequence-to-One method, since we get `Sequence` more data points to use for training.
Our results providence strong evidence indicating a transformer can solve partially observable domains faster than previous recurrent approaches.
Our paper is now publicly available on arXiv! 
You can read it [here](https://arxiv.org/abs/2206.01078).

Please note that we are continuing to work on this repository to extend and improve DTQN.
As such, the code in this branch may not reflect the code submitted with the original paper.
We will keep the [paper](https://github.com/kevslinger/DTQN/tree/paper) branch frozen with the code from the original paper, and only update it as needed to fix bugs.

## Table of Contents
- [Deep Transformer Q-Networks for Partially Observable Reinforcement Learning](#deep-transformer-q-networks-for-partially-observable-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Creating Environment](#creating-environment)
    - [Installing gym-gridverse](#installing-gym-gridverse)
    - [Installing rl-parsers](#installing-rl-parsers)
    - [Installing gym-pomdps](#installing-gym-pomdps)
  - [Running Experiments](#running-experiments)
    - [Experiment Argument Details](#experiment-argument-details)
    - [Ablations](#ablations)
      - [Transformer Decoder Structure](#transformer-decoder-structure)
      - [Positional Encodings](#positional-encodings)
      - [Intermediate Q-value prediction](#intermediate-q-value-prediction)
  - [Citing DTQN](#citing-dtqn)
  - [Contributing](#contributing)

## Installation

To run our code, you must first set up your environment.
We recommend using `virtualenv` with `pip` to set up a virtual environment and dependency manager.
For our experiments, we use `python3.8`; while other versions of python will probably work, they are untested and we cannot guarantee the same performance.

### Creating Environment

First, create a virtual environment with `python3.8`, and then install the dependencies. This can be done with:

```bash
virtualenv venv -p 3.8
source venv/bin/activate
pip install -r requirements.txt
```

This basic setup will allow you to run the `Car Flag` and `Memory Cards` environments from our paper.
If you only have interest in running those domains, skip to the [Running Experiments](#running-experiments) section.
Otherwise, you need to install the `gym-gridverse` repo to run the gridverse experiments, and the `gym-pomdps` and `rl-parsers` repos to run the classic POMDPs experiments.

### Installing gym-gridverse

To install gym-gridverse, you need to clone the `gym-gridverse` [github repo](https://github.com/abaisero/gym-gridverse.git).
Finally, once you have the source code, pip install it into the virtual enviroment.
This can be done as follows:

```bash
git clone git@github.com:abaisero/gym-gridverse.git
cd gym-gridverse
pip install .
```

### Installing rl-parsers

If you also wish to run the Hallway and HeavenHell experiments, you will need to install `rl-parsers` and `gym-pomdps`.
First, install `rl-parsers` by cloning the [github repo](https://github.com/abaisero/rl-parsers.git) and installing it into your python virtual environment.

```bash
git clone git@github.com:abaisero/rl-parsers.git
cd rl-parsers
pip install -e .
```

### Installing gym-pomdps

Next, install `gym-pomdps` by cloning the [github repo](https://github.com/abaisero/gym-pomdps.git) and installing it into your python virtual environment.

```bash
git clone git@github.com:abaisero/gym-pomdps.git
cd gym-pomdps
pip install .
```

Tada! Now, you should be ready to run experiments.

## Running Experiments

Our experiment script is `run.py`. 
The arguments are explained at the bottom of the file.
Using the command
```shell
python run.py
```

You will run the default settings, which will run DTQN on the Car Flag domain as tested in the paper.

Here we provide a mapping from domain name as used in the paper to domain name used in the code:

- Hallway: `POMDP-hallway-episodic-v0`
- HeavenHell: `POMDP-heavenhell_3-episodic-v0`
- Gridverse memory 5x5: `gv_memory.5x5.yaml`
- Gridverse memory 7x7: `gv_memory.7x7.yaml`
- Gridverse Memory 9x9: `gv_memory.9x9.yaml`
- Gridverse four rooms 7x7: `gv_memory_four_rooms.7x7.yaml`
- Car Flag: `DiscreteCarFlag-v0`
- Memory Cards: `Memory-5-v0`

If you want to change the environment, use the `--envs` flag.
You can add multiple `envs` as long as they share the same observation and action space. 
In that case, the agent will collect data with policy rollouts by randomly sampling one of the provided envs.
During evaluation, it will evaluate the policy separately on each environment, and record the results for each domain separately.
To reproduce our results, you may need to change the `in-embed` flag as well.
For `POMDP-hallway-episodic-v0`, `POMDP-heavenhell_3-episodic-v0`, and `DiscreteCarFlag-v0` domains, we used `--in-embed 64`.
For all others tasks, we use `--in-embed 128`.
For instance, to reproduce our Gridverse memory 7x7 experiment, you can use the command:

```shell
python run.py --envs gv_memory.7x7.yaml --in-embed 128
```

Which will run for 2,000,000 timesteps.
In our paper, we run our experiments with random seeds 1, 2, 3, 4, 5.

### Experiment Argument Details

We use [weights and biases](https://wandb.ai) to log our results.
If you do not have a weights and biases account, we recommend you get one!
However, if you still do not want to use weights and biases, you can use the `--disable-wandb` flag.
Then your results will be stored to a CSV file in `policies/<project_name>/<env>/<config>.csv`.

If you do not have access to a gpu, set `--device cpu` to train on CPU.

If you want command line outputs to view training results, use `--verbose`.

| Argument | Description |
| -------- | ----------- |
| `--project-name` | The project name (for wandb) or directory name (for local logging) to store the results. |
| `--disable-wandb` | Use `--disable-wandb` to log locally. |
| `--time-limit` | Time limit allowed for job. Useful for some cluster jobs such as slurm. |
| `--model` | Network model to use. |
| `--envs` | Domain to use. You can supply multiple domains, but they must have the same observation and action space. With multiple environments, the agent will sample a new one on each episode reset for conducting policy rollouts and collection experience. During evaluation, it will perform the same evaluation for each domain (Note: this may significantly slow down your run! Consider increasing the eval-frequency or reducing the eval-episodes). |
| `--num-steps` | Number of steps to train the agent. |
| `--tuf` | How many steps between each (hard) target network update. |
| `--lr` | Learning rate for the optimizer. |
| `--batch` | Batch size. |
| `--buf-size` | Number of timesteps to store in replay buffer. Note that we store the max length episodes given by the environment, so episodes that take longer will be padded at the end. This does not affect training but may affect the number of real observations in the buffer. |
| `--eval-frequency` | How many training timesteps between agent evaluations. |
| `--eval-episodes` | Number of episodes for each evaluation period. |
| `--device` | Pytorch device to use. |
| `--context` | For DRQN and DTQN, the context length to use to train the network. |
| `--obs-embed` | For discrete observation domains only. The number of features to give each observation. |
| `--a-embed` | The number of features to give each action. A value of 0 will prevent the policy from using the previous action. |
| `--in-embed` | The dimensionality of the network. In the transformer, this is referred to as `d_model`. |
| `--max-episode-steps` | The maximum number of steps allowed in the environment. If `env` has a `max_episode_steps`, this will be inferred. Otherwise, this argument must be supplied. |
| `--seed` | The random seed to use. |
| `--save-policy` | Use this to save the policy so you can load it later for rendering. |
| `--verbose` | Print out evaluation results as they come in to the console. |
| `--render` | Enjoy mode (NOTE: must have a trained policy saved). |
| `--history` | This is how many (intermediate) Q-values we use to train for each context. To turn off intermediate Q-value prediction, set `--history 1`. To use the entire context, set history equal to the context length. |
| `--heads` | Number of heads to use for the transformer. |
| `--layers` | Number of transformer blocks to use for the transformer. |
| `--dropout` | Dropout probability. |
| `--discount` | Discount factor. |
| `--gate` | Combine step to use. |
| `--identity` | Whether or not to use identity map reordering. |
| `--pos` | The type of positional encodings to use. |
| `--bag-size` | The size of the persistent memory bag. |
| `--slurm-job-id` | The `$SLURM_JOB_ID` assigned to this job. |
| |


### Ablations

#### Transformer Decoder Structure 
To run DTQN with the GRU-like gating mechanism, use `--gate gru`.

To run DTQN with identity map reordering, use `--identity`.

To use both GRU-like gating as well as identity map reordering, use `--gate gru --identity`.

#### Positional Encodings
To run DTQN with the sinusoidal positional encodings, use `--pos sin`.

To run DTQN without positional encodings, use `--pos none`.

#### Intermediate Q-value prediction
To run DTQN without intermediate Q-value prediction, use `--history`.

## Citing DTQN

To cite this paper/code in publications, please use the following bibtex:

```bibtex
@article{esslinger2022dtqn,
  title = {Deep Transformer Q-Networks for Partially Observable Reinforcement Learning},
  author = {Esslinger, Kevin and Platt, Robert and Amato, Christopher},
  journal= {arXiv preprint arXiv:2206.01078},
  year = {2022},
}

```

## Contributing

Feel free to open a pull request with updates, contributions, and modifications.