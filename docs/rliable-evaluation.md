# Multi-seed evaluation pipeline

How [`run_rliable_grid.py`](../run_rliable_grid.py) trains and evaluates a grid,
and how [`analysis/rliable_report.py`](../analysis/rliable_report.py) turns the
saved records into rliable metrics. Commands, options and how to read the results
are in the [README](../README.md#multi-seed-evaluation-with-rliable).

## Modules

| Module | Responsibility |
|---|---|
| [`evaluation/grid.py`](../evaluation/grid.py) | Reads the grid JSON, expands (configuration, training seed) jobs, derives `eval_seed = 10000 + train_seed`, filters jobs and checks which are complete |
| [`evaluation/router.py`](../evaluation/router.py) | Decides the execution plan: device (`--device auto` picks CUDA when available), worker count (`--workers auto` calibrates) and mode (one worker is sequential, more are parallel). It never creates CUDA tensors |
| [`evaluation/execution.py`](../evaluation/execution.py) | `ExecutionPlan`, the executors behind one interface (in-process for sequential, spawned processes for parallel), the `EXECUTORS` registry and the single failure-handling loop |
| [`evaluation/calibration.py`](../evaluation/calibration.py) | `--workers auto`: times short probe runs with 1, 2, 4 and 8 workers in worker processes and picks the fastest count that still pays off |
| [`evaluation/job.py`](../evaluation/job.py) | Runs one job: `train_run` with a step hook that evaluates periodically, then writes the records and metadata |
| [`evaluation/runner.py`](../evaluation/runner.py) | Runs greedy episodes against any policy with `reset`/`act`/`observe`; knows nothing about training |
| [`evaluation/sides.py`](../evaluation/sides.py) | Registry of junction-side readers per environment |
| [`evaluation/records.py`](../evaluation/records.py) | `EpisodeRecord` and the tidy CSV format |
| [`analysis/metrics.py`](../analysis/metrics.py) | `(n_seeds, n_tasks)` score matrices and rliable interval estimates |
| [`analysis/diagnostics.py`](../analysis/diagnostics.py) | Fixed-side policy detection |
| [`analysis/plots.py`](../analysis/plots.py), [`analysis/tables.py`](../analysis/tables.py) | Figures and CSV/Markdown tables from precomputed estimates |
| [`analysis/rliable_report.py`](../analysis/rliable_report.py) | Orchestrates the report |

The parent process must not open a CUDA context before parallel workers start:
on the tested driver, workers on the same GPU then fail with
`CUDA error: out of memory` even with free memory. The router only calls
`torch.cuda.is_available()`, which is safe, and calibration runs every probe in a
worker process, even with one worker.

Add an execution mode by registering its executor factory in `EXECUTORS`; an
executor only yields `(job, error)` pairs, and `run_tasks` handles logging and
failures for every mode.

Training reuses [`utils/experiment.py`](../utils/experiment.py) unchanged except
for an optional `step_hook`, called after every training interaction. Evaluation
saves and restores the Python, NumPy, PyTorch and agent random states, so the
trained weights match a run without evaluation (covered by
`tests/test_evaluation_protocol.py`).

## Flow

```mermaid
flowchart TB
    subgraph CONFIG["1. Configuration"]
        GRID["experiments/rliable_tmaze.json<br/>parameters, train_seeds,<br/>eval_episodes, eval_interval, comparisons"]
        JOBS["expand_jobs<br/>model x context x layers x seed"]
        SEED["derive_eval_seed<br/>eval_seed = 10000 + train_seed"]
        GRID --> JOBS
        SEED -.-> JOBS
    end

    subgraph SCHEDULE["2. Scheduling"]
        FILTER{"--only KEY=VALUE"}
        DONE{"is_complete?"}
        SKIP(["skip: already done"])
        FILTER --> DONE
        DONE -- yes --> SKIP
    end

    subgraph TRAIN["3. Training (existing train_run)"]
        STEP["training step"]
        CHECK{"eval_interval reached<br/>or final step?"}
        STEP --> CHECK
        CHECK -- no --> STEP
    end

    subgraph EVAL["4. Periodic evaluation"]
        SAVE["save RNG state<br/>agent.eval_on()"]
        RUN["run_evaluation<br/>greedy episodes on eval_seed"]
        SIDES["side reader<br/>goal_side, chosen_side"]
        COLLECT["RecordCollector<br/>one EpisodeRecord per episode"]
        RESTORE["restore RNG state<br/>agent.eval_off()"]
        SAVE --> RUN --> SIDES --> COLLECT --> RESTORE
    end

    subgraph STORE["5. Storage: results/raw"]
        CSV[("run_id.csv<br/>tidy episode records")]
        META[("run_id.json<br/>config, commit, versions")]
    end

    subgraph ANALYSIS["6. Analysis: analysis.rliable_report"]
        LOAD["load grid records<br/>error if a run is missing"]
        MATRIX["score matrices<br/>(n_seeds, n_tasks)"]
        AGG["mean, IQM, median,<br/>optimality gap + 95% CI"]
        POI["P(DTQN-kNN > DTQN)"]
        PROF["performance profiles"]
        CURVE["IQM learning curves"]
        BIAS["side bias<br/>flag above 90%"]
        LOAD --> MATRIX
        MATRIX --> AGG
        MATRIX --> POI
        MATRIX --> PROF
        MATRIX --> CURVE
        LOAD --> BIAS
    end

    subgraph OUTPUT["7. Outputs"]
        FIGS[/"results/figures<br/>PNG and PDF"/]
        TABLES[/"results/summary<br/>CSV and summary.md"/]
    end

    JOBS --> FILTER
    DONE -- no --> STEP
    CHECK -- yes --> SAVE
    RESTORE --> STEP
    CHECK -- "training finished" --> CSV
    CHECK -- "training finished" --> META
    CSV --> LOAD
    META --> LOAD
    AGG --> FIGS
    POI --> FIGS
    PROF --> FIGS
    CURVE --> FIGS
    AGG --> TABLES
    POI --> TABLES
    BIAS --> TABLES
```

## Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as run_rliable_grid.py
    participant Grid as evaluation.grid
    participant Job as evaluation.job
    participant Train as train_run
    participant Evaluator as PeriodicEvaluator
    participant Runner as evaluation.runner
    participant Env as eval env
    participant Agent as AgentPolicy
    participant Raw as results/raw
    participant Report as analysis.rliable_report
    participant Metrics as analysis.metrics + rliable
    participant Out as plots + tables

    User->>CLI: --grid, --seeds, --only
    CLI->>Grid: load_grid_spec, expand_jobs
    Grid-->>CLI: jobs with eval_seed = 10000 + train_seed
    CLI->>Grid: select_jobs, is_complete
    Grid-->>CLI: pending jobs

    loop each pending job
        CLI->>Job: run_job(job)
        Job->>Train: train_run(config, step_hook)
        loop each training step
            Train->>Evaluator: hook(agent, step)
            opt every eval_interval steps and at the final step
                Evaluator->>Evaluator: save RNG state, eval_on
                Evaluator->>Env: seed(eval_seed)
                Evaluator->>Runner: run_evaluation(env, policy, writer)
                loop eval_episodes
                    Runner->>Env: reset
                    Runner->>Agent: reset(observation)
                    loop until done
                        Runner->>Agent: act
                        Agent-->>Runner: action
                        Runner->>Env: step(action)
                        Env-->>Runner: observation, reward, done
                        Runner->>Agent: observe(transition)
                    end
                    Runner->>Env: read goal and chosen side
                    Runner->>Evaluator: write(episode, outcome)
                end
                Evaluator->>Evaluator: restore RNG state, eval_off
            end
        end
        Train-->>Job: final model and training metadata
        Job->>Raw: write run_id.csv, then run_id.json
    end

    User->>Report: --grid, --reps
    Report->>Raw: read records of every grid run
    Raw-->>Report: episode records or missing run error
    Report->>Metrics: build score matrices
    Metrics-->>Report: aggregates, P(improvement), profiles, curves
    Report->>Report: side bias diagnostic
    Report->>Out: figures and tables
    Out-->>User: results/figures, results/summary
```
