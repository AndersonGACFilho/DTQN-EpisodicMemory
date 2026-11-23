# Associative Corridor validation

Validated locally on 2026-11-22, based on Passive T-Maze commit `d8a6a99`.

## Checks

- Full unit suite: **36 tests passed**, including 1,000 observation-driven oracle
  episodes with four correct answers, return 4, and the specified horizon.
- Schedules contain all four instruction/use pairs, allow interleaving, and are
  reproducible by seed. Minimum length 8 and length 300 both pass.
- Wrong movements and answers end immediately. A wrong answer after an earlier
  correct one preserves the positive partial return but is still a failure.
- Type checking: **40 source files passed**, using the repository's CI-pinned
  Torch 1.11 type environment. All pre-commit lint, format and hygiene hooks passed.
- Runtime checks used Python 3.10.21, NumPy 1.22.4, Gym 0.23.1 and Torch 2.6.0+cpu.
  No dependencies were added.

## CPU runner checks

The following command completed for each of `AssociativeCorridor-v0`,
`DiscreteCarFlag-v0` and `Memory-5-v0`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_legacy.py --model DTQN --envs ENV_ID \
  --corridor-length 12 --context 8 --history 8 --device cpu \
  --num-steps 64 --eval-frequency 32 --eval-episodes 2 --batch 8 \
  --buf-size 2000 --seed 1 --project-name associative-validation-FRESH \
  --disable-wandb --verbose
```

Each run completed replay prepopulation, training, and two evaluation points,
writing two rows to its results CSV. The actual local project name was
`dtqn-associative-corridor-smoke-bn8yt_2o`; raw files are ignored under `policies/`.
These are integration checks, not evidence of learned task performance.

Direct checks through `run.evaluate` also passed over 20 episodes each:

| Policy/task                                         | Success rate | Mean return | Mean steps |
|-----------------------------------------------------|-------------:|------------:|-----------:|
| Observation-driven associative oracle               |            1 |           4 |         12 |
| Correct first answer, wrong second answer           |            0 |           1 |        8.5 |
| Legacy positive-return task with `is_success=False` |            1 |           1 |          1 |

Only the new environment declares `requires_explicit_success=True`; the legacy
success rule is unchanged for other environments. kNN integration, rendering,
multi-seed learning experiments and publication are not part of this change.
