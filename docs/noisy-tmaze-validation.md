# Noisy T-Maze validation for issue #20

Validated locally on 2026-11-22, based on Passive T-Maze commit `d8a6a99`.

## Acceptance criteria

| Criterion | Evidence | Local result |
|---|---|---|
| AC1: select/configure the variant | `NoisyTMaze-v0` registration, factory forwarding tests, and an actual `run.py` CPU training run with `--noise-level 0.5 --noise-std 0.1` | Passed |
| AC2: zero noise matches Passive T-Maze | Equal spaces and seeded observations/rewards/termination/success across 30 episodes each at lengths 1, 10 and 300, including correct routes, wrong branches and random movements | Passed |
| AC3: oracle succeeds at every supported noise level | 1,000 observation-driven oracle episodes per combination of probability `{0, 0.25, 0.5, 1}` and standard deviation `{0.1, 1}`: **8,000/8,000 successes**; additional length 1/300 checks | Passed |

The oracle result extends to the entire supported parameter range: its initial
cue and junction observations are always clean, and noise never changes movement
or rewards. The tested grid samples that invariant; it is not an exhaustive test
of every real-valued probability/standard deviation.

## Additional checks

- Full unit suite: **34 tests passed**. Tests cover mixture means/spread, replacement
  frequency, unclipped values and their observation bounds, seeding, unchanged
  dynamics/rewards and clean locations, invalid inputs, and unrelated factories.
- Type checking: **40 source files passed**, using the repository's CI-pinned
  Torch 1.11 type environment. All pre-commit lint, format and hygiene hooks passed.
- Runtime checks used Python 3.10.21, NumPy 1.22.4, Gym 0.23.1 and Torch 2.6.0+cpu.
  No dependencies were added.

## Existing environment compatibility and CPU runner checks

This command completed for `NoisyTMaze-v0`, `DiscreteCarFlag-v0` and `Memory-5-v0`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py --model DTQN --envs ENV_ID \
  --corridor-length 12 --noise-level 0.5 --noise-std 0.1 \
  --context 8 --history 8 --device cpu --num-steps 64 \
  --eval-frequency 32 --eval-episodes 2 --batch 8 --buf-size 2000 --seed 1 \
  --project-name noisy-validation-FRESH --disable-wandb --verbose
```

Each run completed replay prepopulation, training and two evaluations, writing
two results CSV rows. The actual local project name was
`dtqn-noisy-tmaze-smoke-1zzhuexl`; raw files are ignored under `policies/`.
Passing the noise flags did not affect Car Flag or Memory Cards. These checks
establish execution compatibility, not learned performance or experiment #21.

## Definition of done

- [x] Code, tests and documented flags/defaults/effects prepared locally.
- [x] Local sanity checks and required existing-environment startup checks pass.
- [x] AC1-AC3 have the evidence above.
- [ ] Reviewed PR merged into `main`, after resolving the Passive T-Maze dependency.

The local checks do not close issue #20. Gaussian distractors can resemble the
cue, but its initial-position identity remains available; this is not a claim
that success proves general retrieval or the necessity of kNN.
