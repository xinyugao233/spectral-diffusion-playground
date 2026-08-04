# E008 Baseline-Only Checkpoint Preflight

> **COMPLETE — `BLOCKED_NO_ELIGIBLE_PAIR`**
>
> The baseline pilot is complete. E008 swaps remain unexecuted.

## Objective

Determine prospectively whether existing intermediate checkpoints from the
matched clean-room EDM-1K and EDM-50K runs can form a nondegenerate model pair
for a future E008 intervention. This preflight evaluates complete no-swap
trajectories only. It is not E008 execution and cannot generate a target,
control, or donor-model condition.

## Frozen Candidate Pools

Inventory every file matching the EDM snapshot convention, plus malformed
files beginning with `network-snapshot-`, under:

```text
EDM-1K  /home/xggh8/data/zw-lab/exp_004_standard_edm_n1000_40000kimg_20260415/
EDM-50K /home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/
```

Every candidate receives a path, filename, SHA-256, duration, size,
modification time, architecture identity, subset size, training-config source,
EMA status, conditioning status, and loadability status. Accepted and rejected
records remain visible. The inventory and its hash are frozen before pilot
inference; later checkpoint discovery cannot alter the active pool.

## Pilot And Evaluator

- Pilot seeds: exactly `10000..10127` (128 independent per-seed latents).
- Reserved confirmatory seeds: `0..255`; the preflight cannot use them.
- Sampling: 18 pure-Euler denoiser calls on the frozen EDM schedule, without
  churn or Heun correction and with `class_labels=None`.
- Output representation: unclamped, unquantized NCHW values in `[-1,1]`.
- Reference: the frozen E005 clean-room CIFAR-10 1K subset.
- Distance: exact float64 Euclidean pixel distance after channel-major
  flattening. Generated samples move to CPU before nearest-neighbor evaluation;
  direct float64 differences determine squared distances, reference subset
  position breaks exact ties, and square roots preserve Euclidean `d1NN` and
  `d2NN` values.
- Memorized: strictly `d1NN < d2NN / 3`.

Each checkpoint supplies all 18 denoiser calls. The command line has no donor
checkpoint or swap-window arguments, and the output schema has no swap fields.

## Prospective Eligibility And Selection

A checkpoint is eligible if and only if its successful 128-seed pilot has
`13..115` memorized samples, corresponding to the frozen inclusive rate range
`[0.10,0.90]`. Clopper-Pearson 95% intervals are descriptive and do not alter
eligibility.

If both roles have eligible candidates, select the cross-role pair minimizing
the absolute pilot-rate difference. Break an exact tie by the lexicographic
pair `(edm_1k_sha256, edm_50k_sha256)`. If either role has no eligible
candidate, return `BLOCKED_NO_ELIGIBLE_PAIR`; do not relax the interval or
start training.

## Execution Stages

```bash
sbatch scripts/e008_checkpoint_preflight.slurm inventory
sbatch scripts/e008_checkpoint_preflight.slurm smoke
sbatch scripts/e008_checkpoint_preflight.slurm full
```

The launcher requires explicit `E008_REPO_ROOT` and `E008_REPO_COMMIT`
contracts. Inventory and smoke use separate outputs. Full mode resumes only
against the frozen pool and does not regenerate completed checkpoint/seed
records.

## Required Outputs

Compact outputs are imported under `results/experiment_08_preflight/` and
figures under `figures/experiment_08_preflight/`. The full external run
directory remains the provenance source.

## Smoke-Recovery Record

- Job `15623452` failed before execution at the repository commit guard.
- Job `15623680` failed the original exact row-comparison smoke guard; no
  checkpoint-seed row was persisted.
- Diagnostic job `15623703` established byte-identical generated samples,
  identical neighbor indices and memorization decisions, and only a maximum
  `3.552713678800501e-15` GPU distance difference. It intentionally failed
  after writing the diagnostic.

This evidence motivated deterministic CPU distance evaluation. The frozen
config, checkpoint pool, pilot seeds, eligibility rule, and no-swap boundary
were not changed.

## Completed Pilot

Slurm job `15625456` completed with exit code `0:0` at executed commit
`b15b60e2df6191bbf1dc865ce6f2bc22e87141a6`. All `5,376` expected rows were
successful. Six EDM-1K checkpoints were eligible, while all 21 EDM-50K
checkpoints produced `0/128`. The frozen outcome is
`BLOCKED_NO_ELIGIBLE_PAIR`; no pair was selected and no swap or confirmatory
evaluation was run.

See the [validated results](experiment_08_checkpoint_preflight_results.md).
