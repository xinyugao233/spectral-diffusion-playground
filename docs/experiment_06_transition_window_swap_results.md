# Experiment 06: Transition-Window Swap Results

## Status

- Commit: `ae0febb9b983c50c5946d61423fda72358887523`
- Slurm job: `15511459`
- State: `COMPLETED`, exit code `0:0`
- Runtime: `986.67` seconds
- Scientific outcome: **INCONCLUSIVE**

The run completed successfully. The scientific result is `INCONCLUSIVE`
because the frozen baseline-degeneracy safeguard was triggered.

## Validation

- Conditions: 18
- Per-sample rows: 4608
- Nearest-neighbor rows: 9216
- Paired comparisons: 16
- Failure rows: 0
- Finite nearest-neighbor records: True
- Stable row order: True
- Unique per-sample keys: True
- Validation status: **pass**

All 15 manifest-recorded artifacts passed SHA-256 verification. The 13 compact
artifacts imported into the repository match the frozen source byte for byte.

## Baselines

- EDM-1K no-swap memorized count: 247 / 256
- EDM-50K no-swap memorized count: 0 / 256

The zero EDM-50K baseline is degenerate under the preregistered rule, so the
protocol prevents a directional conclusion.

## Descriptive Findings

### Low-frequency transition window

- EDM-1K base with EDM-50K donor:
  96.48% to
  0.39%
  (`-96.09` percentage-point difference);
  influential: `True`.
- EDM-50K base with EDM-1K donor:
  0.00% to
  79.30%
  (`+79.30` percentage-point difference);
  influential: `True`.

### High-frequency transition window

- EDM-1K base with EDM-50K donor:
  `-39.84` percentage-point difference;
  influential: `False`.
- EDM-50K base with EDM-1K donor:
  `+4.69` percentage-point difference;
  influential: `False`.

The low-transition window passed the frozen influence rule in both directions.
The high-transition window did not pass it in either direction.

## Interpretation

The preregistered result remains **INCONCLUSIVE**. Descriptively, the results
isolate the E005 low-frequency transition window as the tested window most
strongly associated with changes in the pixel-space memorization criterion.

This does not justify overriding the frozen decision rule or assigning a
causal memorization label to any sigma range.

## Storage

Compact summaries are stored in `results/experiment_06/`, and figures are
stored in `figures/experiment_06/`.

Large generated-sample and per-sample artifacts remain at:

`/home/xggh8/data/zw-lab/e006_transition_window_swaps`
