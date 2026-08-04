# E009: Staged Intermediate-Dataset Model Design

> **STAGE A PROTOCOL FROZEN — TRAINING NOT YET STARTED**
>
> E009 designs a nondegenerate model pair. It does not execute E008 swaps.

## Objective

Find a larger-data EDM checkpoint with nonzero, nondegenerate baseline
memorization while minimizing training compute and preserving prospective
decision rules. E008 found six eligible EDM-1K checkpoints but every available
EDM-50K checkpoint was `0/128`; extending that same 50K trajectory is therefore
not the primary next control.

## Stage A Training

Train three unconditional matched EDM models in one parallel Slurm array:

| Dataset | Run | Budget | Snapshots |
| ---: | --- | ---: | --- |
| 2K | `e009_edm2k_12000kimg` | 12,000 kimg | `0, 1K, ..., 12K` |
| 5K | `e009_edm5k_12000kimg` | 12,000 kimg | `0, 1K, ..., 12K` |
| 10K | `e009_edm10k_12000kimg` | 12,000 kimg | `0, 1K, ..., 12K` |

The array is exactly `0-2%3`. Every run uses training seed `0`, matching the
existing clean-room runs and holding initialization/randomness fixed across
dataset sizes. The architecture, `EDMLoss`, Adam optimizer, learning rate,
batch size, EMA, dropout, precision, augmentation, and preprocessing match the
existing EDM-1K/EDM-50K models. Operational differences are the run name,
subset archive, subset size, 12K duration, 1K snapshot/state cadence, and
persistent scratch path.

Stage A does not include 20K or any automatic 40K extension. Its maximum
authorized training budget is three L40S tasks at 24 hours each (72 GPU-hours
hard cap; approximately 42 GPU-hours expected from the matched 50K timing).
Persistent checkpoint storage is capped at 12 GB.

## Frozen Nested Subsets

The complete E005 clean-room 1K manifest is retained unchanged. For each
class, E009 removes the anchor indices from the canonical CIFAR-10 class pool,
shuffles the remaining indices once with `RandomState(9 + class_id)`, takes
the required prefix, combines it with the anchor, and sorts the final indices.
This gives:

```text
existing 1K subset ⊂ 2K subset ⊂ 5K subset ⊂ 10K subset
```

Each E009 set is exactly class balanced: 200, 500, or 1,000 images per class.
The source PNG bytes are copied unchanged into deterministic subset ZIPs.

| Size | Text manifest SHA-256 | Little-endian int64 SHA-256 | Archive SHA-256 |
| ---: | --- | --- | --- |
| 2K | `badb7ccf1fc797f016c6fac18b735c2fee121e95fbb4da1de5d8c23c4dd72a4c` | `8d9bd13a9d2b5c6737b164bf279bf55cc25fd7397f72fc6b21445a46cb56700e` | `98e1196377fcb3abe14a0b0dc99a58c3e047a28b5aa88e55d08660b5745887e2` |
| 5K | `9f41e233172109ef3995a2ec621d880c7bd4fe08257be3b39cea4f820664a5e3` | `712d5afecedfad56edc48030fa8e0cb83e6264942507c74928eebffc60711cc4` | `1e96a4f7a701bd067f71c725bbe83f1dcd65a750b310f206eee878ce2c07355a` |
| 10K | `6e16d015d518b499775f6389c4e72f0fd14dd12d18efb7fd3d26b865264a220d` | `db3ea573af4e41d3e542325f58021a9582e103bfa852a99554ae2c84850380f4` | `f128bfbbb213769f4f087b8c6e46380f29d861152e470e98b985de52f2db4439` |

The committed machine-readable manifest is
[`data/e009_nested_subsets_manifest.json`](../data/e009_nested_subsets_manifest.json).
Subset preparation job `15672797` completed `0:0`; it performed no training.
The ZIP archives remain external at
`/home/xggh8/data/zw-lab/e009_stage_a_subsets/`.

## Stage A Baseline Pilot

After all training tasks complete, evaluate every one of the expected 39
checkpoints on exactly 128 no-swap seeds:

```text
pilot seeds:            20000..20127
E008 pilot seeds:       10000..10127 excluded
confirmatory seeds:     0..255 excluded and untouched
expected records:       39 x 128 = 4,992
memorization criterion: d1NN < d2NN / 3
eligibility:            13..115 memorized out of 128, inclusive
```

Use the same 18-call pure-Euler sampler, clean-room 1K nearest-neighbor
reference, unclamped/unquantized output representation, and deterministic CPU
`float64` nearest-neighbor evaluator validated in E008. The pilot is
baseline-only: donor models, swap windows, targets, and controls are forbidden.

## Deterministic Pair Selection

The small-data candidates are the six eligible EDM-1K checkpoints frozen by
E008. The larger-data candidates must come from 5K or 10K Stage A runs; 2K is
tracked diagnostically but does not satisfy the preregistered minimum 5x
dataset-size contrast.

Among eligible cross-role pairs:

1. minimize absolute pilot memorization-rate difference;
2. if tied, prefer the larger new dataset size;
3. if still tied, minimize the lexical tuple
   `(new_checkpoint_sha256, edm_1k_checkpoint_sha256)`.

No uncertainty interval changes eligibility or selection.

## Frozen Stopping Rules

- **Any eligible 5K or 10K checkpoint:** apply the pair rule, freeze one pair,
  mark E009 complete, and prepare a separate E008 execution review.
- **Only 2K eligible:** record the best 2K checkpoint as provisional but do not
  unblock E008; trigger Stage B because the required contrast is at least 5x.
- **No new checkpoint eligible:** trigger Stage B.
- **Task failure:** resume only that exact run from its latest valid state;
  never replace the seed, subset, or config based on observed metrics.

Stage B is specified but not authorized automatically. If triggered, it adds
a 20K run to 12K kimg and extends exactly one of the 5K/10K runs to 20K kimg.
Choose the extension by the minimum distance of any observed count to the
inclusive interval `[13,115]`; ties prefer larger dataset size, then config
SHA-256. Stage B requires a separate review and Slurm authorization. No run is
automatically extended to 40K.

## Execution And Storage

- Smoke: one 2K model for 1 kimg, one L40S, 30-minute limit.
- Stage A: three parallel one-L40S tasks, 8 CPUs, 48 GB RAM, 24-hour limit.
- Pilot: one L40S, 8 CPUs, 64 GB RAM, 2-hour limit.
- Checkpoints and states are written first to lab PixStor persistent scratch;
  completed artifacts are staged under `/home/xggh8/data/zw-lab/`.
- All caches and temporary files use the job-specific PixStor temporary root;
  no large artifact is written to `$HOME` or Git.

The launcher refuses wrong commits, dirty checkouts, hash mismatches,
unexpected outputs, implicit resume, or non-Slurm execution. The smoke must
produce finite training statistics and readable `0`/final snapshots before
the Stage A array may be submitted.

## Current Gate

At protocol freeze, no E009 training, Stage A pilot, confirmatory inference,
model-pair selection, or E008 swap has occurred. Training is authorized only
from the exact committed protocol/config identities after the smoke passes.
