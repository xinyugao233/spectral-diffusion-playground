# E010: Directional Memorization-Transfer Protocol

Status: **FROZEN BEFORE INFERENCE**

## Research Question

Can a memorizing denoiser transfer memorization behavior into a generalizing
trajectory, and can a generalizing denoiser suppress memorization in a
memorizing trajectory, when the whole denoiser is swapped over separately
identified low- and high-frequency-derived intervals?

E010 intentionally accepts asymmetric no-swap baselines. It is not the
baseline-matched E008 experiment. E008 remains **BLOCKED** and **UNEXECUTED**.

## Models

The memorizing endpoint is selected prospectively from the six E008-eligible
EDM-1K checkpoints: maximize the historical memorized count on seeds
`10000..10127`, retain the eligibility ceiling of `115/128`, then break an
exact tie by checkpoint SHA-256. This selects the 12K checkpoint at `113/128`.

The generalizing endpoint is the historical matched clean-room EDM-50K 40K
checkpoint used by E006/E008. No memorized samples were observed for it under
the frozen E006 (`0/256`) or E008 (`0/128`) seeds. This does not establish a
mathematically zero population probability.

Exact paths, hashes, baseline evidence, and compatibility records are frozen
in [`e010_model_pair_manifest.json`](../data/e010_model_pair_manifest.json).
The two networks are unconditional 32 x 32 RGB EDM networks with the same
inference interface. They are not baseline matched and differ in training-data
size and training trajectory.

## Geometry-Derived Intervals

E010 uses the E004B lower-confidence-bound targets at cutoff `r=4`:

| Band-derived role | Before | Target | After |
| --- | --- | --- | --- |
| Low frequency | `{7}` | `{8}` | `{9}` |
| High frequency | `{7,8}` | `{9,10}` | `{11,12}` |

These are E004B frequency-restricted geometry intervals, not the E005 residual
transition windows. A swap replaces the **whole denoiser** only at the listed
calls and restores the recipient immediately afterward. E010 does not swap a
frequency component and cannot establish frequency-component causality.

## Conditions And Seeds

Direction A tests suppression: the memorizing EDM-1K model is the recipient
and the generalizing EDM-50K model is the donor. Direction B tests induction:
the roles are reversed. Each direction contains a no-swap baseline and the six
before/target/after conditions, for 14 conditions total. The exact registry is
[`e010_condition_manifest.json`](../data/e010_condition_manifest.json).

All conditions use identical latent seeds `40000..40255` (256 seeds), yielding
`14 x 256 = 3,584` expected records. This range is disjoint from all earlier
frozen evaluation ranges. The exact policy is in
[`e010_seed_manifest.json`](../data/e010_seed_manifest.json).

## Sampling And Evaluation

- Pure Euler sampling on the frozen 18-point EDM schedule.
- No churn and no Heun correction.
- `class_labels=None`; outputs remain unclamped and unquantized.
- Per-seed device-local latent generation.
- Frozen CIFAR-10 1K reference subset in `[-1,1]`.
- Deterministic CPU `float64` nearest neighbors using direct differences.
- Euclidean distances with stable `(distance, reference position)` ordering.
- Strict memorization rule `d1NN < d2NN / 3`.

No training interface belongs to E010.

## Frozen Analysis

For each swap, paired seed outcomes are classified as memorized to
non-memorized, non-memorized to memorized, unchanged memorized, or unchanged
non-memorized.

For suppression condition `c`:

```text
effect(c) = mean(Y_memorizing_baseline - Y_swap_c)
```

For induction condition `c`:

```text
effect(c) = mean(Y_swap_c - Y_generalizing_baseline)
```

For each direction and band:

```text
contrast = effect(target) - mean(effect(before), effect(after))
```

A target passes only when its effect is positive, exceeds each neighboring
control, its contrast is positive, and the deterministic paired bootstrap 95%
interval for the contrast lies strictly above zero. Bootstrap settings are
100,000 resamples, RNG seed 0, and latent seed as the resampling unit. All raw
effects and intervals are reported regardless of outcome.

Permitted formal labels are the four direction-band `*_SUPPORTED` labels,
`NO_DIRECTIONAL_TARGET_OUTPERFORMS_CONTROLS`, and
`MIXED_DIRECTIONAL_EVIDENCE`. Multiple direction-band results may pass.

## Execution Gates

Before inference, the implementation must fail closed on checkpoint,
configuration, model-pair, condition, seed, geometry, reference-subset, or
execution-commit mismatch; an incompatible output directory; a non-CPU
`float64` evaluator; or an unregistered model/condition. The frozen branch
commit is supplied through `E010_REPO_COMMIT` and recorded in run provenance,
avoiding a self-referential commit hash inside committed files.

Execution proceeds only as:

1. no-inference provenance preflight;
2. isolated 28-record smoke on seeds `40000,40001`, including an exact rerun;
3. one frozen 3,584-record full run;
4. deterministic summarization and figure generation.

Successful records are never silently rerun. A resume may address only an
explicitly missing or failed `(condition, seed)` key.

## Outputs And Acceptance

The external run retains per-sample rows and generated samples. Git receives
only compact summaries, validation/provenance records, representative images,
and figures. Acceptance requires 3,584 unique explicit records, finite
distances, no silent failures, reproducible bootstrap output, exact input
hashes, and seven review figures.

## Interpretation Boundaries

E010 can provide directional whole-denoiser intervention evidence associated
with geometry-derived timing. It cannot by itself establish that a frequency
component causes memorization, that the E004B intervals are universal, or that
training-data size alone causes an observed difference.
