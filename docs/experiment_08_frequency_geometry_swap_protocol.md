# E008: Frequency-Specific Geometry-Aligned Swap Protocol

> **PROPOSED — NOT EXECUTED**
>
> **PRIMARY BIDIRECTIONAL DESIGN BLOCKED BY KNOWN BASELINE DEGENERACY**

## Objective

Test whether whole-denoiser swaps over the E004B low- and high-frequency
geometry targets alter the frozen memorization criterion more than
width-matched neighboring controls. E008 is a proposed intervention. It does
not reinterpret E004B geometry as causal evidence and it is not a Fourier
coefficient swap.

## Frozen Target Source

E004B selected targets at cutoff `r=4` using the 95% lower confidence bounds
of frequency-restricted Gaussian-shell coverage and maximum posterior weight,
with `q_C=q_W=0.8`:

| Role | Inclusive indices | Sigma values |
| --- | --- | --- |
| Low-band geometry target | `8..8` | `3.256821519765537` |
| High-band geometry target | `9..10` | `1.9233398370400518..1.088170636545279` |

Only evaluated points are eligible. E005 residual curves did not select or
modify these targets.

## Proposed Conditions

For each target, use the frozen 18-step pure-Euler sampler and swap the whole
denoiser at every inclusive target index. Use the base denoiser elsewhere.
Controls are immediately adjacent and width matched:

| Target | Pre-control | Post-control |
| --- | --- | --- |
| Low `8..8` | `7..7` | `9..9` |
| High `9..10` | `7..8` | `11..12` |

Run both model directions and both no-swap baselines using the same frozen
confirmatory seeds. The eventual model pair, pilot pool, seeds, baseline
eligibility rule, inference representation, nearest-neighbor reference, and
decision thresholds must be frozen before execution.

These are temporal whole-network interventions. They do not replace selected
Fourier coefficients, mix low/high network outputs, or modify the sampler
state in frequency space.

## Component Policy

- An empty E004B target produces no primary swap and is reported as not
  applicable.
- A noncontiguous target remains separate connected components. Each component
  receives its own width-matched adjacent controls.
- Gaps may never be filled to manufacture one interval.
- Sensitivity cutoffs `r=3,5` are reported separately and cannot revise the
  primary `r=4` targets after inspection.

## Known Blocker

The historical E006 model pair is not eligible for a primary bidirectional
E008 conclusion. Its EDM-50K no-swap baseline is `0/256`, so the existing
degeneracy guard forces `INCONCLUSIVE` regardless of target effects. This is a
model-pair baseline blocker, not a compute-availability blocker.

A future execution must first pass a preregistered baseline-only model-pair
preflight on pilot seeds disjoint from confirmatory seeds. No swap effect may
be inspected during pair selection. If no nondegenerate pair is available,
E008 remains blocked.

The completed preflight returned `BLOCKED_NO_ELIGIBLE_PAIR`: six EDM-1K
checkpoints were eligible, but every one of the 21 EDM-50K checkpoints was
`0/128`. The historical pair therefore cannot be replaced from the existing
matched checkpoint pools. See the
[preflight results](experiment_08_checkpoint_preflight_results.md).

The baseline-only preflight is frozen separately in
[the E008 checkpoint preflight protocol](experiment_08_checkpoint_preflight.md).
It inventories all existing matched-run snapshots before sampling, uses pilot
seeds `10000..10127`, reserves confirmatory seeds `0..255`, requires an
inclusive pilot count of `13..115`, and selects an eligible pair solely by
minimum absolute baseline-rate difference with a SHA-256 tie-break. Preparing
this preflight does not execute E008.

## Memorization Criterion And Guardrails

Unless superseded by a separately reviewed freeze, retain the E006
representation and strict criterion: unquantized, unclamped RGB tensors in
`[-1,1]`, flattened channel-major, with Euclidean distances to the frozen
clean-room 1K training subset. Count a sample as memorized only when

```text
d_1NN < d_2NN / 3.
```

The final decision rule, uncertainty procedure, seed count, and model-pair
eligibility interval must be frozen in an execution amendment before running
E008. No positive result may be described as a universal paper boundary or a
memorization danger zone without the clean-room, frequency-scale, and
intervention limitations stated nearby.

## Current Status

Baseline-only checkpoint evaluation is complete. No E008 swap, confirmatory
inference, target/control sampling, or swap result has been produced. E008 is
blocked pending a separately designed and validated nondegenerate model pair.
