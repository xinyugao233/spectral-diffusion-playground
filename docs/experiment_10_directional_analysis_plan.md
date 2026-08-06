# E010 Frozen Analysis Plan

Status: **FROZEN BEFORE INFERENCE**

The analysis unit is the latent seed. Every seed is shared across all 14
conditions. Let `Y` be the strict binary decision `d1NN < d2NN / 3`.

## Directional Effects

- Suppression: `Y_mem_baseline - Y_swap`.
- Induction: `Y_swap - Y_gen_baseline`.

For each swap condition, report the mean effect and all four paired transition
counts. For each band and direction, compute target minus the arithmetic mean
of before and after control effects at the seed level.

## Uncertainty And Decision Rule

Bootstrap 100,000 paired seed resamples with NumPy RNG seed 0. The percentile
95% interval is the 2.5th and 97.5th percentiles. A direction-band target is
supported only if:

1. target effect is positive;
2. target effect exceeds both control effects;
3. target-minus-mean-controls contrast is positive;
4. contrast bootstrap lower bound is greater than zero.

No multiplicity-adjusted global claim is made. Report all four tests and all
condition effects. Floor and ceiling baselines remain visible.

## Formal Classification

Each passing result receives one of:

- `LOW_DERIVED_SUPPRESSION_SUPPORTED`
- `LOW_DERIVED_INDUCTION_SUPPORTED`
- `HIGH_DERIVED_SUPPRESSION_SUPPORTED`
- `HIGH_DERIVED_INDUCTION_SUPPORTED`

If none pass, use `NO_DIRECTIONAL_TARGET_OUTPERFORMS_CONTROLS`. If more than
one but not all pass, additionally report `MIXED_DIRECTIONAL_EVIDENCE`.

Descriptions must say "whole-denoiser swap over a [low/high]-frequency-derived
interval." They must not call an interval a danger zone or attribute causality
to an isolated frequency component.
