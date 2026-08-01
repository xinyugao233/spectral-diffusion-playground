# Region-Definition Audit

## Purpose

This audit separates every noise interval used by E004A, E005, and E006. The
objects have different sources and must not silently replace one another.
Machine-readable definitions are in
[`results/region_definition_registry.json`](../results/region_definition_registry.json).
Their canonical experimental order is frozen in
[`canonical_experiment_pipeline.md`](canonical_experiment_pipeline.md).

## Canonical Definitions

| Name | Indices | Sigma values or range | Source | Selection rule | Frozen before outcomes? | Intended role | Allowed interpretation | Prohibited interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paper-reported medium reference (historical key `paper_medium_reference`) | `6..13` | `8.400935..0.139516` | Paper Table 1 / Figure 10 | Literature compatibility convention selected before E006; not computed from local geometry | Yes, before E006 | Historical comparison condition | A paper-reported medium-window reference under one of several conventions | A unique paper boundary or an E004A-derived region |
| E005 low-frequency spectral transition (historical key `low_transition`) | `5..11` | `12.910082..0.585348` | E005 low-band residual energy at `r=4` | Frozen 20%-to-80% normalized-recovery rule | Yes, before E006 | Spectral transition and E006 swap window | Interval where the measured low-band residual moves through its operational transition | Original danger zone, learned-structure interval, or proven memorization interval |
| E005 high-frequency spectral transition (historical key `high_transition`) | `11..14` | `0.585348..0.0599473` | E005 high-band residual energy at `r=4` | Frozen 20%-to-80% normalized-recovery rule | Yes, before E006 | Spectral transition and E006 swap window | Interval where the measured high-band residual moves through its operational transition | Original danger zone, learned-detail interval, or proven memorization interval |
| E005 combined spectral transition (historical key `combined_transition`) | `5..14` | `12.910082..0.0599473` | E005 low/high windows | Inclusive union of the two spectral windows | Yes, before E006 | Secondary E006 context | Union of the two operational spectral transitions | Geometry-derived region or positive memorization finding |
| E004A clean-room geometric high-high region | `{8,9}` on the E006 grid | `{3.2568215, 1.9233398}` | E004A coverage and posterior concentration evaluated on the E006 schedule | Evaluated points satisfying `C_sigma >= 0.8` and `W_sigma >= 0.8`; no interpolation or gap filling | Computed after historical E006 | Clean-room candidate geometric region and proposed E007 target | Threshold-dependent high-high points under the frozen clean-room configuration | Universal paper boundary, exact original result, or a condition tested by historical E006 |

The E004A 95% lower-confidence-bound classification also yields indices
`{8,9}`. This agreement is specific to the frozen subset, seed, draws, grid,
and thresholds; it is not robustness evidence across alternative clean-room
configurations.

## Historical E006 Clarification

E006 tested spectral-aligned windows and a literature-derived paper medium
reference. It did not preregister a window from locally computed coverage and
posterior-weight curves, because E004A did not yet exist.

The low-frequency spectral transition produced the strongest descriptive E006
swap effect, but E006 remained formally `INCONCLUSIVE` and did not identify a
memorization danger zone. The newly computed E004A set cannot be retroactively
substituted for an E006 condition or used to revise that outcome.

The geometry-derived set remains a valid proposed E007 target, but the
historical E006 model pair cannot support the proposed bidirectional decision:
its EDM-50K no-swap baseline is already known to be `0/256`. E007 is blocked
until a prospectively selected model pair passes a baseline-only
nondegeneracy preflight.

## Overlap On The Frozen E006 Grid

The E004A point-estimate and lower-bound high-high indices are both `{8,9}`.
They are contained in:

- the paper-reported medium reference `6..13`;
- the E005 low-frequency spectral transition `5..11`;
- the E005 combined spectral transition `5..14`.

They do not overlap the E005 high-frequency spectral transition `11..14`.
These set relationships are descriptive bookkeeping, not causal or semantic
equivalences.

## Terminology Policy

Human-facing text uses **paper-reported medium reference**, **E005
low-frequency spectral transition**, **E005 high-frequency spectral
transition**, and **E004A clean-room geometric high-high region**. Frozen
machine-readable E006 keys remain unchanged for provenance.

The phrase **candidate danger region** is reserved for the E004A geometric
high-high set and must appear with `q_C=q_W=0.8` plus the clean-room limitation.
No interval in this repository is a universal or exact paper-defined boundary.

E004A alone selects the clean-room geometry-defined candidate target. E005
interprets where that target falls relative to low/high residual transitions;
it does not define the target. Historical E006 tested spectral and literature
reference windows, not the E004A-selected `8..9` interval. E007 is the blocked
final geometry-aligned intervention.
