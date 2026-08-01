# Documentation Index

Protocols freeze decisions before execution. Result documents report the
validated run without changing those decisions. Provenance documents connect
external data and checkpoints to exact hashes and commits.

Start with the [canonical experimental pipeline](canonical_experiment_pipeline.md).
It freezes the order E004 cutoff -> E004A full-space geometry -> E004B
frequency-restricted geometry -> E005 spectral interpretation -> historical
E006 exploration -> proposed E007/E008 interventions.

## E004: Operational CIFAR-10 Cutoff

1. [Frozen cutoff-selection protocol](experiment_04_frequency_cutoff_protocol.md)
2. [Reviewer instructions](experiment_04_reviewer_instructions.md)
3. [Final single-reviewer decision](experiment_04_frequency_cutoff_decision.md)

Result: operational reference `r = 4`, primary sensitivity `r = 3, 5`, and
optional extended sensitivity `r = 6`.

## E004A: Paper Coverage-Concentration Geometry

1. [Paper definition and source audit](paper_geometry_source_audit.md)
2. [Frozen clean-room protocol](experiment_04a_paper_geometry_protocol.md)
3. [Validated clean-room results](experiment_04a_paper_geometry_results.md)

Result: the paper's qualitative small-, medium-, and large-noise geometry is
recovered in a deterministic clean-room setup. The sampled primary-threshold
high-high region is `sigma in {2,3,4,5}`; it is not defined by E005.
Direct evaluation on the exact E006 schedule selects indices `{8,9}` under
both the point-estimate and 95% lower-bound rules at `q_C=q_W=0.8`.

## E004B: Frequency-Restricted Geometry

1. [Frozen clean-room protocol](experiment_04b_frequency_restricted_geometry_protocol.md)
2. [Validated clean-room results](experiment_04b_frequency_restricted_geometry_results.md)

Result at `r=4`: low-band target `{8}` and high-band target `{9,10}` under
the primary lower-confidence-bound rule. These are subspace coverage and
posterior-concentration curves, not E005 denoising residual energies.
E004B is complete as a descriptive result. Its low/high ranks are `147/2925`,
so rank, covariance, and energy differences prevent attribution to frequency
organization alone without matched controls.

## E005: Spectral Residual Curves

1. [Frozen residual-energy protocol](experiment_05_spectral_residual_protocol.md)
2. [Clean-room model and checkpoint provenance](experiment_05_clean_room_models.md)
3. [Validated residual-curve results](experiment_05_spectral_residual_results.md)

Result: an ordered low-frequency then high-frequency residual transition under
the frozen clean-room setup.

## E006: Historical Spectral-Window Swaps

1. [Frozen whole-denoiser swap protocol](experiment_06_transition_swap_protocol.md)
2. [Validated swap results](experiment_06_transition_window_swap_results.md)

Formal outcome: `INCONCLUSIVE` because the EDM-50K no-swap baseline was
degenerate at `0/256`. Descriptively, the E005 low-frequency spectral
transition passed the frozen influence test in both directions; the E005
high-frequency spectral transition did not.

## Region Definitions And Proposed E007

1. [Canonical region-definition audit](danger_zone_definition_audit.md)
2. [Blocked geometry-aligned E007 protocol](experiment_07_geometry_aligned_swap_protocol.md)

The audit distinguishes the literature-derived paper medium reference, E005
spectral transitions, and E004A clean-room geometric high-high set. E007 is
`PROPOSED — BLOCKED BY KNOWN BASELINE DEGENERACY` and does not alter
historical E006. The target remains indices `8..9`, but the historical E006
pair cannot support an informative bidirectional decision because its EDM-50K
baseline is already `0/256`.

The main scientific chain remains incomplete until E007 or an equivalent
preregistered geometry-aligned swap is executed with a nondegenerate model
pair. E004A selects the candidate region; E005 interprets its spectral
location; historical E006 does not substitute for that final test.

## Proposed E008

[E008](experiment_08_frequency_geometry_swap_protocol.md) freezes only the
design for whole-denoiser swaps over the E004B low target `8..8` and high
target `9..10`. It is `PROPOSED — NOT EXECUTED`; the primary bidirectional
design is blocked by the same known historical EDM-50K `0/256` baseline.

## Reading Order

New readers should start with the [root README](../README.md), then the
canonical pipeline, E004 decision, E004A source audit and results, E005
results, historical E006 results, and blocked E007 protocol.
Reviewers auditing a specific experiment should read its protocol before its
result document and then inspect the linked compact [`results/`](../results/)
and [`figures/`](../figures/) directories.
