# Results Index

This directory contains compact, machine-readable outputs intended for code
review, plotting, and provenance checks. Large raw arrays, generated samples,
downloaded datasets, and checkpoints remain outside Git.

The [canonical pipeline registry](canonical_experiment_pipeline.json) records
the ordered E004-E007 roles, the geometry-selected target, the distinction
between selection and spectral interpretation, and the current E007 blocker.

## E004: Operational Frequency Cutoff

Files stored directly under `results/`:

| File | Purpose |
| --- | --- |
| `experiment_04_manifest.json` | Dataset, FFT, mask, and execution identity |
| `experiment_04_image_manifest.csv` | Frozen CIFAR-10 examples and indices |
| `experiment_04_cutoff_measurements.csv` | Per-image reconstruction diagnostics |
| `experiment_04_cutoff_energy.csv` | Retained and complementary energy fractions |
| `experiment_04_cutoff_review.csv` | Canonical review-row ordering |
| `experiment_04_reviewer_A.csv` | Blank reviewer template |
| `experiment_04_reviewer_B.csv` | Blank reviewer template |

The templates remain blank because the planned two-independent-reviewer
scoring procedure was not completed. The final `r = 4` decision is documented
as a single-reviewer qualitative result in
[the decision record](../docs/experiment_04_frequency_cutoff_decision.md).

## E004A: Paper Coverage-Concentration Geometry

[`experiment_04a/`](experiment_04a/) contains the clean-room coverage curve,
maximum-posterior-weight curve, validation record, and provenance manifest.
The result is imported from the validated full-space gate in the shared
research-context hub; the source hashes and deviations from the unavailable
paper execution are recorded in the manifest and
[source audit](../docs/paper_geometry_source_audit.md).

Curve rows use the stable schema `sigma_index`, `sigma`, `metric`, `estimate`,
`ci95_low`, `ci95_high`, `training_examples`, `query_examples`, `shell_c`,
`subset_sha256`, `seed`, and `status`. All source values are finite.

[`experiment_04a_reproduction/`](experiment_04a_reproduction/) contains a
fresh end-to-end local computation from the hash-verified CIFAR-10 archive.
It adds Monte Carlo standard errors, a per-sigma committed-versus-fresh
comparison, complete execution provenance, and regenerated figures. Both
metrics agree at every sigma under the tolerance frozen before execution.

The three `e006_grid_geometry.*` files in
[`experiment_04a/`](experiment_04a/) evaluate both E004A quantities directly
on the exact 18-point E006 schedule. The point-estimate and lower-bound
high-high sets are both indices `{8,9}` at `q_C=q_W=0.8`; no interpolation or
gap filling is used. Canonical distinctions among all historical regions are
stored in [`region_definition_registry.json`](region_definition_registry.json).

## E005: Spectral Residual Curves

[`experiment_05/`](experiment_05/) contains aggregated curves, numerical
identity validation, transition-window extraction, and the compact provenance
manifest.

The 2,304,000-row per-sample residual table remains external at
`/home/xggh8/data/zw-lab/e005_spectral_residual_curves`. Its hash and
reproduction command are recorded in the
[E005 results document](../docs/experiment_05_spectral_residual_results.md).

## E006: Historical Spectral-Window Swaps

[`experiment_06/`](experiment_06/) contains condition summaries, paired
comparisons, the frozen `INCONCLUSIVE` outcome, validation and failure records,
qualitative-selection metadata, and the compact provenance manifest.

Generated samples, nearest-neighbor rows, and per-sample rows remain external
at `/home/xggh8/data/zw-lab/e006_transition_window_swaps`. They are excluded
from Git intentionally. See the
[E006 results document](../docs/experiment_06_transition_window_swap_results.md).

E006's internal `paper_medium_reference` key is retained for provenance. It is
a literature-derived Table 1 / Figure 10 compatibility condition, not an
E004A-derived geometry window.

## Artifact Policy

Commit an output only when it is small, deterministic, reviewable, and needed
to verify a reported conclusion. External raw artifacts must be identified by
storage path, row count or shape, SHA-256, and a reproduction command in the
corresponding manifest or results document.
