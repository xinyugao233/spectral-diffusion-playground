# Frequency-Resolved Pipeline Audit

Audit date: 2026-08-04

Audited branch: `e009-stage-a-protocol`

Audited HEAD: `1b78e1f49ee2e3f22d816ec38d596eadd425fb88`

## 1. Executive Verdict

**PASS.** The intended frequency-resolved scientific pipeline is implemented
correctly through target selection. E004B computes low- and high-frequency
geometry independently, selects the low target `{8}` only from the low curves,
and selects the high target `{9,10}` only from the high curves. E005 residual
transition windows remain distinct measurements. E008 separately specifies
whole-denoiser swaps over the low- and high-derived temporal targets.

The causal pipeline is not complete. E008 is blocked and unexecuted because no
eligible cross-role model pair exists. E009 Stage A training is running to seek
an eligible 5K or 10K larger-data checkpoint. No frequency-specific causal
claim is supported.

## 2. Four-Curve Implementation Audit

**PASS.** `create_frequency_mask()` in
`src/spectral_diffusion_playground/filters.py` uses an inclusive centered
circular mask and retains DC. `_band_mask()` in
`src/spectral_diffusion_playground/frequency_restricted_geometry.py` defines
the high mask as the exact complement. `project_to_band()` applies a
channelwise two-dimensional orthonormal FFT and returns the real spatial
projection.

`band_projector_rank()` counts retained Fourier coefficients per channel after
checking conjugate symmetry. For these real-image, conjugate-symmetric masks,
that count is the real subspace rank, not the flattened storage dimension.

| Cutoff | Low rank | High rank | Sum |
| ---: | ---: | ---: | ---: |
| `r=3` | 87 | 2,985 | 3,072 |
| `r=4` | 147 | 2,925 | 3,072 |
| `r=5` | 243 | 2,829 | 3,072 |

The validation record
`results/experiment_04b/frequency_restricted_geometry_validation.json` reports
maximum reconstruction error `9.71445146547012e-16` and maximum Parseval error
`1.1368683772161603e-12`. Tests freeze reconstruction tolerance `2e-15` and
Parseval tolerances `rtol=2e-15`, `atol=2e-12` in
`tests/test_frequency_restricted_geometry.py`.

`evaluate_frequency_restricted_geometry()` and `_evaluate_projected_band()`
compute all four quantities separately:

```text
C_sigma^low, W_sigma^low, C_sigma^high, W_sigma^high.
```

The frozen configuration is
`configs/e004b_frequency_restricted_geometry.json`: the first 1,000 canonical
CIFAR-10 training and test images, `[-1,1]` normalization, the exact 18-point
EDM sigma schedule, seed `0`, four posterior draws, eight coverage draws, 500
hierarchical-bootstrap replicates, and the same projected full-dimensional
Gaussian draws across bands and cutoffs. The low and high panels are separate:

- `figures/experiment_04b/low_frequency_coverage_and_posterior.png`
- `figures/experiment_04b/high_frequency_coverage_and_posterior.png`

## 3. Low-Target Selection Audit

**PASS.** `summarize_targets()` receives band-specific classifications and
does not combine bands or fill gaps. The primary rule requires both 95% lower
confidence bounds to satisfy `q_C=q_W=0.8`. The low target is derived only
from `C_sigma^low` and `W_sigma^low`.

Frozen primary result: index `{8}`, sigma `3.256821519765537`. The low target
remains `{8}` at sensitivity cutoffs `r=3,4,5`. Evidence:
`results/experiment_04b/band_target_summary.json` and
`results/experiment_04b/cutoff_sensitivity.json`.

## 4. High-Target Selection Audit

**PASS.** The high target is derived only from `C_sigma^high` and
`W_sigma^high` under the same lower-confidence-bound rule.

Frozen primary result: indices `{9,10}`, sigma values
`{1.9233398370400518, 1.088170636545279}`. It remains `{9,10}` at `r=3,4`;
at `r=5`, the lower-bound result narrows to `{10}` while the point-estimate
result remains `{9,10}`. This sensitivity is reported and does not revise the
primary `r=4` target.

## 5. Geometry-Target Versus Residual-Window Distinction

**PASS.** The repository keeps the following objects distinct:

| Object | Indices | Selection source |
| --- | --- | --- |
| E004B low geometry candidate | `{8}` | Low-subspace coverage and posterior lower bounds |
| E004B high geometry candidate | `{9,10}` | High-subspace coverage and posterior lower bounds |
| E005 low residual transition | `5..11` | Independent 20%-to-80% residual-recovery rule |
| E005 high residual transition | `11..14` | Independent 20%-to-80% residual-recovery rule |

E004B geometry candidates, not E005 residual windows, define the proposed E008
target intervals. E005 provides descriptive spectral context only. This is
frozen in `results/canonical_experiment_pipeline.json`,
`results/region_definition_registry.json`, and
`docs/canonical_experiment_pipeline.md`.

## 6. Swap-Condition Audit

**PASS (designed); NOT EXECUTED.** E008 specifies independent conditions:

| Derived condition | Pre-control | Target | Post-control |
| --- | --- | --- | --- |
| Low frequency | `7..7` | `8..8` | `9..9` |
| High frequency | `7..8` | `9..10` | `11..12` |

The controls were specified before swap results, are adjacent and width
matched, and remain within the valid `0..17` index range. Within each
comparison, controls do not overlap its target. The protocol never merges,
intersects, averages, or fills the gap between low and high targets. Evidence:
`docs/experiment_08_frequency_geometry_swap_protocol.md`, section “Proposed
Conditions.”

The low post-control `{9}` overlaps the independently defined high target
`{9,10}`. This is transparent and scientifically interpretable because the two
band-derived comparisons are separate; no protocol rule requires controls
from one comparison to avoid the other comparison's target.

## 7. Whole-Denoiser Versus Frequency-Component Intervention

**PASS.** E008 proposes temporal swaps of the **whole denoiser** over intervals
derived from frequency-space geometry. It does not swap Fourier coefficients,
projected denoiser outputs, or sampler-state frequency components. Therefore,
even a future positive E008 result would establish interval influence under a
whole-network intervention, not frequency-component-specific causality.

## 8. Baseline-Pair Eligibility Gate

**PASS.** The completed baseline-only preflight has outcome
`BLOCKED_NO_ELIGIBLE_PAIR`. It evaluated 42 checkpoints and 5,376 pilot records
without swaps. Six EDM-1K checkpoints met the frozen inclusive eligibility
range `13..115 / 128`; all 21 EDM-50K checkpoints produced `0/128`.

Evidence:

- `results/experiment_08_preflight/preflight_outcome.json`
- `results/experiment_08_preflight/preflight_validation.json`
- `docs/experiment_08_checkpoint_preflight_results.md`

E008 remains blocked and unexecuted. It may start only after E009 yields an
eligible larger-data checkpoint, an eligible EDM-1K checkpoint is paired with
it under the frozen matching rule, both checkpoint hashes are committed, and
a no-swap pair smoke test passes. As of 2026-08-04, Slurm array `15673597` is
running the frozen 2K/5K/10K Stage A training jobs; no E009 checkpoint pilot or
pair selection has occurred.

## 9. Figure And README Audit

**PASS after documentation clarification.** The README shows the full-space
E004A curves, separate low and high E004B panels, selected candidate regions,
E005 residual curves, historical executed E006 results, E008 preflight
results, and a visibly labeled planned E008 target/control diagram. Planned
E008 interventions are explicitly marked unexecuted.

No plot overlays are used as the decision source: the separate E004B low and
high panels make each band-specific classification auditable. The combined
comparison figure is supplementary.

## 10. Scientific Claims That Are Currently Justified

- **PASS:** Separate low/high coverage and posterior-weight curves exist under
  the frozen clean-room configuration.
- **PASS:** Separate geometry-derived candidates `{8}` and `{9,10}` were
  selected under the frozen lower-confidence-bound rule.
- **PASS:** E005 independently observed low residual transition `5..11` and
  high residual transition `11..14`.
- **PASS:** Separate whole-denoiser E008 target/control comparisons have been
  designed.
- **PASS:** The existing checkpoint pool cannot supply an eligible cross-role
  pair under the frozen baseline gate.

## 11. Scientific Claims That Must Not Be Made

- A confirmed or causal frequency-specific memorization danger zone.
- A frequency-component intervention: E008 swaps the whole denoiser.
- That E005 selected the E004B geometry candidates.
- That E006 executed or validated the E004B targets.
- That the low/high target ordering is caused by frequency alone; projector
  rank, covariance, and energy differ substantially.
- That E008 has run, that E009 has selected a pair, or that confirmatory seeds
  have been used.

## 12. Exact Files Requiring Changes

Only presentation files required clarification:

- `README.md`: stale E009 state, implicit E008 controls, and candidate-region
  terminology.
- `docs/README.md`: current E009 operational state and audit navigation.
- `docs/frequency_resolved_pipeline_audit.md`: this durable audit record.

No source code, configuration, result, manifest, CSV, JSON, PNG, seed,
checkpoint-pool, or eligibility-rule change is required.

## 13. Recommended Minimal Corrections

The minimal corrections have been applied locally and remain uncommitted:

1. State the exact low/high E008 target and control indices in the README.
2. Mark the E008 intervention diagram as planned and unexecuted.
3. Replace headline wording with “geometry-derived candidate region.”
4. Update E009 presentation text to say Stage A training is running while its
   checkpoint pilot and pair selection remain unexecuted.

No numerical artifact was regenerated or overwritten.

## Truth Table

| Item | Status | Evidence |
| --- | --- | --- |
| Low coverage curve | PASS | `frequency_restricted_geometry.py`; low E004B figure and CSV |
| Low posterior-weight curve | PASS | Same implementation and low E004B panel |
| High coverage curve | PASS | High projection path and high E004B panel |
| High posterior-weight curve | PASS | Same implementation and high E004B panel |
| Low target identified separately | PASS | `band_target_summary.json`: `{8}` |
| High target identified separately | PASS | `band_target_summary.json`: `{9,10}` |
| Low swap condition designed | PASS | E008 protocol: `7 / 8 / 9` |
| High swap condition designed | PASS | E008 protocol: `7..8 / 9..10 / 11..12` |
| Low swap executed | NOT EXECUTED | E008 outcome and validation records |
| High swap executed | NOT EXECUTED | E008 outcome and validation records |
| Eligible cross-role pair frozen | FAIL | `BLOCKED_NO_ELIGIBLE_PAIR` |
| Frequency-specific causal claim supported | FAIL | No valid E008 intervention exists |
