# Experiment 4 Reviewer Instructions

## Purpose

Experiment 4 asks whether one centered radial Fourier cutoff can serve as a
defensible operational split on 32 x 32 CIFAR-10 images. The bands are
frequency-scale proxies, not semantic definitions of general structure or fine
detail.

The scientific protocol is frozen in
[`experiment_04_frequency_cutoff_protocol.md`](experiment_04_frequency_cutoff_protocol.md).
These instructions do not replace or modify it.

## Independence

- Reviewer A completes `results/experiment_04_reviewer_A.csv`.
- Reviewer B completes `results/experiment_04_reviewer_B.csv`.
- Reviewers work independently and must not see the other reviewer's scores
  until both files are complete.
- Codex generates the packet and validates completed files but does not serve
  as either reviewer.
- No denoiser output or Experiment 5 curve may be consulted.

## Required Coverage

Score every frozen image at every cutoff:

```text
r in {2, 3, 4, 5, 6, 8}
```

Each reviewer file contains 20 images x 6 cutoffs = 120 rows. No image or
cutoff may be omitted, including unattractive, ambiguous, or failed examples.
Do not add, remove, reorder, or replace images.

The five class-grouped montages under `figures/` collectively contain the
complete review set. For each image:

- the original and low-pass reconstructions are mapped from the computational
  `[-1,1]` domain to display RGB;
- the high-pass panels use one symmetric per-image 99.5th-percentile scale
  shared across all six cutoffs;
- zero high-pass value maps to neutral gray;
- the high-pass rescaling is display-only and does not affect measurements.

## Score Scales

Enter an integer `0`, `1`, or `2` in each score column.

### `layout_score`

- `0`: coarse spatial layout is absent or misleading.
- `1`: layout is recognizable but substantially degraded.
- `2`: layout is clearly retained.

### `identity_score`

- `0`: important object identity is lost.
- `1`: identity is ambiguous but plausible.
- `2`: identity is clearly retained.

### `high_localization_score`

- `0`: the high band is dominated by global silhouette or broad intensity
  structure.
- `1`: the high band mixes global and localized content.
- `2`: the high band primarily contains localized edges and texture.

Set `ambiguous` to exactly `true` or `false`. Every score of `0` or `1`, and
every row marked `ambiguous=true`, requires a nonempty `comment`. Use
`failure_category` to record a concise category such as:

```text
layout_lost
identity_lost
global_content_in_high_band
mixed_or_unclear
display_limitation
other
```

Do not resolve ambiguity by changing a score after inspecting the other
reviewer's file.

## Frozen Qualification Rule

A cutoff qualifies for one reviewer only when:

- at least 16 of 20 images have `layout_score >= 1`;
- at least 14 of 20 images have `identity_score >= 1`;
- at least 16 of 20 images have `high_localization_score >= 1`;
- every class has at least one image with both `layout_score >= 1` and
  `identity_score >= 1`;
- all numerical reconstruction and energy gates pass.

A cutoff qualifies overall only if it qualifies separately for both reviewers.
The frozen selection rule then chooses the smallest interior candidate in
`{3,4,5,6}` that qualifies overall and whose immediately higher candidate also
qualifies overall. Its immediately lower and higher candidates are retained
for sensitivity analysis.

If this rule is not satisfied, the result is `no_selection`. Reviewers do not
choose a preferred cutoff directly.

## Completion Procedure

1. Confirm you have the complete five-file montage series.
2. Complete only your assigned CSV.
3. Score all 120 rows.
4. Add required comments and ambiguity flags.
5. Do not open the other reviewer's CSV.
6. Return the completed file for mechanical validation.

Blank templates are intentional. No cutoff selection should be run until both
independent files are complete and validated.

