# Experiment 4: CIFAR-10 Frequency Cutoff Protocol

## Status

This document freezes the protocol for selecting an operational Fourier cutoff
on CIFAR-10 before any denoiser curves are examined. Experiment 4 is a
calibration study, not a model experiment.

The future implementation is part of a **paper-derived clean-room
reimplementation**. The original executed paper code was unavailable, so this
repository does not claim code-identical or numerically exact reproduction.

No Experiment 4 results exist at the time this protocol is frozen.

## Scientific Objective

Experiment 4 will select an operational cutoff \(r_\star\) for complementary
Fourier projections on 32 x 32 RGB images. Experiments 5 and 6 will use the
same projections without retuning them from model outputs.

For a residual

\[
e_\sigma = m_\sigma(X + \sigma Z) - X,
\]

the later clean-room evaluator will measure

\[
\operatorname{MSE}_{\mathrm{low},r}
= \mathbb{E}\left[\lVert P_{\mathrm{low},r} e_\sigma\rVert_2^2\right],
\]

\[
\operatorname{MSE}_{\mathrm{high},r}
= \mathbb{E}\left[\lVert P_{\mathrm{high},r} e_\sigma\rVert_2^2\right].
\]

Because the masks are exact complements under an orthonormal FFT,

\[
\operatorname{MSE}_{\mathrm{full}}
= \operatorname{MSE}_{\mathrm{low},r}
+ \operatorname{MSE}_{\mathrm{high},r}.
\]

The bands are operational frequency-scale proxies. This protocol does not
define low frequency as semantic "general structure" or high frequency as
semantic "fine detail."

## Frozen Dataset Domain

- Dataset: canonical CIFAR-10 test split.
- Shape: 32 x 32 RGB.
- Source pixels: decoded uint8 values in \(\{0,\ldots,255\}\).
- Computational representation: float64 in \([-1,1]\), computed as
  \(x = 2(u / 255) - 1\).
- Preprocessing: no crop, resize, augmentation, per-channel standardization,
  or mean subtraction.
- Channel order: RGB.
- The retained natural-image calibration assets are not used to choose the
  primary CIFAR-10 cutoff.

Experiment 5 must use this same image representation for both the clean target
and the denoiser residual. If a future model interface requires another
external representation, conversion must occur at the model boundary and the
residual must be converted back to this frozen domain before projection. Any
incompatible change requires an explicit protocol amendment before model
curves are inspected.

## Frozen Visualization Set

The montage contains exactly 20 images: two images from each of the ten
CIFAR-10 classes.

Selection is deterministic:

1. Use the canonical test-set ordering.
2. Scan dataset indices from 0 through 9999.
3. For each class ID from 0 through 9, select the first two encountered
   examples of that class.
4. Display images ordered by class ID, then by encounter order.

No image may be replaced after cutoff outputs are viewed. Before Fourier
processing begins, the implementation must materialize the selected indices
and labels in `results/experiment_04_image_manifest.csv`. The manifest must
also record the dataset implementation/version and an integrity identifier for
the source data when one is available.

This rule guarantees class coverage but does not guarantee equal visual
difficulty or spectral content. Such variation is evidence to report, not a
reason to swap images.

## Frozen Fourier Convention

For each RGB channel independently:

1. Compute the two-dimensional FFT over the spatial axes with
   `norm="ortho"`.
2. Apply `fftshift` so that DC is at array coordinate `(16, 16)`.
3. Define centered integer frequency coordinates
   \(k=(k_y,k_x)\) and radius
   \(\lVert k\rVert_2=\sqrt{k_y^2+k_x^2}\).
4. Define the inclusive low-frequency mask
   \[
   M_{\mathrm{low},r}(k)=\mathbf{1}\{\lVert k\rVert_2\le r\}.
   \]
5. Define the high-frequency mask exactly as
   \[
   M_{\mathrm{high},r}=1-M_{\mathrm{low},r}.
   \]
6. Apply the same two-dimensional mask to every channel.

DC is always retained in the low band. Coefficients exactly on the cutoff
boundary belong to the low band. The implementation must use the preserved
`compute_fft()`, complementary-mask, and frequency-decomposition foundations
rather than introduce a second FFT convention.

On a 32 x 32 grid, one radius unit is one discrete Fourier bin from DC. The
per-axis Nyquist distance is 16 bins; therefore \(r/16\) is reported as a
normalized per-axis Nyquist radius. Corner frequencies extend to
\(\sqrt{16^2+16^2}\), so this normalization is not a fraction of the corner
radius.

## Candidate Cutoffs

The complete candidate set is frozen as

\[
r \in \{2,3,4,5,6,8\}.
\]

The corresponding normalized per-axis Nyquist radii are
\(\{0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5\}\).

Every candidate must be evaluated and displayed. Candidates may not be added,
removed, or selected using Experiment 5 denoiser outputs.

## Required Outputs

The future Experiment 4 implementation must produce:

- `results/experiment_04_image_manifest.csv`: frozen image identities.
- `results/experiment_04_cutoff_energy.csv`: numerical projection and
  reconstruction measurements for every image and cutoff.
- `results/experiment_04_cutoff_review.csv`: rubric scores and comments for
  every image, cutoff, and reviewer.
- `figures/experiment_04_frequency_cutoff_montage.png`: originals, every
  low-pass reconstruction, and every complementary high-pass component.
- `docs/experiment_04_frequency_cutoff_decision.md`: the applied rubric,
  selected cutoff or no-selection conclusion, sensitivity cutoffs, ambiguous
  examples, and failures.

No output may omit a candidate or an image because its result is unattractive
or ambiguous.

### Display Convention

Originals and low-pass reconstructions are displayed by mapping the
computational range with \((x+1)/2\), followed by clipping to \([0,1]\) for
display only.

High-pass components are signed. For image \(i\), define one display scale
shared across all candidate cutoffs and channels:

\[
s_i = \operatorname{percentile}_{99.5}
\left(\left|P_{\mathrm{high},r}x_i\right|:
r\in\{2,3,4,5,6,8\}\right).
\]

Display uses

\[
0.5 + 0.5\operatorname{clip}
\left(P_{\mathrm{high},r}x_i/s_i,-1,1\right),
\]

so zero maps to neutral gray. If \(s_i=0\), use \(s_i=1\). Each scale must be
recorded in the numerical CSV. This normalization affects display only.

### Numerical Checks and Energy Fractions

For each image and cutoff, compute

\[
x_{\mathrm{low}}=P_{\mathrm{low},r}x,\qquad
x_{\mathrm{high}}=P_{\mathrm{high},r}x.
\]

The implementation must verify:

- `max_abs(x - (x_low + x_high)) <= 1e-10`;
- relative energy decomposition error is at most `1e-12`;
- relative low/high inner-product magnitude is at most `1e-12`.

The relative quantities are defined exactly as

\[
\delta_E =
\frac{\left|E_{\mathrm{full}}-E_{\mathrm{low}}-E_{\mathrm{high}}\right|}
{\max(E_{\mathrm{full}},\epsilon)},
\]

\[
\delta_\perp =
\frac{\left|\operatorname{Re}\langle
x_{\mathrm{low}},x_{\mathrm{high}}\rangle\right|}
{\max(E_{\mathrm{full}},\epsilon)},
\]

where \(\epsilon\) is the smallest positive normal float64 value. Energies and
inner products sum over all spatial positions and RGB channels.

With \(E_{\mathrm{full}}=\lVert x\rVert_2^2\), report

\[
f_{\mathrm{low}}=
\frac{\lVert x_{\mathrm{low}}\rVert_2^2}{E_{\mathrm{full}}},
\qquad
f_{\mathrm{high}}=
\frac{\lVert x_{\mathrm{high}}\rVert_2^2}{E_{\mathrm{full}}}.
\]

The high-band fraction is the measured complementary energy fraction, not a
value filled in solely as \(1-f_{\mathrm{low}}\). Their sum must agree with one
within the energy tolerance.

### Machine-Readable Schemas

`experiment_04_image_manifest.csv` must contain:

```text
experiment_id,dataset_name,dataset_split,dataset_index,image_id,class_id,
class_name,dataset_version,source_integrity_id
```

`experiment_04_cutoff_energy.csv` must contain:

```text
experiment_id,image_id,dataset_index,class_id,class_name,cutoff_radius,
cutoff_normalized,total_energy,low_energy,high_energy,low_energy_fraction,
high_energy_fraction,reconstruction_max_abs_error,
energy_decomposition_relative_error,orthogonality_relative_error,
high_display_scale
```

`experiment_04_cutoff_review.csv` must contain:

```text
experiment_id,reviewer_id,image_id,dataset_index,class_id,class_name,
cutoff_radius,layout_score,identity_score,high_localization_score,
ambiguous,failure_category,comment
```

Numeric files must use a stable row order: image order from the manifest, then
ascending cutoff. Review rows additionally sort by reviewer ID first.

## Frozen Cutoff Decision Rubric

Two reviewers must independently inspect the complete montage. They score each
image and cutoff without hiding candidates or consulting model outputs.
Reviewers may compare each reconstruction with its original.

Each criterion uses a three-point ordinal scale:

- `layout_score`: 0 means coarse spatial layout is absent or misleading; 1
  means recognizable but substantially degraded; 2 means clearly retained.
- `identity_score`: 0 means important object identity is lost; 1 means
  ambiguous but plausible; 2 means clearly retained.
- `high_localization_score`: 0 means the high band is dominated by global
  silhouette or broad intensity structure; 1 means mixed global and localized
  content; 2 means it primarily contains localized edges and texture.

Every score of 0 or 1 requires a comment. Reviewers must flag ambiguous cases
and assign a failure category rather than resolve disagreement by changing the
image set or cutoff candidates.

A candidate qualifies for one reviewer only if all of the following hold:

- at least 16 of 20 images have `layout_score >= 1`;
- at least 14 of 20 images have `identity_score >= 1`;
- at least 16 of 20 images have `high_localization_score >= 1`;
- for every class, at least one of its two images has both
  `layout_score >= 1` and `identity_score >= 1`;
- every numerical reconstruction and energy check passes.

A candidate qualifies overall only if it qualifies separately for both
reviewers. Reviewer disagreements remain in the record.

Select the smallest interior candidate in \(\{3,4,5,6\}\) that qualifies
overall and whose immediately higher candidate also qualifies overall. The
second condition is the frozen adjacent-cutoff stability check. This rule
prefers the smallest defensible low band while requiring that the decision is
not an isolated one-radius judgment.

The selected cutoff must be justified using the recorded scores, energy
fractions, class variability, adjacent-cutoff behavior, and all documented
failures. Energy fraction alone cannot select the cutoff.

## Sensitivity and Failure Policy

After selecting \(r_\star\), preserve the immediately adjacent candidate below
and above it as \(r_-\) and \(r_+\). Experiment 5 must report the full residual
decomposition at all three radii. The reference cutoff cannot be changed after
Experiment 5 curves are examined.

If no interior candidate satisfies the rubric and stability condition,
Experiment 4 must report **no defensible single cutoff**. Experiment 5 is then
blocked from making a single-cutoff interpretation. A later, separately
reviewed protocol may adopt explicitly multiscale reporting, but it may not
retroactively tune a universal semantic boundary from denoiser results.

The decision report must retain:

- every failed numerical check;
- every image-level rubric failure;
- reviewer disagreement;
- classes or examples with unusual behavior;
- cutoffs where separation is ambiguous;
- sensitivity to adjacent candidates;
- the possibility that frequency-scale dependence prevents a universal
  semantic interpretation.

## Acceptance Gates

Experiment 4 implementation may begin only after this protocol freezes:

- [x] CIFAR-10 split, shape, channel order, and \([-1,1]\) representation;
- [x] deterministic 20-image selection rule;
- [x] orthonormal channelwise FFT and centered complementary masks;
- [x] inclusive cutoff boundary and DC treatment;
- [x] candidate cutoffs \(\{2,3,4,5,6,8\}\);
- [x] required visual and machine-readable outputs;
- [x] numerical reconstruction and energy checks;
- [x] cutoff decision rubric;
- [x] adjacent-cutoff sensitivity policy;
- [x] ambiguity and failure-reporting policy.

Passing these specification gates authorizes only a later Experiment 4
implementation phase. It does not authorize Experiments 5 or 6.

## Frozen Ambiguities

The following scientific limitations are intentionally not resolved by this
protocol:

- A radial cutoff is isotropic and cannot represent orientation-specific or
  spatially localized frequency structure.
- CIFAR-10's 32 x 32 resolution makes each radius increment coarse.
- Human recognizability and identity judgments are operational and may vary
  between reviewers.
- Low- and high-frequency bands are not semantic definitions.
- The fixed 20-image montage covers every class but is not a statistical
  estimate of the full CIFAR-10 distribution.
- A useful cutoff may depend on image content, class, noise level, or model;
  the protocol may therefore conclude that no universal boundary is
  defensible.
- Experiment 5's sigma grid, model invocation, sample aggregation, and common
  MSE normalization are not selected in this phase. Its protocol must apply
  one identical normalization to the full, low-band, and high-band energies so
  additivity is preserved.
