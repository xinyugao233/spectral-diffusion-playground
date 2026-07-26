# Experiment 5: Spectral Residual Curve Protocol

## Status

**Protocol frozen; execution blocked on model and dataset provenance.**

This document specifies a **paper-derived clean-room reimplementation** of the
fixed-noise denoising MSE in Eq. (5) of *Two Calm Ends and the Wild Middle: A
Geometric Picture of Memorization in Diffusion Models* (arXiv:2602.17846v1).
The original Figure 1 evaluator, executed checkpoints, CIFAR-10 1K permutation,
test indices, and random seeds were not recovered. Exact numerical
reproduction is not claimed.

This phase creates no evaluator, model output, result, or figure. Experiment 5
may not execute until every blocker in the acceptance-gate section is resolved
in a run manifest or by a separately reviewed protocol amendment.

## Scientific Objective

For a clean image \(X\), standard Gaussian noise \(Z\), and a fixed-noise
denoiser \(m_\sigma\), define

\[
X_\sigma = X + \sigma Z,
\qquad
e_\sigma = m_\sigma(X_\sigma) - X.
\]

The paper's fixed-noise quantity is

\[
\operatorname{MSE}_{\mathrm{full}}(\sigma)
= \mathbb{E}\left[\lVert e_\sigma\rVert_2^2\right].
\]

Experiment 5 applies the E004 complementary Fourier projections directly to
the residual:

\[
\operatorname{MSE}_{\mathrm{low},r}(\sigma)
= \mathbb{E}\left[
\lVert P_{\mathrm{low},r}e_\sigma\rVert_2^2
\right],
\]

\[
\operatorname{MSE}_{\mathrm{high},r}(\sigma)
= \mathbb{E}\left[
\lVert P_{\mathrm{high},r}e_\sigma\rVert_2^2
\right].
\]

The primary numerical identity is

\[
\operatorname{MSE}_{\mathrm{full}}(\sigma)
= \operatorname{MSE}_{\mathrm{low},r}(\sigma)
+ \operatorname{MSE}_{\mathrm{high},r}(\sigma)
\]

up to the frozen floating-point tolerances below. These curves are an
orthogonal decomposition of the paper's residual energy, not a new recovery
score.

## Frozen Frequency Cutoffs

- Reference cutoff: \(r_\star=4\).
- Primary sensitivity cutoffs: \(r=3\) and \(r=5\).
- Optional extended sensitivity check: \(r=6\).

The primary analysis always reports \(r\in\{3,4,5\}\). If computationally
feasible, \(r=6\) may be added as a declared extended check; it may not replace
either primary sensitivity cutoff.

The reference cutoff came from a single-reviewer qualitative E004 visual
decision. The planned two-independent-reviewer scoring procedure was not
completed. The choice is not an inter-rater, blinded, or statistically
representative review result.

## Experimental Conditions

All conditions use CIFAR-10 RGB images with shape \(32\times32\times3\).

### EDM-1K

- Model: the paper's EDM trained on its fixed 1,000-image CIFAR-10 subset.
- Training evaluation: all 1,000 images used to train that exact checkpoint.
- Test evaluation: canonical CIFAR-10 test indices `0..999`.
- Required provenance: checkpoint bytes and SHA-256, model configuration,
  training code repository and commit, training command if available, and the
  ordered 1,000-element training-index manifest.

The paper says that the 1K set is the first 1,000 elements of a randomly
indexed CIFAR-10 training set, but the permutation was not recovered. That
description is insufficient to reconstruct the subset.

### EDM-50K

- Model: the paper's EDM trained on all 50,000 CIFAR-10 training images.
- Training evaluation: canonical CIFAR-10 training indices `0..999`, used as a
  fixed Monte Carlo sample from the model's training distribution.
- Test evaluation: canonical CIFAR-10 test indices `0..999`.
- Required provenance: checkpoint bytes and SHA-256, model configuration,
  training code repository and commit, and training command if available.

### EMP-1K

- Model: the empirical posterior-mean denoiser in paper Eqs. (2)–(3), using
  the exact ordered EDM-1K training subset.
- Training evaluation: the same 1,000 EDM-1K training images.
- Test evaluation: canonical CIFAR-10 test indices `0..999`.
- Computation: posterior weights must use float64 log-sum-exp over all 1,000
  centers in the frozen \([-1,1]\) domain.

EMP-1K is a secondary analytic control. It must be run if a pre-execution
resource estimate shows that all frozen rows can be evaluated without changing
the grid, sample counts, or repeats. If omitted, the manifest must record the
resource estimate and `condition_status: omitted_infeasible`; its omission
does not permit substituting another model.

### Provenance Gate

No condition may execute until:

1. the EDM-1K and EDM-50K checkpoint identities are recorded;
2. the exact ordered EDM-1K subset indices are recorded;
3. checkpoint files and the subset manifest have stable hashes;
4. the model invocation is traced to a concrete function and source commit;
5. the CIFAR-10 implementation, version, archive integrity, and canonical
   index ordering are recorded.

If the paper artifacts remain unavailable, a newly trained clean-room model
may be used only after a separate amendment freezes its training code,
configuration, subset, seed, and checkpoint selection before any E005 curve is
viewed. Such a run must not be labeled an exact paper reproduction.

## Data and Model Representation

### Scientific Domain

- Source image: decoded CIFAR-10 uint8 RGB in \(\{0,\ldots,255\}\).
- Evaluator image:
  \(X=\operatorname{float64}(2(u/255)-1)\in[-1,1]^{32\times32\times3}\).
- Preprocessing: no crop, resize, augmentation, standardization, channel mean
  subtraction, or quantization.
- Scientific residual layout: HWC RGB float64.

This is the E004 computational domain and the paper's documented CIFAR-10
embedding.

### Model Boundary

1. Convert HWC RGB to the checkpoint's required layout, expected to be NCHW.
2. Cast \(X_\sigma\) to the checkpoint's required floating dtype and device.
3. Invoke the traced EDM clean-image prediction interface at the exact
   \(\sigma\), including any checkpoint-required preconditioning.
4. Convert the unquantized floating output back to HWC RGB float64.
5. Compute \(e_\sigma=\hat X-X\) in float64.

Scientific residuals use the raw floating denoiser output. Do not clamp to
\([-1,1]\), round to uint8, or apply display normalization before residual
calculation. A clamped diagnostic may be added only as a separately labeled
secondary output and cannot replace the primary result.

The run manifest must record the exact model call, input/output layout, model
dtype, autocast state, preconditioning path, and any sigma rounding. Execution
is blocked if those semantics cannot be traced.

## Gaussian Noise and Seeds

For every image, repeat, and primary sigma:

\[
Z\sim\mathcal{N}(0,I_{3\times32\times32}),
\qquad
X_\sigma=X+\sigma Z.
\]

The clean evaluator-domain image is converted to CHW before adding noise. Noise
is generated on CPU as float64 and then cast with \(X_\sigma\) at the model
boundary.

The clean-room seed policy is:

```text
master_noise_seed = 20260726
generator = NumPy PCG64DXSM
seed material = [master_noise_seed, split_code, dataset_index,
                 noise_repeat, sigma_index]
split_code: train=0, test=1
noise_repeats_per_image = 8
```

Each tensor is generated from its own `SeedSequence` using this material.
Model name, checkpoint, cutoff, batch index, worker ID, and device rank are not
part of the seed. Therefore:

- the same \(Z\) is shared across EDM-1K, EDM-50K, and EMP-1K for a matching
  split, image index, repeat, and sigma;
- the same residual is projected at every cutoff;
- reordering batches, changing batch size, or resuming a run cannot change a
  row's noise tensor.

The manifest records the NumPy version and the 64-bit derived seed or complete
seed material for every raw row. No noise row may be regenerated with a
different policy after curves are inspected.

## Sigma Grids

### Primary Grid

The primary grid is the paper's 18-point EDM polynomial sampling grid with
\(\rho=7\), \(\sigma_{\max}=80\), and \(\sigma_{\min}=0.002\):

\[
\sigma_i =
\left[
80^{1/7}
+ \frac{i}{17}
\left(0.002^{1/7}-80^{1/7}\right)
\right]^7,
\qquad i=0,\ldots,17.
\]

It is stored in descending high-noise-to-low-noise order as float64:

```text
80
57.58598472124816
40.785573796507961
28.374584604156844
19.352452980325229
12.91008238075732
8.4009353090998165
5.3151945217963821
3.2568215197655368
1.9233398370400518
1.088170636545279
0.58534812319454221
0.29644228447915727
0.13951646873101678
0.05994731123547159
0.022934518372333384
0.0075280199627840785
0.0020000000000000031
```

The terminal sampler value \(\sigma=0\) is not part of fixed-noise Eq. (5) and
is not an E005 evaluation point.

### Optional Dense Diagnostic Grid

A secondary diagnostic may use exactly 64 geometrically spaced float64 values
from 80 to 0.002, inclusive and descending:

```text
numpy.geomspace(80.0, 0.002, num=64, dtype=float64)
```

This grid is optional, must be declared in the manifest before inference, and
cannot select or revise transition windows. The 18-point grid is primary even
if the dense diagnostic is run. Neither grid may be tuned after curves are
examined.

## Fourier Projection and Numerical Identities

Use the preserved repository convention without a second implementation:

- channelwise 2D FFT over spatial axes;
- `norm="ortho"`;
- centered radial coordinates with DC at `(16, 16)`;
- inclusive low mask \(\mathbf{1}\{\lVert k\rVert_2\le r\}\);
- exact high-mask complement \(1-M_{\mathrm{low},r}\);
- identical mask on all RGB channels.

Projection is applied to the float64 residual after model invocation. For each
raw residual and cutoff, retain:

\[
e_{\mathrm{low},r}=P_{\mathrm{low},r}e,
\qquad
e_{\mathrm{high},r}=P_{\mathrm{high},r}e.
\]

Each raw row must pass:

```text
max_abs(e - (e_low + e_high)) <= 1e-10
abs(full_energy - low_energy - high_energy)
    / max(full_energy, smallest_normal_float64) <= 1e-12
abs(real(inner_product(e_low, e_high)))
    / max(full_energy, smallest_normal_float64) <= 1e-12
```

The measured high energy must be computed from \(e_{\mathrm{high},r}\), not
filled in by subtraction. A failed identity aborts aggregation and plotting.

## Reduction Convention

The canonical paper-faithful value for one image/noise row is summed squared
L2 energy over all 3,072 elements:

\[
E_{\mathrm{sum}}=\sum_{c,h,w} e_{c,h,w}^2.
\]

The per-element mean squared error is also reported:

\[
E_{\mathrm{mean}}=E_{\mathrm{sum}}/3072.
\]

Summed energy is the primary value because paper Eq. (5) writes
\(\lVert\cdot\rVert_2^2\). Per-element MSE is a deterministic secondary
conversion. Full, low, and high values always use the same reduction, so
additivity holds in both conventions.

## Aggregation and Uncertainty

The primary clean-room sample sizes are:

```text
training images per model condition = 1000
test images per model condition = 1000
noise repeats per image and sigma = 8
```

All per-image, per-repeat rows are retained before averaging. For each model,
split, sigma, cutoff, and band:

1. average the eight repeats within each image;
2. take the equally weighted mean across the 1,000 images.

Report the point mean and a 95% percentile image-cluster bootstrap confidence
interval using 10,000 resamples and bootstrap seed `20260727`. Resample image
identities and retain all repeats, sigmas, cutoffs, and models associated with
each resampled image. Test-set model comparisons use paired image resamples.
Train/test gaps resample the two splits independently because their image
identities differ. Do not resample Fourier coefficients, sigma points, or
individual repeat rows as independent observations.

Every expected raw row must have `status=ok`. Exceptions, nonfinite model
outputs, nonfinite energies, missing rows, and identity failures are written to
the validation report and cause the condition to fail. They are never silently
dropped, replaced, winsorized, or converted with `nan_to_num`.

## Transition-Window Extraction

Raw summed-energy curves are primary. Transition windows are secondary
descriptive summaries and are never called memorization danger zones.

The primary transition extraction uses only EDM-1K held-out test curves on the
18-point grid. It runs separately for low and high residual energies at
\(r=3,4,5\); \(r=6\) is optional extended sensitivity.

### Normalization

For each band, model, split, and cutoff, define:

- \(E_{\mathrm{high}}\): median energy over sigma indices `0,1,2`;
- \(E_{\mathrm{low}}\): median energy over sigma indices `15,16,17`;
- normalized recovery
  \[
  R(\sigma)=
  \frac{E_{\mathrm{high}}-E(\sigma)}
       {E_{\mathrm{high}}-E_{\mathrm{low}}}.
  \]

Do not clip \(R\). Endpoint overshoot remains visible. If
\(E_{\mathrm{high}}-E_{\mathrm{low}}\le0\), either endpoint is nonfinite, or
the denominator is numerically indistinguishable from zero, return
`no_clear_transition`.

### Crossing Rule

Traverse indices from high noise to low noise. No smoothing or interpolation
is applied.

- Entry index: the first index \(i\) for which \(R_i\ge0.20\) and
  \(R_{i+1}\ge0.20\).
- Exit index: the first index \(j\ge i\) for which \(R_j\ge0.80\) and
  \(R_{j+1}\ge0.80\).
- Equality counts as crossing.
- Earliest qualifying index resolves ties.
- The reported window contains all zero-based indices from entry through exit,
  inclusive.
- A valid window must span at least two distinct sampler indices. Otherwise
  return `no_clear_transition`.

If either crossing is absent, exit precedes entry, later values exhibit a
two-point recrossing below the corresponding threshold, or the minimum width
fails, report `no_clear_transition` with a reason. Do not widen, smooth, or
manually move the window.

The \(r=4\) window is the reference. It is labeled
`adjacent_cutoff_stable=true` only when both its entry and exit indices differ
by at most one index from the corresponding \(r=3\) and \(r=5\) boundaries.
Otherwise it remains a valid reference-cutoff description but is explicitly
`cutoff_sensitive=true`; no cutoff-invariant transition may be claimed. The
optional \(r=6\) result is reported separately and does not determine this
flag.

Because the primary E005 grid is the sampler grid, no sigma-to-step snapping is
needed. A later E006 protocol must decide how to use identified or unidentified
windows without modifying this rule.

## Required Machine-Readable Outputs

This protocol freezes output contracts only; it does not create them.

### Run Manifest

`results/experiment_05_manifest.json` must contain:

```text
experiment_id, run_id, git_commit, protocol_commit, reproduction_claim,
paper_title, paper_sha256, corrected_plan_sha256,
model_conditions, checkpoint_paths, checkpoint_sha256,
model_source_repositories, model_source_commits, model_call_semantics,
dataset_implementation, dataset_version, dataset_archive_integrity,
train_index_manifests, test_indices, index_manifest_sha256,
image_domain, model_domain, layout_conversion, model_dtype, autocast,
output_clamping, output_quantization,
primary_sigma_formula, primary_sigma_values, dense_grid_status,
cutoffs, master_noise_seed, noise_generator, noise_repeats,
bootstrap_seed, bootstrap_resamples, host, device, dependency_versions
```

Paths may be redacted for publication, but hashes and stable artifact
identities may not be omitted.

### Per-Sample Residual Rows

`results/experiment_05_bandwise_residuals.csv` or an exactly equivalent Parquet
file must contain:

```text
experiment_id,run_id,model,checkpoint_sha256,split,image_index,
image_manifest_position,noise_repeat,noise_seed,sigma_index,sigma,
sigma_grid,cutoff,cutoff_normalized,reduction_elements,
full_squared_error,low_squared_error,high_squared_error,
full_mean_squared_error,low_mean_squared_error,high_mean_squared_error,
reconstruction_max_abs_error,energy_additivity_absolute_error,
energy_additivity_relative_error,orthogonality_relative_error,status,error
```

Stable row order is model, split, image-manifest position, noise repeat,
sigma index, then ascending cutoff. The unique key is all fields through
`cutoff`.

### Aggregated Curves

`results/experiment_05_aggregated_curves.csv` must contain:

```text
experiment_id,run_id,model,split,sigma_index,sigma,sigma_grid,cutoff,
band,n_images,n_repeats,mean_summed_squared_error,
ci95_low_summed_squared_error,ci95_high_summed_squared_error,
mean_per_element_mse,ci95_low_per_element_mse,ci95_high_per_element_mse,
aggregation_status
```

Allowed `band` values are `full`, `low_frequency_residual`, and
`high_frequency_residual`. Full-energy rows are repeated by cutoff only when
needed for direct identity joins; repeated values must be exactly identical.

### Transition Summary

`results/experiment_05_transition_windows.json` must record raw endpoint
estimates, every normalized value, crossing indices, sigma bounds, persistence
checks, recrossing checks, minimum-width status, adjacent-cutoff stability,
and the reason for every `no_clear_transition` result.

### Identity Validation

`results/experiment_05_identity_validation.json` must report expected and
observed row counts, duplicate and missing keys, maximum reconstruction error,
maximum absolute and relative additivity errors, maximum orthogonality error,
all nonfinite or failed rows, and pass/fail status. Plotting is blocked unless
this report passes.

## Required Figures

Figures are generated only from validated raw and aggregate outputs:

1. full residual energy versus sigma for train and test splits;
2. low-frequency residual energy versus sigma;
3. high-frequency residual energy versus sigma;
4. train-test residual-energy gaps by band;
5. \(r=3,4,5\) cutoff sensitivity, with optional \(r=6\) clearly secondary;
6. normalized transition curves with entry/exit annotations or an explicit
   `no clear transition` label;
7. reconstruction and energy-additivity error diagnostics.

The horizontal axis uses \(\sigma\) on a log scale in descending
high-to-low-noise order, matching the paper's interpretation. Raw full, low,
and high curves use consistent units and state whether summed energy or
per-element MSE is shown.

Figure labels use **low-frequency residual** and **high-frequency residual**.
The phrases "general-structure proxy" and "fine-detail proxy" may appear only
as parenthetical operational interpretations. Curves, gaps, and transition
windows are not memorization evidence.

## Reproduction Claim

Every report and figure caption must state:

> This is a paper-derived clean-room reimplementation. The original Figure 1
> evaluator, executed checkpoint identities, CIFAR-10 1K subset permutation,
> test selection, and random seeds were not recovered. Exact numerical
> reproduction is not claimed. The implementation follows the paper's
> mathematical definitions and reported settings where available and records
> all clean-room choices explicitly.

If additional original artifacts are later recovered, compare them against
this protocol before execution. Do not silently replace a frozen clean-room
choice.

## Acceptance Gates and Current Blockers

Protocol gates:

- [x] Eq. (5) residual and orthogonal decomposition defined.
- [x] reference and sensitivity cutoffs frozen.
- [x] CIFAR-10 scientific domain and model-boundary conversion frozen.
- [x] unquantized, unclamped residual policy frozen.
- [x] batching-independent paired noise and repeat count frozen.
- [x] primary and optional sigma grids frozen.
- [x] summed-energy and per-element reduction conventions frozen.
- [x] aggregation, uncertainty, failure, and transition rules frozen.
- [x] output schemas, numerical identities, and figures frozen.
- [x] clean-room reproduction disclosure frozen.

Execution blockers:

- [ ] exact EDM-1K checkpoint identity and hash recovered or replaced through
  an approved clean-room training amendment;
- [ ] exact EDM-50K checkpoint identity and hash recovered or replaced through
  an approved clean-room training amendment;
- [ ] exact ordered EDM-1K training indices and hash recovered or newly frozen
  before replacement-model training;
- [ ] concrete model invocation and source commit recorded;
- [ ] CIFAR-10 implementation and archive integrity recorded;
- [ ] EMP-1K feasibility disposition recorded;
- [ ] run manifest reviewed before inference.

Passing this documentation phase does not authorize model inference. Experiment
5 implementation and execution require a separate reviewed phase. Experiment 6
remains blocked.
