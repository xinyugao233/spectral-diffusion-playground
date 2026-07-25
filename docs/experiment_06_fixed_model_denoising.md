# Experiment 6: Fixed-Model Frequency-Band Recovery

Status: Protocol frozen on 2026-07-25. Implementation is complete. Checkpoint
acquisition and model evaluation have not started.

## Objective

For one fixed pretrained denoiser, quantify how `S_low` and `S_high` vary with
the noise level of a known-target observation.

This is an inference-time baseline. It measures when frequency bands become
recoverable from noisy inputs under one fixed model. It does not identify when
the model learned a band, establish memorization, or compare training
checkpoints.

The experiment addresses Q003/I003 by establishing the fixed-model reference
needed before a checkpoint-aligned train-versus-held-out study.

## Frozen Model

Use the official OpenAI guided-diffusion unconditional ImageNet 256 x 256
model.

- Upstream repository:
  `https://github.com/openai/guided-diffusion`
- Pinned source revision:
  `22e0df8183507e13a7813f8d38d51b072ca1e67c`
- Checkpoint filename: `256x256_diffusion_uncond.pt`
- Official checkpoint URL:
  `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`
- Published object size: `2,211,383,297` bytes
- Published Azure content MD5: `fd9dd2335b8736d521de0aed54bd90ca`
- Conditioning: none
- Training domain: ImageNet at 256 x 256

Use the official architecture and diffusion flags:

```text
attention_resolutions=32,16,8
class_cond=False
diffusion_steps=1000
image_size=256
learn_sigma=True
noise_schedule=linear
num_channels=256
num_head_channels=64
num_res_blocks=2
resblock_updown=True
use_fp16=True
use_scale_shift_norm=True
```

All flags not shown above retain their defaults at the pinned source revision.
In particular, evaluation uses epsilon prediction, the native unrespaced
schedule, no class conditioning, and no classifier guidance.

The source revision, checkpoint URL, object size, and published MD5 identify
the intended artifact without downloading it during protocol work. Before the
first run, checkpoint acquisition must be a separate recorded step that:

1. computes and records SHA-256;
2. verifies the byte size and published MD5;
3. loads the state dictionary with no missing or unexpected keys;
4. records the local path without committing the 2.1 GB checkpoint.

Any mismatch fails the acquisition gate. Do not silently substitute a
Diffusers conversion, class-conditional checkpoint, LSUN checkpoint, or newer
source revision.

## Frozen Evaluation Set

Use the six Experiment 5 images in metadata order:

```text
image_001
image_002
image_003
image_004
image_005
image_006
```

The exact files and provenance are fixed by playground commit `a56e230` and
`assets/examples/metadata.csv`. Run
`scripts/validate_natural_image_dataset.py` before model evaluation.

Apply the unchanged Experiment 5 preprocessing:

1. Decode and convert to RGB.
2. Center crop to the largest square.
3. Resize to 256 x 256 with bicubic interpolation.
4. Convert to `float32`.
5. Scale channel values to `[0, 1]`.

Map the clean image to the model domain only after preprocessing:

```text
y_0 = 2 x_0 - 1
```

The calibration images are not claimed to be ImageNet samples. Domain mismatch
between this small provenance-recorded set and the model's training
distribution is a mandatory limitation, not a quantity to tune away.

## Evaluation Object

Evaluate direct clean-image prediction from independently constructed forward
diffusion observations. Do not run an unconditional reverse-sampling chain.

For native diffusion timestep `t`, let `alpha_bar_t` be the cumulative product
defined by the pinned 1,000-step linear schedule. For a fixed standard-normal
noise realization `epsilon`:

```text
y_t = sqrt(alpha_bar_t) y_0
      + sqrt(1 - alpha_bar_t) epsilon
```

Pass `(y_t, t)` to the fixed model and obtain the raw clean-image estimate:

```text
y_hat_0_raw = diffusion.p_mean_variance(
    model,
    y_t,
    t,
    clip_denoised=False,
)["pred_xstart"]
```

The checkpoint learns variance as well as the mean parameterization. The
variance output is not a recovery target; use the upstream
`p_mean_variance` implementation so channel splitting and epsilon-to-`x_0`
conversion follow the pinned source exactly.

Map the estimate back to the repository image domain without clipping:

```text
x_hat_0_raw = (y_hat_0_raw + 1) / 2
```

Compute all errors and recovery scores from `x_hat_0_raw`. A separately clipped
copy may be used for display only:

```text
x_hat_0_display = clip(x_hat_0_raw, 0, 1)
```

Also record the fraction of raw prediction values outside `[0, 1]`. This makes
score saturation and display clipping visible.

This known-target design avoids the correspondence ambiguity of unconditional
sampling. A generated reverse trajectory has no predetermined clean target and
is outside Experiment 6.

## Noise Axis

The model uses a variance-preserving diffusion process rather than the additive
`x_0 + sigma epsilon` parameterization used in Experiment 2. Define the
effective noise-to-signal ratio:

```text
sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t)
```

After dividing `y_t` by `sqrt(alpha_bar_t)`, this is equivalent to an additive
observation `y_0 + sigma_t epsilon`. Therefore:

- primary axis: `sigma_t`;
- scale: logarithmic;
- figure direction: high noise on the left, low noise on the right;
- secondary recorded coordinate: native integer timestep `t`;
- solver or sampling step: not applicable.

Evaluate the following 41 native timesteps:

```text
{0, 25, 50, ..., 950, 975, 999}
```

Use the unrespaced 1,000-step schedule. Record the exact `alpha_bar_t` and
derived `sigma_t` values in the raw output. Do not relabel native timesteps as
solver steps.

## Seeds and Pairing

Use noise seeds:

```text
{0, 1, 2, 3, 4}
```

For each seed, generate one six-image Gaussian-noise batch in metadata order.
Each image receives a distinct noise tensor, but its tensor is reused at every
timestep and cutoff. This pairing changes only the noise level along a curve
and reduces avoidable between-timestep variance.

The implementation must record:

- NumPy, PyTorch, CUDA, and cuDNN versions;
- noise generator and version;
- device and GPU model;
- deterministic-algorithm settings;
- batch order and batch size.

Generate noise from one NumPy random generator initialized separately for each
listed seed. For each seed, draw the full `6 x 3 x 256 x 256` batch once in
metadata order, then reuse it throughout the run. The manifest must pin the
NumPy version because generator streams are part of run identity.

The model runs in evaluation mode under inference-only execution. Use the
published FP16 model setting, then transfer raw predictions to CPU `float64`
for Fourier projections and metric computation. No autocast policy may vary
between runs.

## Frozen Frequency Metrics

Use the unchanged Experiment 4/5 definitions:

```text
cutoff in {20, 40, 80}
E_low  = ||L*_r(x_hat_0_raw) - L*_r(x_0)||_2
         / ||L*_r(x_0)||_2
E_high = ||H_r(x_hat_0_raw) - H_r(x_0)||_2
         / ||H_r(x_0)||_2
S_low  = max(0, 1 - E_low)
S_high = max(0, 1 - E_high)
```

`L*_r` is the circular low-pass reconstruction with per-channel DC removed.
`H_r` is its exact complementary high-pass reconstruction. Preserve
orthonormal FFT scaling and inclusive centered-radius masks.

Raw `E_low` and `E_high` are primary measurements. The clipped scores remain
useful visual summaries, but score values at zero must not hide
worse-than-baseline errors.

The cutoffs are a required sensitivity analysis:

- `r=20` is known to make high-band-first separation fragile;
- `r=80` is known to increase low-band-first across-image variability;
- `r=40` is the matched construction/evaluation reference from calibration.

No cutoff is primary by post hoc visual preference.

## Raw Score Schema

Write:

```text
results/experiment_06_scores.csv
```

with:

```csv
experiment_id,image_id,split,checkpoint,trajectory,axis_name,axis_value,timestep,alpha_bar,sigma,cutoff,seed,low_relative_error,high_relative_error,S_low,S_high,out_of_range_fraction
```

Use these fixed values:

- `experiment_id = experiment_06`
- `split = calibration`
- `checkpoint = 256x256_diffusion_uncond.pt`
- `trajectory = direct_x0_prediction`
- `axis_name = vp_noise_to_signal_ratio`
- `axis_value = sigma_t`

The run manifest must additionally record the playground commit, upstream
commit, checkpoint SHA-256, model flags, dataset commit, metadata file hash,
preprocessing, timestep grid, seeds, software environment, hardware, and exact
command.

## Aggregation and Uncertainty

For every cutoff and noise level:

1. average the five seeds within each image;
2. report the mean and sample standard deviation across the six image means;
3. report within-image seed standard deviations separately;
4. estimate a 95% hierarchical bootstrap interval.

The hierarchical bootstrap first resamples images, then resamples seeds within
each selected image. Use 10,000 resamples and bootstrap seed `20260725`.
Neither timesteps nor Fourier coefficients are independent resampling units.

With six images, intervals describe this calibration set and its five noise
realizations. They are not population-level guarantees.

The score threshold `0.8` is retained only as a secondary descriptive
reference. If a curve is summarized by a crossing, define it as the first
crossing while moving from high to low noise that remains at or above `0.8` at
all subsequently evaluated lower-noise points. Report `not_reached` when this
condition fails. Do not force monotonicity or interpolate across reversals.

## Required Outputs

Machine-readable outputs:

```text
results/experiment_06_scores.csv
results/experiment_06_summary.json
results/experiment_06_manifest.json
```

Figures:

```text
figures/experiment_06_mean_recovery_curves.png
figures/experiment_06_per_image_recovery_curves.png
figures/experiment_06_raw_error_and_clipping_diagnostics.png
```

The main figure must show `S_low` and `S_high` with uncertainty for all three
cutoffs. The diagnostics figure must preserve raw errors and out-of-range
fractions so clipped scores cannot create a misleading plateau.

## Numerical and Reproducibility Gates

The run fails unless:

- dataset validation passes;
- checkpoint identity and state-dictionary loading pass;
- every model input and prediction has shape `N x 3 x 256 x 256`;
- all raw outputs and metrics are finite;
- the forward-noise implementation matches the pinned schedule;
- low- and high-frequency projections reconstruct their input within numerical
  tolerance;
- metric computation receives the unclipped prediction;
- repeated same-hardware runs agree within a predeclared numerical tolerance;
- Experiment 4 and Experiment 5 tests and canonical outputs remain unchanged.

The implementation declares a repeatability tolerance of `1e-6` before the
first full run and performs two complete inference passes. If the selected GPU
kernels cannot satisfy deterministic execution, preserve repeated-run
differences and report that failure rather than relaxing the gate after
observing results.

## Interpretation Contract

Permitted conclusion:

> For this fixed checkpoint, dataset, corruption protocol, and cutoff, recovery
> of the two operational frequency bands varies with effective noise level in
> the measured way.

Not permitted:

- the model learned low or high frequencies at a particular time;
- the model memorized a recovered band;
- low frequencies are semantic structure;
- high frequencies are memorized detail;
- unconditional generation follows the same path;
- one cutoff is universally correct.

The baseline is weakened if ordering or crossing locations change materially
across seeds, images, or cutoffs; if raw errors reveal that clipped scores are
mostly saturated; or if domain mismatch dominates the six-image results. Such
outcomes are failure analysis, not reasons to tune the frozen protocol.

## Experiment 7 Gate

Experiment 7 remains blocked until Experiment 6 has:

- a verified checkpoint hash and source identity;
- a reproducible known-target evaluation;
- raw and clipped metrics for all cutoffs;
- image- and seed-level uncertainty;
- clipping and non-monotonicity diagnostics;
- explicit failure analysis;
- a report that preserves the inference-time versus training-time distinction.

## Primary Sources

- OpenAI guided-diffusion repository and checkpoint instructions:
  `https://github.com/openai/guided-diffusion/tree/22e0df8183507e13a7813f8d38d51b072ca1e67c`
- Pinned diffusion implementation:
  `https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py`
- Pinned model card:
  `https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/model-card.md`
