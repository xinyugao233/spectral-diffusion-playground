# Experiment 5: Natural Image Calibration of `S_low` and `S_high`

Status: Specification frozen; image curation has not started.

## Objective

Quantify the stability and uncertainty of frequency-band recovery metrics
across natural images before applying them to learned denoisers.

This is a metric-calibration study. It does not train a model, study diffusion,
or assign semantic meaning to either frequency band.

## Calibration Set

Curate 5–10 challenging natural RGB images under
`assets/examples/natural/`. Include varied spectral characteristics such as
smooth landscapes, natural edges, faces or people, architecture, isolated
objects, and texture-heavy scenes.

The set must intentionally include both smooth-dominated and texture-dominated
images, rather than relying on visual or semantic diversity alone.

Every image must have a row in `assets/examples/metadata.csv` with:

```csv
image_id,filename,source,creator,license,url,download_date,original_resolution,preprocessing
```

Use the exact source license. In particular, do not label the Unsplash License
as CC0.

## Frozen Preprocessing

Apply these deterministic operations in order:

1. Decode and convert to RGB.
2. Center crop to the largest square.
3. Resize to 256 × 256 with bicubic interpolation.
4. Convert to `float32`.
5. Scale channel values to `[0, 1]`.

Record this exact pipeline in every metadata row. Preserve original downloaded
files; write processed derivatives separately if preprocessing is materialized.

## Evaluation Grid

- Cutoffs: `r ∈ {20, 40, 80}` centered Fourier pixels.
- Controls: `low_band_first`, `high_band_first`, and `together`.
- Recovery threshold: `0.8`.
- Seeds: fixed and recorded.
- Construction: identical to Experiment 4 unless a deviation is documented.

## Raw Score Schema

Use one schema that can extend to later model experiments:

```csv
experiment_id,image_id,split,checkpoint,trajectory,axis_name,axis_value,cutoff,seed,S_low,S_high
```

Experiment 5 uses:

- `experiment_id = experiment_05`
- `split = calibration`
- `checkpoint = not_applicable`
- `trajectory ∈ {low_band_first, high_band_first, together}`
- `axis_name = synthetic_recovery_progress`
- `axis_value ∈ [0, 1]`

Separating `checkpoint`, `trajectory`, and the measurement axis prevents later
experiments from overloading `trajectory` with checkpoint identifiers or
silently changing the meaning of `progress`.

## Quantitative Analyses

### Ordering Survival

For each image and cutoff, compute whether threshold-crossing order matches the
known control:

- Low-band-first: `t_low < t_high`.
- High-band-first: `t_high < t_low`.
- Together: `|t_low - t_high|` is no greater than one progress step.

Report the count and proportion of successful images for every cutoff and
control.

### Crossing Gap

For each image, define:

```text
delta_t = t_high - t_low
```

Report its mean, standard deviation, and bootstrap confidence interval across
images for every cutoff and control.

### Aggregate Curves

For `S_low` and `S_high`, report the across-image mean and standard deviation at
every progress value. Plot the mean with a shaded standard-deviation band.

### Bootstrap Uncertainty

Resample images, not trajectory points. Use a fixed bootstrap seed and record
the number of resamples. Estimate confidence intervals for:

- ordering survival rates
- mean crossing gaps
- mean recovery curves

With only 5–10 images, bootstrap intervals are descriptive calibration
uncertainty, not population-level guarantees.

## Required Outputs

Machine-readable results:

```text
results/experiment_05_scores.csv
results/experiment_05_crossings.csv
results/experiment_05_summary.json
```

Figures:

```text
figures/experiment_05_per_image_curves.png
figures/experiment_05_mean_curves.png
figures/experiment_05_cutoff_comparison.png
```

## Failure Analysis

The final report must identify:

- images where expected ordering fails
- cutoffs where separation collapses
- unusual spectral content associated with failures
- sensitivity to preprocessing or band energy

Perfect separation is not required. Success means cutoff dependence and
across-image variability are measured, ordering behavior is interpretable, and
failure cases are understood well enough to define a defensible model-study
protocol.

## Experiment 6 Gate

Do not start the fixed-model denoising baseline until:

- provenance is complete
- preprocessing is frozen
- the raw schema is finalized
- all cutoffs have been evaluated
- uncertainty is reported
- failure cases are documented
