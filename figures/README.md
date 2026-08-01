# Figures Index

This directory contains curated outputs from completed experiments. Figures
show measurements already recorded in machine-readable form; they are not a
substitute for result tables or frozen protocols.

## README Showcase

The root README surfaces a deliberately small visual sequence rather than every
available plot:

- E001 RGB FFT visualization;
- E002 noise/frequency grid and normalized radial distribution;
- E003 reconstruction grid and complementary residuals;
- one representative E004 class-grouped cutoff montage;
- the E004A coverage/posterior-concentration geometry;
- both E005 low/high residual-curve figures and the frozen transition-window
  extraction, plus their comparison with the paper geometry;
- E006 transition/control and generated/nearest-neighbor figures.

Secondary figures remain linked from the corresponding experiment section and
from the indexes below.

## E001-E003: Fourier Foundations

- `understanding_images_in_fourier_space_default_fft_reference_rgb.png`
- `how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png`
- `how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_radial_energy.png`
- `how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_normalized_radial_distribution.png`
- `where_image_information_lives_grid.png`
- `high_frequency_residuals.png`
- `reconstruction_error_vs_frequency_radius.png`

## E004: Operational Frequency Cutoff

The five `experiment_04_cutoff_montage_classes_*.png` files preserve all 20
frozen CIFAR-10 examples and every candidate cutoff. Difficult examples remain
visible; the montages were not filtered after review.

## E004A: Paper Coverage-Concentration Geometry

[`experiment_04a/`](experiment_04a/) contains the paper-derived clean-room
coverage and maximum-posterior-weight figure. The red interval is the
`clean-room high-high region (q_C = q_W = 0.8)` based on frozen point-estimate
thresholds, not a boundary inferred from the E005 spectral curves.

[`e006_grid_geometry_alignment.png`](experiment_04a/e006_grid_geometry_alignment.png)
uses the exact E006 sigma schedule and deliberately encodes the E004A
high-high points, E005 spectral transitions, and paper-reported medium
reference differently so they cannot be read as interchangeable definitions.

## E005: Spectral Residual Curves

[`experiment_05/`](experiment_05/) contains the six frozen E005 figures plus a
new cross-analysis figure covering the
EDM-1K and EDM-50K curves, train/test comparison, cutoff sensitivity,
transition windows, and additivity diagnostics.

The primary portfolio figure is
[`experiment_05_edm1k_low_high_residual_curves.png`](experiment_05/experiment_05_edm1k_low_high_residual_curves.png).
The cross-analysis figure
[`geometry_and_spectral_transitions.png`](experiment_05/geometry_and_spectral_transitions.png)
places the geometric and spectral measurements on a shared sigma orientation
without interpolating either grid for decisions.

## E006: Historical Spectral-Window Swaps

[`experiment_06/`](experiment_06/) contains six final figures covering
memorization rates, paired changes, transition/control comparisons,
nearest-neighbor ratios, the paper-reported medium reference, and deterministic
generated/nearest-neighbor pairs.

The formal E006 outcome remains `INCONCLUSIVE`. The figures support only the
documented descriptive finding that the E005 low-frequency spectral transition
was the tested window most strongly associated with changes in the
memorization criterion.

## Figure Policy

Keep filenames stable, use one experiment subdirectory for scientific result
figures, and commit only final reviewable artifacts. Intermediate plots and
large image arrays remain external.
