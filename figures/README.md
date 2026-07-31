# Figures Index

This directory contains curated outputs from completed experiments. Figures
show measurements already recorded in machine-readable form; they are not a
substitute for result tables or frozen protocols.

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

## E005: Spectral Residual Curves

[`experiment_05/`](experiment_05/) contains six final figures covering the
EDM-1K and EDM-50K curves, train/test comparison, cutoff sensitivity,
transition windows, and additivity diagnostics.

The primary portfolio figure is
[`experiment_05_edm1k_low_high_residual_curves.png`](experiment_05/experiment_05_edm1k_low_high_residual_curves.png).

## E006: Transition-Window Swaps

[`experiment_06/`](experiment_06/) contains six final figures covering
memorization rates, paired changes, transition/control comparisons,
nearest-neighbor ratios, the paper-style medium reference, and deterministic
generated/nearest-neighbor pairs.

The formal E006 outcome remains `INCONCLUSIVE`. The figures support only the
documented descriptive finding that the low-transition window was the tested
window most strongly associated with changes in the memorization criterion.

## Figure Policy

Keep filenames stable, use one experiment subdirectory for scientific result
figures, and commit only final reviewable artifacts. Intermediate plots and
large image arrays remain external.
