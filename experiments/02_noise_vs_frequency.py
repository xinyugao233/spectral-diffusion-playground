"""Visualize how additive Gaussian noise changes an image's spectral structure.

This experiment is intentionally educational rather than model-centric. It
connects the forward perturbation step ``x_sigma = x + sigma * epsilon`` to two
observable consequences:

1. what the image looks like in pixel space after noise is added
2. how the corresponding Fourier magnitude changes across frequency radii
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

# Import the shared bootstrap so this script can be run directly from the repo root.
import _bootstrap  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_TITLE = "How Gaussian Noise Changes Frequency Content"
SIGMA_VALUES: Final[tuple[float, ...]] = (0.0, 0.05, 0.1, 0.2, 0.5)
CURVE_COLORS: Final[tuple[str, ...]] = (
    "#0f172a",
    "#2563eb",
    "#0f766e",
    "#d97706",
    "#dc2626",
)

from spectral_diffusion_playground.fft import (
    compute_fft,
    log_magnitude,
    magnitude_spectrum,
    normalize_radial_energy,
    radial_frequency_energy,
    shift_fft,
)
from spectral_diffusion_playground.noise import add_gaussian_noise
from spectral_diffusion_playground.utils import load_experiment_image, slugify_stem
from spectral_diffusion_playground.visualization import (
    ImagePanel,
    LineCurve,
    normalize_scalar_fields,
    prepare_image_for_display,
    save_curve_plot,
    save_panel_grid,
)


@dataclass(frozen=True, slots=True)
class SigmaAnalysis:
    """Views and summary statistics for one noise scale."""

    sigma: float
    noisy_image: np.ndarray
    log_spectrum: np.ndarray
    radius: np.ndarray
    radial_energy: np.ndarray
    normalized_radius: np.ndarray
    normalized_radial_energy: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how additive Gaussian noise changes both pixel-space "
            "appearance and Fourier-space structure."
        )
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional path to an input image. If omitted, a default reference image is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to generate deterministic Gaussian noise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
        help="Directory where generated figures will be saved.",
    )
    return parser.parse_args()


def analyze_noise_progression(image: np.ndarray, *, seed: int) -> list[SigmaAnalysis]:
    """Compute noisy images, spectra, and radial energy curves for each sigma.

    All log-spectrum panels share one global 99.5th-percentile normalization
    after channel averaging. This keeps side-by-side comparisons visually fair:
    no panel silently rescales itself.
    """
    noisy_images: list[np.ndarray] = []
    raw_log_spectra: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    radial_energies: list[np.ndarray] = []

    for sigma in SIGMA_VALUES:
        noisy_image = add_gaussian_noise(image, sigma, seed=seed)
        shifted_spectrum = shift_fft(compute_fft(noisy_image))
        magnitude = magnitude_spectrum(shifted_spectrum)
        raw_log_spectra.append(log_magnitude(magnitude))
        noisy_images.append(prepare_image_for_display(noisy_image))

        radius, energy = radial_frequency_energy(shifted_spectrum, is_shifted=True)
        radii.append(radius)
        radial_energies.append(np.maximum(energy, 1e-12))

    normalized_log_spectra = normalize_scalar_fields(
        raw_log_spectra,
        normalization="global_percentile",
        percentile=99.5,
    )

    return [
        SigmaAnalysis(
            sigma=sigma,
            noisy_image=noisy_images[index],
            log_spectrum=normalized_log_spectra[index],
            radius=radii[index],
            radial_energy=radial_energies[index],
            normalized_radius=radii[index][1:],
            normalized_radial_energy=normalize_radial_energy(radial_energies[index][1:]),
        )
        for index, sigma in enumerate(SIGMA_VALUES)
    ]


def build_output_paths(
    source_path: Path,
    *,
    output_dir: Path,
    seed: int,
) -> tuple[Path, Path, Path]:
    """Resolve descriptive output paths for the grid and radial-analysis figures."""
    safe_stem = slugify_stem(source_path)
    prefix = f"how_gaussian_noise_changes_frequency_content_{safe_stem}_seed{seed}"
    return (
        output_dir / f"{prefix}_grid.png",
        output_dir / f"{prefix}_radial_energy.png",
        output_dir / f"{prefix}_normalized_radial_distribution.png",
    )


def save_noise_grid(analyses: list[SigmaAnalysis], output_path: Path) -> Path:
    """Save the main side-by-side pixel-space and Fourier-space grid."""
    panel_rows = [
        [
            ImagePanel(
                title=f"σ = {analysis.sigma:.2f}",
                image=analysis.noisy_image,
            )
            for analysis in analyses
        ],
        [
            ImagePanel(
                title="",
                image=analysis.log_spectrum,
                cmap="magma",
            )
            for analysis in analyses
        ],
    ]
    return save_panel_grid(
        panel_rows,
        output_path,
        figure_title=EXPERIMENT_TITLE,
        row_labels=("Noisy Image", "Log Fourier Spectrum"),
        figure_title_size=14.0,
        figure_title_y=0.985,
    )


def save_radial_energy_figure(analyses: list[SigmaAnalysis], output_path: Path) -> Path:
    """Save a curve plot of annulus-averaged power versus frequency radius."""
    curves = [
        LineCurve(
            label=f"σ = {analysis.sigma:.2f}",
            x=analysis.radius,
            y=analysis.radial_energy,
            color=CURVE_COLORS[index],
        )
        for index, analysis in enumerate(analyses)
    ]
    return save_curve_plot(
        curves,
        output_path,
        figure_title="Radial Frequency Energy Across Noise Scales",
        x_label="Frequency radius r (pixels from the centered origin)",
        y_label="Annulus-averaged power E(r)",
        yscale="log",
        legend_title="Noise scale",
    )


def save_normalized_radial_distribution_figure(
    analyses: list[SigmaAnalysis],
    output_path: Path,
) -> Path:
    """Save a DC-excluded normalized radial plot with a white-noise reference."""
    radius = analyses[0].normalized_radius
    white_noise_reference = np.full_like(radius, 1.0 / radius.size, dtype=np.float64)
    curves = [
        LineCurve(
            label=f"σ = {analysis.sigma:.2f}",
            x=analysis.normalized_radius,
            y=analysis.normalized_radial_energy,
            color=CURVE_COLORS[index],
        )
        for index, analysis in enumerate(analyses)
    ]
    curves.append(
        LineCurve(
            label="Expected white-noise distribution",
            x=radius,
            y=white_noise_reference,
            color="#6b7280",
            linestyle="--",
            linewidth=1.9,
            alpha=0.95,
        )
    )
    return save_curve_plot(
        curves,
        output_path,
        figure_title="Normalized Radial Spectral Distribution (DC Excluded)",
        x_label="Frequency radius r (excluding the centered DC bin)",
        y_label="Normalized radial power density E(r) / Σ_{r>0} E(r)",
        yscale="log",
        legend_title="Curves",
    )


def main() -> int:
    """Run the noise-versus-frequency experiment."""
    args = parse_args()
    image, source_path = load_experiment_image(
        args.image_path,
        default_path=REPO_ROOT / "assets" / "default_fft_reference.png",
    )
    analyses = analyze_noise_progression(image, seed=args.seed)
    (
        main_figure_path,
        radial_figure_path,
        normalized_radial_figure_path,
    ) = build_output_paths(
        source_path,
        output_dir=args.output_dir.expanduser().resolve(),
        seed=args.seed,
    )

    save_noise_grid(analyses, main_figure_path)
    save_radial_energy_figure(analyses, radial_figure_path)
    save_normalized_radial_distribution_figure(analyses, normalized_radial_figure_path)

    print(f"Loaded image: {source_path}")
    print(f"Random seed: {args.seed}")
    print(f"Saved grid figure: {main_figure_path}")
    print(f"Saved radial-energy figure: {radial_figure_path}")
    print(f"Saved normalized radial-distribution figure: {normalized_radial_figure_path}")
    print()
    print("Display normalization:")
    print("- Noisy images are clipped only for display; the additive noise is not clipped in computation.")
    print(
        "- Log spectra use log1p(magnitude + 1e-12), then a shared global "
        "99.5th-percentile normalization after channel averaging."
    )
    print("- Raw radial curves show annulus-averaged power E(r) on a log-scaled y-axis.")
    print(
        "- Normalized radial curves divide each profile by Σ_r E(r), which isolates "
        "how relative power is distributed across frequency radii."
    )
    print(
        "- The normalized plot excludes the centered DC bin before normalization so "
        "the comparison focuses on non-constant spatial structure."
    )
    print(
        "- The dashed reference line marks the flat normalized profile expected for "
        "white noise across the remaining radial bins."
    )
    print()
    print("At sigma = 0, the spectrum reflects the original image structure.")
    print("As sigma increases, the noisy image loses visible structure in pixel space.")
    print("Gaussian noise raises spectral energy across many frequency bands.")
    print("The normalized radial plot shows that larger sigma values spread relative power more uniformly across frequencies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
