"""Reveal how image structure returns as the retained frequency radius grows.

This experiment decomposes a centered Fourier spectrum with circular low-pass
masks. It visualizes both the retained coefficients and the corresponding
inverse-FFT reconstructions, then measures the unrecovered image content with a
relative L2 error curve.
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
EXPERIMENT_TITLE = "Where Does Image Information Live in Frequency Space?"
DEFAULT_RADII: Final[tuple[float, ...]] = (10.0, 20.0, 40.0, 80.0, 120.0)

from spectral_diffusion_playground.fft import compute_fft, compute_ifft, shift_fft
from spectral_diffusion_playground.filters import (
    create_frequency_mask,
    high_pass_filter,
    low_pass_filter,
)
from spectral_diffusion_playground.utils import load_experiment_image
from spectral_diffusion_playground.visualization import (
    ImagePanel,
    LineCurve,
    normalize_signed_fields,
    prepare_image_for_display,
    save_curve_plot,
    save_panel_grid,
)


@dataclass(frozen=True, slots=True)
class FrequencyReconstruction:
    """Reconstruction and error produced by one circular frequency cutoff."""

    radius: float
    mask: np.ndarray
    reconstruction: np.ndarray
    high_frequency_residual: np.ndarray
    relative_l2_error: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for frequency decomposition."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an image from progressively larger centered Fourier "
            "regions and measure the remaining reconstruction error."
        )
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional input image. If omitted, the default reference image is used.",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=list(DEFAULT_RADII),
        help="Increasing low-pass cutoff radii in centered Fourier pixels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
        help="Directory where generated figures will be saved.",
    )
    return parser.parse_args()


def validate_radii(radii: list[float]) -> tuple[float, ...]:
    """Validate and freeze the requested sequence of frequency radii."""
    radius_array = np.asarray(radii, dtype=np.float64)
    if radius_array.ndim != 1 or radius_array.size == 0:
        raise ValueError("--radii must contain at least one value.")
    if np.any(~np.isfinite(radius_array)) or np.any(radius_array < 0.0):
        raise ValueError("--radii values must be finite and nonnegative.")
    if np.any(np.diff(radius_array) <= 0.0):
        raise ValueError("--radii values must be strictly increasing.")
    return tuple(float(radius) for radius in radius_array)


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return ``||reference - estimate||_2 / ||reference||_2``.

    The arrays are measured before any display clipping. Relative error makes
    results comparable across images and resolutions.
    """
    reference_array = np.asarray(reference, dtype=np.float64)
    estimate_array = np.asarray(estimate, dtype=np.float64)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("reference and estimate must have identical shapes.")

    reference_norm = float(np.linalg.norm(reference_array.ravel()))
    if reference_norm == 0.0:
        raise ValueError("reference must have nonzero L2 norm.")
    residual_norm = float(np.linalg.norm((reference_array - estimate_array).ravel()))
    return residual_norm / reference_norm


def analyze_frequency_radii(
    image: np.ndarray,
    radii: tuple[float, ...],
) -> list[FrequencyReconstruction]:
    """Build low-pass reconstructions and errors for each requested radius."""
    centered_spectrum = shift_fft(compute_fft(image))
    height, width = image.shape[:2]
    analyses: list[FrequencyReconstruction] = []

    for radius in radii:
        mask = create_frequency_mask(height, width, radius)
        filtered_spectrum = low_pass_filter(centered_spectrum, radius)
        reconstruction = compute_ifft(filtered_spectrum, is_shifted=True)
        residual_spectrum = high_pass_filter(centered_spectrum, radius)
        high_frequency_residual = compute_ifft(
            residual_spectrum,
            is_shifted=True,
        )
        analyses.append(
            FrequencyReconstruction(
                radius=radius,
                mask=mask,
                reconstruction=reconstruction,
                high_frequency_residual=high_frequency_residual,
                relative_l2_error=relative_l2_error(image, reconstruction),
            )
        )

    return analyses


def _format_radius(radius: float) -> str:
    """Format a radius compactly for panel titles."""
    return str(int(radius)) if radius.is_integer() else f"{radius:g}"


def save_decomposition_grid(
    image: np.ndarray,
    analyses: list[FrequencyReconstruction],
    output_path: Path,
) -> Path:
    """Save the original, low-pass reconstructions, and matching masks."""
    reconstruction_panels = [
        ImagePanel(title="Original", image=prepare_image_for_display(image))
    ]
    reconstruction_panels.extend(
        ImagePanel(
            title=f"r = {_format_radius(analysis.radius)}",
            image=prepare_image_for_display(analysis.reconstruction),
        )
        for analysis in analyses
    )

    mask_panels = [
        ImagePanel(
            title="All Frequencies",
            image=np.ones(image.shape[:2], dtype=np.float64),
            cmap="magma",
        )
    ]
    mask_panels.extend(
        ImagePanel(
            title=f"r = {_format_radius(analysis.radius)}",
            image=analysis.mask,
            cmap="magma",
        )
        for analysis in analyses
    )

    return save_panel_grid(
        [reconstruction_panels, mask_panels],
        output_path,
        figure_title=EXPERIMENT_TITLE,
        row_labels=("Reconstruction", "Frequency Mask"),
        figure_title_size=14.0,
        figure_title_y=0.982,
        column_width=3.15,
        row_height=3.2,
    )


def save_high_frequency_residual_grid(
    analyses: list[FrequencyReconstruction],
    output_path: Path,
) -> Path:
    """Save complementary high-frequency residuals with shared display scaling."""
    display_residuals = normalize_signed_fields(
        [analysis.high_frequency_residual for analysis in analyses],
        percentile=99.5,
    )
    panels = [
        ImagePanel(
            title=f"r = {_format_radius(analysis.radius)}",
            image=display_residuals[index],
        )
        for index, analysis in enumerate(analyses)
    ]
    return save_panel_grid(
        [panels],
        output_path,
        figure_title="What the Low-Pass Reconstruction Leaves Behind",
        row_labels=("High-Frequency Residual",),
        figure_title_size=14.0,
        figure_title_y=0.965,
        column_width=3.2,
        row_height=3.35,
    )


def save_reconstruction_error_plot(
    analyses: list[FrequencyReconstruction],
    output_path: Path,
) -> Path:
    """Save relative reconstruction error as a function of cutoff radius."""
    radii = np.asarray([analysis.radius for analysis in analyses], dtype=np.float64)
    errors = np.asarray(
        [analysis.relative_l2_error for analysis in analyses],
        dtype=np.float64,
    )
    curve = LineCurve(
        label="Low-pass reconstruction",
        x=radii,
        y=errors,
        color="#0f766e",
        linewidth=2.5,
        marker="o",
    )
    return save_curve_plot(
        [curve],
        output_path,
        figure_title="Reconstruction Error vs. Frequency Radius",
        x_label="Low-pass cutoff radius r (Fourier pixels)",
        y_label="Relative L2 error ||x - x_r||₂ / ||x||₂",
    )


def main() -> int:
    """Run the frequency-decomposition experiment."""
    args = parse_args()
    radii = validate_radii(args.radii)
    image, source_path = load_experiment_image(
        args.image_path,
        default_path=REPO_ROOT / "assets" / "default_fft_reference.png",
    )
    analyses = analyze_frequency_radii(image, radii)

    output_dir = args.output_dir.expanduser().resolve()
    grid_path = output_dir / "where_image_information_lives_grid.png"
    residual_path = output_dir / "high_frequency_residuals.png"
    error_path = output_dir / "reconstruction_error_vs_frequency_radius.png"
    save_decomposition_grid(image, analyses, grid_path)
    save_high_frequency_residual_grid(analyses, residual_path)
    save_reconstruction_error_plot(analyses, error_path)

    print(f"Loaded image: {source_path}")
    print(f"Frequency radii: {', '.join(_format_radius(radius) for radius in radii)}")
    print(f"Saved decomposition grid: {grid_path}")
    print(f"Saved high-frequency residual grid: {residual_path}")
    print(f"Saved reconstruction-error figure: {error_path}")
    print()
    print("Measurement:")
    print("- Circular masks use centered FFT coordinates and retain distances <= r.")
    print("- Reconstructions are measured before display clipping.")
    print("- Error is relative L2: ||x - x_r||₂ / ||x||₂.")
    print("- Nested masks imply non-increasing L2 error under the orthonormal FFT.")
    print(
        "- Residual panels show inverse-FFT high-pass components with one shared "
        "symmetric 99.5th-percentile display scale centered at gray."
    )
    for analysis in analyses:
        print(
            f"  r = {_format_radius(analysis.radius):>3}: "
            f"{analysis.relative_l2_error:.6f}"
        )
    print()
    print(
        "Low-frequency components first recover coarse geometry and smooth variation."
    )
    print("Increasing the radius restores progressively finer spatial detail.")
    print("This experiment does not establish that low frequencies contain semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
