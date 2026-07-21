"""Understand how an image moves into and out of Fourier space.

This script is intentionally educational. It shows how a 2D Fourier transform
maps an image from pixel space into frequency space, then uses the inverse
transform to return to pixel space without losing information.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import the shared bootstrap so this script can be run directly from the repo root.
import _bootstrap  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_TITLE = "Understanding Images in Fourier Space"

from spectral_diffusion_playground.fft import (
    compute_fft,
    compute_ifft,
    log_magnitude,
    magnitude_spectrum,
    shift_fft,
)
from spectral_diffusion_playground.utils import (
    ensure_default_reference_image,
    load_rgb_image,
    rgb_to_grayscale,
)
from spectral_diffusion_playground.visualization import (
    ImagePanel,
    prepare_image_for_display,
    save_panel_grid,
    spectrum_to_display_image,
)


@dataclass(frozen=True, slots=True)
class FourierViews:
    """The main image-space and frequency-space views used in the figure."""

    name: str
    original: np.ndarray
    centered_magnitude: np.ndarray
    centered_log_magnitude: np.ndarray
    reconstruction: np.ndarray
    reconstruction_error: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how an image is represented in Fourier space and how the "
            "inverse FFT returns it to pixel space."
        )
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional path to an input image. If omitted, a default reference image is used.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path for the saved figure. Defaults to figures/experiment01_*.png.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Also render a grayscale row to compare RGB and grayscale spectra.",
    )
    return parser.parse_args()


def load_experiment_image(image_path: Path | None) -> tuple[np.ndarray, Path]:
    """Load a user image or fall back to the repository's default reference image."""
    if image_path is not None:
        resolved_path = image_path.expanduser().resolve()
        image = load_rgb_image(resolved_path)
        return image, resolved_path

    default_path = REPO_ROOT / "assets" / "default_fft_reference.png"
    reference_path = ensure_default_reference_image(default_path)
    image = load_rgb_image(reference_path)
    return image, reference_path


def compute_fourier_views(image: np.ndarray, *, name: str) -> FourierViews:
    """Compute the forward and inverse FFT views needed for visualization.

    The linear magnitude panel uses the centered magnitude spectrum with exact
    max normalization after channel averaging. The log panel applies
    ``log1p(magnitude + 1e-12)`` first, then uses the same max normalization.
    """
    spectrum = compute_fft(image)
    shifted_spectrum = shift_fft(spectrum)
    magnitude = magnitude_spectrum(shifted_spectrum)
    log_magnitude_spectrum = log_magnitude(magnitude)
    reconstruction = compute_ifft(shifted_spectrum, is_shifted=True)
    reconstruction_error = float(np.max(np.abs(reconstruction - image)))

    return FourierViews(
        name=name,
        original=prepare_image_for_display(image),
        centered_magnitude=spectrum_to_display_image(magnitude, normalization="max"),
        centered_log_magnitude=spectrum_to_display_image(
            log_magnitude_spectrum,
            normalization="max",
        ),
        reconstruction=prepare_image_for_display(reconstruction),
        reconstruction_error=reconstruction_error,
    )


def make_panel_block(views: FourierViews) -> list[list[ImagePanel]]:
    """Create a compact two-row panel block for a Fourier visualization."""
    original_cmap = "gray" if views.original.ndim == 2 else None
    reconstruction_cmap = "gray" if views.reconstruction.ndim == 2 else None

    return [
        [
            ImagePanel(
                title="Original Image",
                image=views.original,
                cmap=original_cmap,
            ),
            ImagePanel(
                title="Centered Magnitude Spectrum",
                image=views.centered_magnitude,
                cmap="magma",
            ),
        ],
        [
            ImagePanel(
                title="Log-Magnitude Spectrum",
                image=views.centered_log_magnitude,
                cmap="magma",
            ),
            ImagePanel(
                title="Inverse FFT Reconstruction",
                image=views.reconstruction,
                cmap=reconstruction_cmap,
            ),
        ],
    ]


def build_output_path(image_path: Path, *, grayscale: bool, override: Path | None) -> Path:
    """Resolve a meaningful output filename inside ``figures/`` unless overridden."""
    if override is not None:
        return override.expanduser().resolve()

    safe_stem = image_path.stem.lower().replace(" ", "_")
    suffix = "_rgb_and_grayscale" if grayscale else "_rgb"
    filename = f"understanding_images_in_fourier_space_{safe_stem}{suffix}.png"
    return REPO_ROOT / "figures" / filename


def main() -> int:
    """Run the Fourier-space visualization experiment."""
    args = parse_args()
    image, source_path = load_experiment_image(args.image_path)

    rgb_views = compute_fourier_views(image, name="RGB")
    panel_rows: list[list[ImagePanel]] = make_panel_block(rgb_views)

    grayscale_views: FourierViews | None = None
    if args.grayscale:
        grayscale_image = rgb_to_grayscale(image)
        grayscale_views = compute_fourier_views(grayscale_image, name="Grayscale")
        panel_rows.extend(make_panel_block(grayscale_views))

    output_path = build_output_path(
        source_path,
        grayscale=args.grayscale,
        override=args.output_path,
    )

    save_panel_grid(panel_rows, output_path, figure_title=EXPERIMENT_TITLE)

    print(f"Loaded image: {source_path}")
    print(f"Saved figure: {output_path}")
    print(f"RGB reconstruction max absolute error: {rgb_views.reconstruction_error:.3e}")
    if grayscale_views is not None:
        print(
            "Grayscale reconstruction max absolute error: "
            f"{grayscale_views.reconstruction_error:.3e}"
        )

    print()
    print("Bright pixels near the center of the spectrum correspond to low spatial frequencies.")
    print("Pixels farther from the center correspond to higher spatial frequencies.")
    print("The inverse FFT reconstructs the original image up to floating-point precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
