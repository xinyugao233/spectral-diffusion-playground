"""Shared utilities used by experiment entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Final

import numpy as np
from PIL import Image

SCAFFOLD_STATUS: Final[str] = "Scaffold only; experiment implementation pending."
LUMINANCE_WEIGHTS: Final[np.ndarray] = np.asarray([0.2126, 0.7152, 0.0722])
NON_ALPHANUMERIC_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ExperimentStub:
    """Experiment metadata for a not-yet-implemented script."""

    script_name: str
    title: str
    question: str


def format_experiment_stub(spec: ExperimentStub) -> str:
    """Render a consistent message for an unimplemented experiment."""
    return dedent(
        f"""
        [{spec.script_name}] {spec.title}
        Question: {spec.question}
        Status: {SCAFFOLD_STATUS}
        """
    ).strip()


def run_experiment_stub(spec: ExperimentStub) -> int:
    """Print a consistent message and return a success code."""
    print(format_experiment_stub(spec))
    return 0


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image file as a floating-point RGB array in ``[0, 1]``."""
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float64) / 255.0


def load_experiment_image(
    image_path: Path | None,
    *,
    default_path: Path,
    default_size: int = 384,
) -> tuple[np.ndarray, Path]:
    """Load a user image or fall back to the deterministic default reference image."""
    if image_path is not None:
        resolved_path = image_path.expanduser().resolve()
        return load_rgb_image(resolved_path), resolved_path

    reference_path = ensure_default_reference_image(default_path, size=default_size)
    return load_rgb_image(reference_path), reference_path


def save_rgb_image(path: Path, image: np.ndarray) -> Path:
    """Save a floating-point RGB image in ``[0, 1]`` to disk."""
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with shape (H, W, 3), "
            f"but received shape {image_array.shape}."
        )

    clipped = np.clip(image_array, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="RGB").save(path)
    return path


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image in ``[0, 1]`` to a single-channel luminance image."""
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim == 2:
        return image_array
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with shape (H, W, 3), "
            f"but received shape {image_array.shape}."
        )
    return np.tensordot(image_array, LUMINANCE_WEIGHTS, axes=([2], [0]))


def create_reference_image(size: int = 384) -> np.ndarray:
    """Create a deterministic reference image with smooth regions, edges, and texture.

    The synthetic example intentionally mixes low-frequency gradients,
    piecewise-constant shapes, and periodic patterns so that its Fourier
    representation is informative even before a user provides a custom image.
    """
    if size < 64:
        raise ValueError("size must be at least 64 pixels.")

    coordinates = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)

    image = np.stack(
        [
            0.55 + 0.24 * xx - 0.10 * yy,
            0.46 + 0.18 * yy + 0.05 * np.cos(2.0 * np.pi * xx),
            0.58 - 0.16 * xx + 0.10 * yy,
        ],
        axis=-1,
    )

    plaid = np.cos(2.0 * np.pi * 8.0 * xx) * np.cos(2.0 * np.pi * 6.0 * yy)
    diagonal_wave = np.sin(2.0 * np.pi * 9.0 * (0.86 * xx + 0.5 * yy))
    radial_ring = np.exp(-((radius - 0.45) / 0.08) ** 2)
    ripple = np.sin(18.0 * radius - 3.0 * angle)

    image[..., 0] += 0.18 * radial_ring + 0.08 * diagonal_wave
    image[..., 1] += 0.14 * plaid * (radius < 0.85)
    image[..., 2] += 0.10 * ripple

    warm_disc = radius < 0.28
    image[warm_disc] = 0.65 * image[warm_disc] + 0.35 * np.array([0.96, 0.62, 0.24])

    diagonal_band = np.abs(yy - 0.35 * xx + 0.12) < 0.06
    image[diagonal_band] = 0.55 * image[diagonal_band] + 0.45 * np.array(
        [0.96, 0.89, 0.34]
    )

    cool_square = (np.abs(xx - 0.42) < 0.13) & (np.abs(yy + 0.40) < 0.13)
    image[cool_square] = 0.60 * image[cool_square] + 0.40 * np.array(
        [0.17, 0.35, 0.92]
    )

    cyan_circle = (xx + 0.52) ** 2 + (yy - 0.34) ** 2 < 0.12**2
    image[cyan_circle] = 0.40 * image[cyan_circle] + 0.60 * np.array(
        [0.16, 0.83, 0.90]
    )

    emerald_glow = np.exp(-((xx + 0.48) ** 2 + (yy - 0.38) ** 2) / 0.015)
    image[..., 1] += 0.10 * emerald_glow

    sapphire_glow = np.exp(-((xx - 0.40) ** 2 + (yy + 0.42) ** 2) / 0.018)
    image[..., 2] += 0.12 * sapphire_glow

    return np.clip(image, 0.0, 1.0)


def ensure_default_reference_image(path: Path, *, size: int = 384) -> Path:
    """Ensure that the default FFT reference image exists on disk."""
    if not path.exists():
        save_rgb_image(path, create_reference_image(size=size))
    return path


def slugify_stem(path: Path) -> str:
    """Create a filesystem-friendly lowercase stem from a source path."""
    normalized = NON_ALPHANUMERIC_PATTERN.sub("_", path.stem.lower()).strip("_")
    return normalized or "image"
