"""Figure-building helpers for research-facing outputs."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_DEFAULT_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "spectral_diffusion_playground_mplconfig"
_DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True, slots=True)
class ImagePanel:
    """A single panel in a multi-image figure."""

    title: str
    image: np.ndarray
    cmap: str | None = None


def prepare_image_for_display(image: np.ndarray) -> np.ndarray:
    """Clip an image-like array into a safe display range of ``[0, 1]``."""
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim not in (2, 3):
        raise ValueError(
            "Expected a grayscale or RGB image for display, "
            f"but received shape {image_array.shape}."
        )
    return np.clip(image_array, 0.0, 1.0)


def collapse_channels(image: np.ndarray) -> np.ndarray:
    """Reduce an RGB image to a single intensity map by averaging channels."""
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim == 2:
        return image_array
    if image_array.ndim != 3:
        raise ValueError(
            "Expected a grayscale or RGB image, "
            f"but received shape {image_array.shape}."
        )
    return image_array.mean(axis=2)


def normalize_scalar_field(field: np.ndarray, *, percentile: float = 99.5) -> np.ndarray:
    """Normalize a scalar field into ``[0, 1]`` using a robust percentile scale."""
    field_array = np.asarray(field, dtype=np.float64)
    if field_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D scalar field for normalization, got {field_array.shape}."
        )

    shifted = field_array - field_array.min()
    scale = np.percentile(shifted, percentile)
    if scale <= 0.0:
        max_value = shifted.max()
        scale = max_value if max_value > 0.0 else 1.0

    return np.clip(shifted / scale, 0.0, 1.0)


def spectrum_to_display_image(
    spectrum_values: np.ndarray, *, normalization: str = "max", percentile: float = 99.5
) -> np.ndarray:
    """Convert a magnitude-like spectrum into a 2D display image.

    Parameters
    ----------
    spectrum_values:
        A nonnegative scalar field or an RGB field that will be collapsed across
        channels by averaging.
    normalization:
        ``"max"`` preserves honest linear scaling by dividing by the global
        maximum after channel collapse. ``"percentile"`` uses the requested
        percentile for a more contrastive display.
    percentile:
        Percentile used only when ``normalization="percentile"``.
    """
    collapsed = collapse_channels(spectrum_values)
    if normalization == "max":
        maximum = float(np.max(collapsed))
        scale = maximum if maximum > 0.0 else 1.0
        return np.clip(collapsed / scale, 0.0, 1.0)
    if normalization == "percentile":
        return normalize_scalar_field(collapsed, percentile=percentile)
    raise ValueError(
        "normalization must be either 'max' for exact max normalization or "
        "'percentile' for robust contrast normalization."
    )


def _apply_publication_style() -> None:
    """Set Matplotlib defaults suitable for tutorial and paper-adjacent figures."""
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "font.size": 11.5,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_panel_grid(
    panel_rows: Sequence[Sequence[ImagePanel]],
    output_path: Path,
    *,
    figure_title: str,
    dpi: int = 220,
) -> Path:
    """Render and save a clean grid of image panels."""
    if not panel_rows or not panel_rows[0]:
        raise ValueError("panel_rows must contain at least one row and one column.")

    column_count = len(panel_rows[0])
    if any(len(row) != column_count for row in panel_rows):
        raise ValueError("Every panel row must have the same number of columns.")

    _apply_publication_style()

    row_count = len(panel_rows)
    figure_width = 4.2 * column_count
    figure_height = 4.0 * row_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(figure_width, figure_height))

    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 0:
        axes_array = axes_array.reshape(1, 1)
    elif axes_array.ndim == 1:
        axes_array = axes_array.reshape(row_count, column_count)

    for row_index, row in enumerate(panel_rows):
        for column_index, panel in enumerate(row):
            axis = axes_array[row_index, column_index]
            image = prepare_image_for_display(panel.image)
            axis.imshow(image, cmap=panel.cmap, vmin=0.0, vmax=1.0)
            axis.set_title(panel.title, pad=7)
            axis.axis("off")

    fig.suptitle(figure_title, fontsize=15, fontweight="bold", y=0.972)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.962))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
