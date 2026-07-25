"""Figure-building helpers for research-facing outputs."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_DEFAULT_MPLCONFIGDIR = (
    Path(tempfile.gettempdir()) / "spectral_diffusion_playground_mplconfig"
)
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


@dataclass(frozen=True, slots=True)
class LineCurve:
    """A single labeled curve in a line plot."""

    label: str
    x: np.ndarray
    y: np.ndarray
    color: str | None = None
    linestyle: str = "-"
    linewidth: float = 2.3
    alpha: float = 1.0
    marker: str | None = None
    y_lower: np.ndarray | None = None
    y_upper: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CurvePanel:
    """A titled collection of curves sharing one plotting axis."""

    title: str
    curves: Sequence[LineCurve]


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


def normalize_scalar_field(
    field: np.ndarray, *, percentile: float = 99.5
) -> np.ndarray:
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


def normalize_scalar_fields(
    fields: Sequence[np.ndarray],
    *,
    normalization: str = "global_percentile",
    percentile: float = 99.5,
) -> list[np.ndarray]:
    """Normalize multiple scalar fields with one shared display scale.

    This is useful when several related spectra must be compared side by side
    without each panel choosing its own independent brightness scale.
    """
    if not fields:
        raise ValueError("fields must contain at least one scalar field.")

    collapsed_fields = [collapse_channels(field).astype(np.float64) for field in fields]
    flattened = np.concatenate([field.ravel() for field in collapsed_fields])
    offset = float(flattened.min())
    shifted_fields = [field - offset for field in collapsed_fields]
    shifted_flattened = flattened - offset

    if normalization == "global_max":
        scale = float(shifted_flattened.max())
    elif normalization == "global_percentile":
        scale = float(np.percentile(shifted_flattened, percentile))
    else:
        raise ValueError(
            "normalization must be either 'global_max' or 'global_percentile'."
        )

    if scale <= 0.0:
        scale = float(shifted_flattened.max())
        if scale <= 0.0:
            scale = 1.0

    return [np.clip(field / scale, 0.0, 1.0) for field in shifted_fields]


def normalize_signed_fields(
    fields: Sequence[np.ndarray],
    *,
    percentile: float = 99.5,
) -> list[np.ndarray]:
    """Map signed fields to display space with one symmetric shared scale.

    Zero maps to neutral gray at ``0.5``. Values at ``-scale`` and ``+scale``
    map to black and white, respectively, where ``scale`` is the requested
    percentile of absolute values pooled across every field and channel.
    Values beyond that range are clipped for display only.
    """
    if not fields:
        raise ValueError("fields must contain at least one signed field.")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in the interval (0, 100].")

    field_arrays = [np.asarray(field, dtype=np.float64) for field in fields]
    if any(field.ndim not in (2, 3) for field in field_arrays):
        shapes = [field.shape for field in field_arrays]
        raise ValueError(f"Expected grayscale or RGB fields, but received {shapes}.")

    absolute_values = np.concatenate([np.abs(field).ravel() for field in field_arrays])
    scale = float(np.percentile(absolute_values, percentile))
    if scale <= 0.0:
        scale = float(absolute_values.max())
        if scale <= 0.0:
            scale = 1.0

    return [np.clip(0.5 + field / (2.0 * scale), 0.0, 1.0) for field in field_arrays]


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
    row_labels: Sequence[str] | None = None,
    figure_title_size: float = 15.0,
    figure_title_y: float = 0.972,
    column_width: float = 4.2,
    row_height: float = 4.0,
    dpi: int = 220,
) -> Path:
    """Render and save a clean grid of image panels."""
    if not panel_rows or not panel_rows[0]:
        raise ValueError("panel_rows must contain at least one row and one column.")

    column_count = len(panel_rows[0])
    if any(len(row) != column_count for row in panel_rows):
        raise ValueError("Every panel row must have the same number of columns.")
    if row_labels is not None and len(row_labels) != len(panel_rows):
        raise ValueError("row_labels must match the number of panel rows.")

    _apply_publication_style()

    row_count = len(panel_rows)
    figure_width = column_width * column_count
    figure_height = row_height * row_count
    fig, axes = plt.subplots(
        row_count, column_count, figsize=(figure_width, figure_height)
    )

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

    fig.suptitle(
        figure_title, fontsize=figure_title_size, fontweight="bold", y=figure_title_y
    )
    left_margin = 0.045 if row_labels is not None else 0.0
    fig.tight_layout(rect=(left_margin, 0.0, 1.0, 0.962))
    if row_labels is not None:
        for row_index, row_label in enumerate(row_labels):
            first_axis = axes_array[row_index, 0]
            bbox = first_axis.get_position()
            fig.text(
                max(bbox.x0 - 0.02, 0.01),
                0.5 * (bbox.y0 + bbox.y1),
                row_label,
                rotation=90,
                va="center",
                ha="right",
                fontsize=12.5,
                fontweight="bold",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def save_curve_plot(
    curves: Sequence[LineCurve],
    output_path: Path,
    *,
    figure_title: str,
    x_label: str,
    y_label: str,
    yscale: str = "linear",
    legend_title: str | None = None,
    figure_title_size: float = 14.0,
    dpi: int = 220,
) -> Path:
    """Render and save a publication-quality line plot."""
    if not curves:
        raise ValueError("curves must contain at least one line.")

    _apply_publication_style()

    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for curve in curves:
        x_values = np.asarray(curve.x, dtype=np.float64)
        y_values = np.asarray(curve.y, dtype=np.float64)
        if (curve.y_lower is None) != (curve.y_upper is None):
            raise ValueError("y_lower and y_upper must be provided together.")
        if curve.y_lower is not None and curve.y_upper is not None:
            axis.fill_between(
                x_values,
                np.asarray(curve.y_lower, dtype=np.float64),
                np.asarray(curve.y_upper, dtype=np.float64),
                color=curve.color,
                alpha=0.16,
                linewidth=0.0,
            )
        axis.plot(
            x_values,
            y_values,
            label=curve.label,
            color=curve.color,
            linewidth=curve.linewidth,
            linestyle=curve.linestyle,
            alpha=curve.alpha,
            marker=curve.marker,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(figure_title, fontsize=figure_title_size, fontweight="bold", pad=12)
    axis.set_yscale(yscale)
    axis.grid(True, which="major", alpha=0.25, linewidth=0.85)
    if yscale == "log":
        axis.grid(True, which="minor", alpha=0.12, linewidth=0.65)
    axis.margins(x=0.01)
    axis.legend(frameon=False, title=legend_title, ncol=min(3, len(curves)))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def save_curve_panel_grid(
    panels: Sequence[CurvePanel],
    output_path: Path,
    *,
    figure_title: str,
    x_label: str,
    y_label: str,
    y_limits: tuple[float, float] | None = None,
    figure_title_size: float = 14.0,
    dpi: int = 220,
) -> Path:
    """Save a horizontal grid of curve panels with shared axis limits."""
    if not panels:
        raise ValueError("panels must contain at least one curve panel.")
    if any(not panel.curves for panel in panels):
        raise ValueError("Every curve panel must contain at least one curve.")

    _apply_publication_style()
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.1 * len(panels), 4.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for panel_index, panel in enumerate(panels):
        axis = axes[0, panel_index]
        for curve in panel.curves:
            x_values = np.asarray(curve.x, dtype=np.float64)
            if (curve.y_lower is None) != (curve.y_upper is None):
                raise ValueError("y_lower and y_upper must be provided together.")
            if curve.y_lower is not None and curve.y_upper is not None:
                axis.fill_between(
                    x_values,
                    np.asarray(curve.y_lower, dtype=np.float64),
                    np.asarray(curve.y_upper, dtype=np.float64),
                    color=curve.color,
                    alpha=0.16,
                    linewidth=0.0,
                )
            axis.plot(
                x_values,
                np.asarray(curve.y, dtype=np.float64),
                label=curve.label,
                color=curve.color,
                linewidth=curve.linewidth,
                linestyle=curve.linestyle,
                alpha=curve.alpha,
                marker=curve.marker,
            )
        axis.set_title(panel.title, pad=10)
        axis.set_xlabel(x_label)
        axis.grid(True, which="major", alpha=0.25, linewidth=0.85)
        axis.margins(x=0.01)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        if panel_index == 0:
            axis.set_ylabel(y_label)
        axis.legend(frameon=False, loc="lower right")

    fig.suptitle(figure_title, fontsize=figure_title_size, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
