"""Calibrate frozen frequency-band recovery metrics across natural images.

This experiment applies the controlled trajectories from Experiment 4 to the
provenance-recorded natural-image set. It quantifies image-to-image variation,
cutoff sensitivity, threshold ordering, and bootstrap uncertainty without
changing the metric, preprocessing, trajectories, or frequency cutoffs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
from PIL import __version__ as PILLOW_VERSION

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_ID: Final[str] = "experiment_05"
CONSTRUCTION_RADIUS: Final[float] = 40.0
EVALUATION_RADII: Final[tuple[float, ...]] = (20.0, 40.0, 80.0)
RECOVERY_SEEDS: Final[tuple[int, ...]] = (0,)
STEPS: Final[int] = 101
NOISE_LEVEL: Final[float] = 0.05
RECOVERY_THRESHOLD: Final[float] = 0.8
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 20_250_725
CONFIDENCE_LEVEL: Final[float] = 0.95
LOW_COLOR: Final[str] = "#0f766e"
HIGH_COLOR: Final[str] = "#d97706"
IMAGE_COLORS: Final[tuple[str, ...]] = (
    "#1d4ed8",
    "#0f766e",
    "#b45309",
    "#be123c",
    "#6d28d9",
    "#334155",
)
TRAJECTORIES: Final[tuple[tuple[str, str], ...]] = (
    ("low_band_first", "Low Band First"),
    ("high_band_first", "High Band First"),
    ("together", "Together"),
)

from spectral_diffusion_playground.calibration import (
    CutoffCurve,
    bootstrap_mean_interval,
    evaluate_cutoff_curves,
    first_threshold_crossing,
    ordering_matches_control,
)
from spectral_diffusion_playground.dataset import (
    load_natural_image_metadata,
    load_preprocessed_natural_image,
    validate_natural_image_dataset,
)
from spectral_diffusion_playground.metrics import frequency_band_components


@dataclass(frozen=True, slots=True)
class ImageCurve:
    """One image-specific controlled recovery curve."""

    image_id: str
    curve: CutoffCurve


@dataclass(frozen=True, slots=True)
class CrossingRecord:
    """Threshold timings and ordering outcome for one image-specific curve."""

    experiment_id: str
    image_id: str
    split: str
    checkpoint: str
    trajectory: str
    cutoff: float
    seed: int
    recovery_threshold: float
    t_low: float
    t_high: float
    delta_t: float
    ordering_success: bool


def parse_args() -> argparse.Namespace:
    """Parse paths without exposing frozen scientific settings as CLI options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen natural-image calibration of S_low and S_high. "
            "Scientific settings are fixed by the Experiment 5 protocol."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=REPO_ROOT / "assets" / "examples" / "metadata.csv",
        help="Path to the frozen provenance metadata.",
    )
    parser.add_argument(
        "--image-directory",
        type=Path,
        default=REPO_ROOT / "assets" / "examples" / "natural",
        help="Directory containing the frozen natural-image sources.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for machine-readable calibration outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
        help="Directory for calibration figures.",
    )
    return parser.parse_args()


def collect_image_curves(
    metadata_rows: list[dict[str, str]],
    image_directory: Path,
    *,
    progress: np.ndarray,
) -> tuple[tuple[ImageCurve, ...], tuple[dict[str, object], ...]]:
    """Preprocess every image and evaluate the frozen controlled trajectories."""
    records: list[ImageCurve] = []
    band_energy: list[dict[str, object]] = []
    for row in metadata_rows:
        image_id = row["image_id"]
        image = load_preprocessed_natural_image(image_directory / row["filename"])
        curves = evaluate_cutoff_curves(
            image,
            progress=progress,
            construction_radius=CONSTRUCTION_RADIUS,
            evaluation_radii=EVALUATION_RADII,
            seeds=RECOVERY_SEEDS,
            noise_level=NOISE_LEVEL,
        )
        records.extend(ImageCurve(image_id=image_id, curve=curve) for curve in curves)

        for cutoff in EVALUATION_RADII:
            low_band, high_band = frequency_band_components(
                image,
                radius=cutoff,
                exclude_dc=True,
            )
            low_energy = float(np.sum(np.square(low_band)))
            high_energy = float(np.sum(np.square(high_band)))
            total_energy = low_energy + high_energy
            band_energy.append(
                {
                    "image_id": image_id,
                    "cutoff": cutoff,
                    "low_energy_fraction": low_energy / total_energy,
                    "high_energy_fraction": high_energy / total_energy,
                }
            )
    return tuple(records), tuple(band_energy)


def compute_crossings(
    image_curves: tuple[ImageCurve, ...],
    *,
    progress_step: float,
) -> tuple[CrossingRecord, ...]:
    """Compute threshold crossings without treating failures as fatal."""
    records: list[CrossingRecord] = []
    for image_curve in image_curves:
        curve = image_curve.curve
        low_crossing = first_threshold_crossing(
            curve.progress,
            curve.low_score,
            threshold=RECOVERY_THRESHOLD,
        )
        high_crossing = first_threshold_crossing(
            curve.progress,
            curve.high_score,
            threshold=RECOVERY_THRESHOLD,
        )
        records.append(
            CrossingRecord(
                experiment_id=EXPERIMENT_ID,
                image_id=image_curve.image_id,
                split="calibration",
                checkpoint="not_applicable",
                trajectory=curve.schedule.name,
                cutoff=curve.evaluation_radius,
                seed=curve.seed,
                recovery_threshold=RECOVERY_THRESHOLD,
                t_low=low_crossing,
                t_high=high_crossing,
                delta_t=high_crossing - low_crossing,
                ordering_success=ordering_matches_control(
                    curve.schedule.name,
                    low_crossing=low_crossing,
                    high_crossing=high_crossing,
                    progress_step=progress_step,
                ),
            )
        )
    return tuple(records)


def save_raw_scores(
    image_curves: tuple[ImageCurve, ...],
    output_path: Path,
) -> Path:
    """Write every score using the frozen future-compatible schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(
            (
                "experiment_id",
                "image_id",
                "split",
                "checkpoint",
                "trajectory",
                "axis_name",
                "axis_value",
                "cutoff",
                "seed",
                "S_low",
                "S_high",
            )
        )
        for image_curve in image_curves:
            curve = image_curve.curve
            for index, progress in enumerate(curve.progress):
                writer.writerow(
                    (
                        EXPERIMENT_ID,
                        image_curve.image_id,
                        "calibration",
                        "not_applicable",
                        curve.schedule.name,
                        "synthetic_recovery_progress",
                        f"{progress:.8f}",
                        f"{curve.evaluation_radius:.8f}",
                        str(curve.seed),
                        f"{curve.low_score[index]:.8f}",
                        f"{curve.high_score[index]:.8f}",
                    )
                )
    return output_path


def save_crossings(
    crossings: tuple[CrossingRecord, ...],
    output_path: Path,
) -> Path:
    """Write threshold timings and ordering outcomes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(CrossingRecord.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in crossings:
            row = asdict(record)
            row["cutoff"] = f"{record.cutoff:.8f}"
            row["recovery_threshold"] = f"{record.recovery_threshold:.8f}"
            row["t_low"] = f"{record.t_low:.8f}"
            row["t_high"] = f"{record.t_high:.8f}"
            row["delta_t"] = f"{record.delta_t:.8f}"
            row["ordering_success"] = str(record.ordering_success).lower()
            writer.writerow(row)
    return output_path


def _curves_for_group(
    image_curves: tuple[ImageCurve, ...],
    *,
    trajectory: str,
    cutoff: float,
) -> tuple[ImageCurve, ...]:
    """Return one curve per image for a trajectory and cutoff."""
    return tuple(
        record
        for record in image_curves
        if record.curve.schedule.name == trajectory
        and record.curve.evaluation_radius == cutoff
        and record.curve.seed == RECOVERY_SEEDS[0]
    )


def build_summary(
    image_curves: tuple[ImageCurve, ...],
    crossings: tuple[CrossingRecord, ...],
    band_energy: tuple[dict[str, object], ...],
    *,
    image_ids: tuple[str, ...],
    progress: np.ndarray,
) -> dict[str, object]:
    """Aggregate image-level uncertainty and mandatory failure analysis."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0,
        len(image_ids),
        size=(BOOTSTRAP_RESAMPLES, len(image_ids)),
    )
    ordering_summaries: list[dict[str, object]] = []
    crossing_summaries: list[dict[str, object]] = []
    aggregate_curves: list[dict[str, object]] = []

    for trajectory, _ in TRAJECTORIES:
        for cutoff in EVALUATION_RADII:
            group_crossings = tuple(
                record
                for record in crossings
                if record.trajectory == trajectory
                and record.cutoff == cutoff
                and record.seed == RECOVERY_SEEDS[0]
            )
            successes = np.asarray(
                [record.ordering_success for record in group_crossings],
                dtype=np.float64,
            )
            gaps = np.asarray(
                [record.delta_t for record in group_crossings],
                dtype=np.float64,
            )
            survival_lower, survival_upper = bootstrap_mean_interval(
                successes,
                bootstrap_indices,
                confidence_level=CONFIDENCE_LEVEL,
            )
            gap_lower, gap_upper = bootstrap_mean_interval(
                gaps,
                bootstrap_indices,
                confidence_level=CONFIDENCE_LEVEL,
            )
            ordering_summaries.append(
                {
                    "trajectory": trajectory,
                    "cutoff": cutoff,
                    "successful_images": int(successes.sum()),
                    "total_images": len(image_ids),
                    "survival_rate": float(successes.mean()),
                    "bootstrap_ci": [
                        float(survival_lower),
                        float(survival_upper),
                    ],
                }
            )
            crossing_summaries.append(
                {
                    "trajectory": trajectory,
                    "cutoff": cutoff,
                    "mean_delta_t": float(gaps.mean()),
                    "std_delta_t": float(gaps.std(ddof=1)),
                    "bootstrap_ci": [float(gap_lower), float(gap_upper)],
                }
            )

            group_curves = _curves_for_group(
                image_curves,
                trajectory=trajectory,
                cutoff=cutoff,
            )
            low_values = np.stack([record.curve.low_score for record in group_curves])
            high_values = np.stack([record.curve.high_score for record in group_curves])
            low_lower, low_upper = bootstrap_mean_interval(
                low_values,
                bootstrap_indices,
                confidence_level=CONFIDENCE_LEVEL,
            )
            high_lower, high_upper = bootstrap_mean_interval(
                high_values,
                bootstrap_indices,
                confidence_level=CONFIDENCE_LEVEL,
            )
            aggregate_curves.append(
                {
                    "trajectory": trajectory,
                    "cutoff": cutoff,
                    "progress": progress.tolist(),
                    "S_low_mean": low_values.mean(axis=0).tolist(),
                    "S_low_std": low_values.std(axis=0, ddof=1).tolist(),
                    "S_low_bootstrap_ci_lower": low_lower.tolist(),
                    "S_low_bootstrap_ci_upper": low_upper.tolist(),
                    "S_high_mean": high_values.mean(axis=0).tolist(),
                    "S_high_std": high_values.std(axis=0, ddof=1).tolist(),
                    "S_high_bootstrap_ci_lower": high_lower.tolist(),
                    "S_high_bootstrap_ci_upper": high_upper.tolist(),
                }
            )

    progress_step = float(progress[1] - progress[0])
    energy_lookup = {
        (record["image_id"], record["cutoff"]): record for record in band_energy
    }

    def failure_record(record: CrossingRecord) -> dict[str, object]:
        """Attach measured spectral energy to one crossing diagnostic."""
        diagnostic = asdict(record)
        energy = energy_lookup[(record.image_id, record.cutoff)]
        diagnostic["low_energy_fraction"] = energy["low_energy_fraction"]
        diagnostic["high_energy_fraction"] = energy["high_energy_fraction"]
        return diagnostic

    ordering_failures = [
        failure_record(record) for record in crossings if not record.ordering_success
    ]
    collapsed_separations = [
        failure_record(record)
        for record in crossings
        if record.trajectory != "together"
        and abs(record.delta_t) <= progress_step + 1e-12
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "objective": (
            "Quantify stability and uncertainty of S_low and S_high across "
            "natural images before applying them to a learned denoiser."
        ),
        "configuration": {
            "num_images": len(image_ids),
            "image_ids": list(image_ids),
            "image_resolution": [256, 256],
            "construction_radius": CONSTRUCTION_RADIUS,
            "evaluation_radii": list(EVALUATION_RADII),
            "trajectories": [name for name, _ in TRAJECTORIES],
            "steps": STEPS,
            "noise_level": NOISE_LEVEL,
            "recovery_seeds": list(RECOVERY_SEEDS),
            "recovery_threshold": RECOVERY_THRESHOLD,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "standard_deviation_ddof": 1,
            "bootstrap_unit": "image",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
            "matplotlib": matplotlib.__version__,
        },
        "ordering_survival": ordering_summaries,
        "crossing_gaps": crossing_summaries,
        "aggregate_curves": aggregate_curves,
        "band_energy": list(band_energy),
        "failure_analysis": {
            "ordering_failures": ordering_failures,
            "collapsed_separations": collapsed_separations,
            "collapse_definition": (
                "For ordered controls, |delta_t| no greater than one progress step."
            ),
            "spectral_content_note": (
                "Band-energy fractions are reported for every failure or collapse; "
                "no post hoc spectral-content threshold is applied."
            ),
            "limitations": [
                "Recovery-noise variability is not estimated because the frozen run "
                "uses one recovery seed.",
                "Image bootstrap intervals describe only this six-image calibration "
                "set and are not population-level guarantees.",
                "An empirical 6/6 survival rate bootstraps to [1,1] because every "
                "resampled image is a success; this is not a binomial confidence bound.",
            ],
            "metric_tuning_performed": False,
        },
    }


def save_summary(summary: dict[str, object], output_path: Path) -> Path:
    """Write the complete calibration summary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    return output_path


def _apply_figure_style() -> None:
    """Apply one restrained publication-style theme to all E005 figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#334155",
            "axes.titleweight": "bold",
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.45,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#f8fafc",
        }
    )


def _plot_score_pair(
    axis: plt.Axes,
    progress: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    low_std: np.ndarray | None = None,
    high_std: np.ndarray | None = None,
) -> None:
    """Plot one `S_low`/`S_high` pair with optional standard-deviation bands."""
    if low_std is not None and high_std is not None:
        axis.fill_between(
            progress,
            np.clip(low - low_std, 0.0, 1.0),
            np.clip(low + low_std, 0.0, 1.0),
            color=LOW_COLOR,
            alpha=0.14,
            linewidth=0.0,
        )
        axis.fill_between(
            progress,
            np.clip(high - high_std, 0.0, 1.0),
            np.clip(high + high_std, 0.0, 1.0),
            color=HIGH_COLOR,
            alpha=0.12,
            linewidth=0.0,
        )
    axis.plot(progress, low, color=LOW_COLOR, linewidth=2.0, label=r"$S_{low}$")
    axis.plot(
        progress,
        high,
        color=HIGH_COLOR,
        linewidth=2.0,
        linestyle="--",
        label=r"$S_{high}$",
    )
    axis.axhline(
        RECOVERY_THRESHOLD,
        color="#64748b",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True)


def save_per_image_figure(
    image_curves: tuple[ImageCurve, ...],
    image_ids: tuple[str, ...],
    output_path: Path,
) -> Path:
    """Save all per-image curves at the construction cutoff."""
    _apply_figure_style()
    fig, axes = plt.subplots(
        len(image_ids),
        len(TRAJECTORIES),
        figsize=(14.5, 17.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, image_id in enumerate(image_ids):
        for column_index, (trajectory, title) in enumerate(TRAJECTORIES):
            axis = axes[row_index, column_index]
            record = next(
                item
                for item in image_curves
                if item.image_id == image_id
                and item.curve.schedule.name == trajectory
                and item.curve.evaluation_radius == CONSTRUCTION_RADIUS
                and item.curve.seed == RECOVERY_SEEDS[0]
            )
            _plot_score_pair(
                axis,
                record.curve.progress,
                record.curve.low_score,
                record.curve.high_score,
            )
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(f"{image_id}\nRecovery score")
            if row_index == len(image_ids) - 1:
                axis.set_xlabel("Synthetic recovery progress")
    axes[0, -1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Per-Image Frequency-Band Recovery Curves "
        f"(Construction and Evaluation r = {int(CONSTRUCTION_RADIUS)})",
        fontsize=15.0,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_mean_curve_figure(
    image_curves: tuple[ImageCurve, ...],
    output_path: Path,
) -> Path:
    """Save across-image means with one-standard-deviation bands."""
    _apply_figure_style()
    fig, axes = plt.subplots(
        len(TRAJECTORIES),
        len(EVALUATION_RADII),
        figsize=(14.5, 10.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, (trajectory, title) in enumerate(TRAJECTORIES):
        for column_index, cutoff in enumerate(EVALUATION_RADII):
            axis = axes[row_index, column_index]
            group = _curves_for_group(
                image_curves,
                trajectory=trajectory,
                cutoff=cutoff,
            )
            low = np.stack([record.curve.low_score for record in group])
            high = np.stack([record.curve.high_score for record in group])
            _plot_score_pair(
                axis,
                group[0].curve.progress,
                low.mean(axis=0),
                high.mean(axis=0),
                low_std=low.std(axis=0, ddof=1),
                high_std=high.std(axis=0, ddof=1),
            )
            if row_index == 0:
                axis.set_title(f"Evaluation cutoff r = {int(cutoff)}")
            if column_index == 0:
                axis.set_ylabel(f"{title}\nMean recovery score")
            if row_index == len(TRAJECTORIES) - 1:
                axis.set_xlabel("Synthetic recovery progress")
    axes[0, -1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Natural-Image Calibration: Mean ± Standard Deviation",
        fontsize=15.0,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_cutoff_comparison_figure(
    crossings: tuple[CrossingRecord, ...],
    image_ids: tuple[str, ...],
    output_path: Path,
) -> Path:
    """Compare per-image crossing gaps and survival across cutoffs."""
    _apply_figure_style()
    fig, axes = plt.subplots(
        1,
        len(TRAJECTORIES),
        figsize=(14.5, 4.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    radius_values = np.asarray(EVALUATION_RADII)
    for panel_index, (trajectory, title) in enumerate(TRAJECTORIES):
        axis = axes[0, panel_index]
        matrix = np.asarray(
            [
                [
                    next(
                        record.delta_t
                        for record in crossings
                        if record.image_id == image_id
                        and record.trajectory == trajectory
                        and record.cutoff == cutoff
                        and record.seed == RECOVERY_SEEDS[0]
                    )
                    for cutoff in EVALUATION_RADII
                ]
                for image_id in image_ids
            ],
            dtype=np.float64,
        )
        for image_index, image_id in enumerate(image_ids):
            axis.plot(
                radius_values,
                matrix[image_index],
                color=IMAGE_COLORS[image_index % len(IMAGE_COLORS)],
                linewidth=1.0,
                marker="o",
                markersize=3.5,
                alpha=0.48,
                label=image_id if panel_index == 0 else None,
            )
        means = matrix.mean(axis=0)
        standard_deviations = matrix.std(axis=0, ddof=1)
        axis.errorbar(
            radius_values,
            means,
            yerr=standard_deviations,
            color="#0f172a",
            linewidth=2.4,
            marker="D",
            markersize=5.0,
            capsize=4,
            label="Mean ± SD" if panel_index == 0 else None,
            zorder=5,
        )
        for radius_index, cutoff in enumerate(EVALUATION_RADII):
            cutoff_records = tuple(
                record
                for record in crossings
                if record.trajectory == trajectory
                and record.cutoff == cutoff
                and record.seed == RECOVERY_SEEDS[0]
            )
            successes = sum(record.ordering_success for record in cutoff_records)
            collapsed = sum(
                trajectory != "together"
                and abs(record.delta_t) <= (1.0 / (STEPS - 1)) + 1e-12
                for record in cutoff_records
            )
            annotation = f"{successes}/{len(image_ids)}"
            if trajectory != "together":
                annotation += f"; c={collapsed}"
            axis.annotate(
                annotation,
                (cutoff, means[radius_index]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#0f172a",
            )
        axis.axhline(0.0, color="#64748b", linewidth=1.0, linestyle=":")
        axis.set_title(title)
        axis.set_xlabel("Evaluation cutoff r")
        axis.set_xticks(radius_values)
        axis.grid(True)
        if panel_index == 0:
            axis.set_ylabel(r"Crossing gap $\Delta t = t_{high} - t_{low}$")
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    fig.suptitle(
        "Cutoff Sensitivity Across Natural Images\n"
        "Annotations: ordering survival; c = collapsed separation",
        fontsize=14.0,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> int:
    """Run the frozen natural-image calibration protocol."""
    args = parse_args()
    metadata_path = args.metadata_path.expanduser().resolve()
    image_directory = args.image_directory.expanduser().resolve()
    validate_natural_image_dataset(metadata_path, image_directory)
    metadata_rows = load_natural_image_metadata(metadata_path)
    image_ids = tuple(row["image_id"] for row in metadata_rows)
    progress = np.linspace(0.0, 1.0, STEPS, dtype=np.float64)

    image_curves, band_energy = collect_image_curves(
        metadata_rows,
        image_directory,
        progress=progress,
    )
    crossings = compute_crossings(
        image_curves,
        progress_step=float(progress[1] - progress[0]),
    )
    summary = build_summary(
        image_curves,
        crossings,
        band_energy,
        image_ids=image_ids,
        progress=progress,
    )

    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    scores_path = save_raw_scores(
        image_curves,
        results_dir / "experiment_05_scores.csv",
    )
    crossings_path = save_crossings(
        crossings,
        results_dir / "experiment_05_crossings.csv",
    )
    summary_path = save_summary(
        summary,
        results_dir / "experiment_05_summary.json",
    )
    per_image_path = save_per_image_figure(
        image_curves,
        image_ids,
        output_dir / "experiment_05_per_image_curves.png",
    )
    mean_path = save_mean_curve_figure(
        image_curves,
        output_dir / "experiment_05_mean_curves.png",
    )
    cutoff_path = save_cutoff_comparison_figure(
        crossings,
        image_ids,
        output_dir / "experiment_05_cutoff_comparison.png",
    )

    failures = summary["failure_analysis"]
    assert isinstance(failures, dict)
    print("Experiment 5 complete: natural-image metric calibration only.")
    print(
        "Frozen settings: construction r=40; evaluation r={20,40,80}; "
        "threshold=0.8; recovery seed=0."
    )
    print(f"Ordering failures: {len(failures['ordering_failures'])}")
    print(
        f"Collapsed ordered-control separations: {len(failures['collapsed_separations'])}"
    )
    for path in (
        scores_path,
        crossings_path,
        summary_path,
        per_image_path,
        mean_path,
        cutoff_path,
    ):
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
