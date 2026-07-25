"""Validate low- and high-frequency band recovery metrics.

This experiment does not run a denoiser. It constructs three deterministic
synthetic recovery trajectories whose low- and high-frequency contributions are
known, then checks whether the proposed scores recover the intended ordering.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

# Import the shared bootstrap so this script can be run directly from the repo root.
import _bootstrap  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_TITLE = "Measuring Low- and High-Frequency Recovery"
DEFAULT_RADIUS: Final[float] = 40.0
DEFAULT_STEPS: Final[int] = 101
DEFAULT_NOISE_LEVEL: Final[float] = 0.05
DEFAULT_SENSITIVITY_RADII: Final[tuple[float, ...]] = (20.0, 40.0, 80.0)
DEFAULT_SENSITIVITY_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
RECOVERY_THRESHOLD: Final[float] = 0.8
FRAME_COUNT: Final[int] = 5
LOW_BAND_COLOR: Final[str] = "#0f766e"
HIGH_BAND_COLOR: Final[str] = "#d97706"

from spectral_diffusion_playground.metrics import (
    FrequencyBandScores,
    frequency_band_components,
    frequency_band_recovery_scores,
)
from spectral_diffusion_playground.utils import load_experiment_image
from spectral_diffusion_playground.visualization import (
    CurvePanel,
    ImagePanel,
    LineCurve,
    prepare_image_for_display,
    save_curve_panel_grid,
    save_panel_grid,
)


@dataclass(frozen=True, slots=True)
class RecoverySchedule:
    """Known low- and high-frequency weights for one controlled scenario."""

    name: str
    title: str
    low_weight: np.ndarray
    high_weight: np.ndarray


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Measured curves and selected frames for one controlled trajectory."""

    schedule: RecoverySchedule
    progress: np.ndarray
    low_score: np.ndarray
    high_score: np.ndarray
    low_relative_error: np.ndarray
    high_relative_error: np.ndarray
    frame_progress: np.ndarray
    frames: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class SensitivityRecord:
    """Threshold crossings for one scenario, cutoff, and noise seed."""

    scenario_name: str
    scenario_title: str
    evaluation_radius: float
    seed: int
    low_crossing: float
    high_crossing: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for metric validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate low- and high-frequency recovery scores on controlled "
            "synthetic trajectories before applying them to a real denoiser."
        )
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional clean target image. The deterministic reference is the fallback.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help="Circular low-pass cutoff in centered Fourier pixels.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of equally spaced trajectory points.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=DEFAULT_NOISE_LEVEL,
        help="Initial relative L2 noise level applied independently to each band.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the fixed band-balanced noise realization.",
    )
    parser.add_argument(
        "--sensitivity-radii",
        type=float,
        nargs="+",
        default=DEFAULT_SENSITIVITY_RADII,
        help=(
            "Measurement cutoffs used to re-evaluate the same controlled "
            "trajectories."
        ),
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SENSITIVITY_SEEDS,
        help="Noise seeds used to estimate cutoff-calibration uncertainty.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for raw metric values.",
    )
    return parser.parse_args()


def validate_configuration(
    image: np.ndarray,
    *,
    radius: float,
    steps: int,
    noise_level: float,
) -> None:
    """Validate parameters that affect the metric interpretation."""
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("--radius must be finite and nonnegative.")
    height, width = image.shape[:2]
    maximum_radius = float(np.hypot(height // 2, width // 2))
    if radius >= maximum_radius:
        raise ValueError(
            f"--radius must be below the corner radius {maximum_radius:.3f} "
            "so the high-frequency band remains nonempty."
        )
    if steps < FRAME_COUNT:
        raise ValueError(f"--steps must be at least {FRAME_COUNT}.")
    if not np.isfinite(noise_level) or noise_level < 0.0:
        raise ValueError("--noise-level must be finite and nonnegative.")


def validate_sensitivity_configuration(
    image: np.ndarray,
    *,
    radii: tuple[float, ...],
    seeds: tuple[int, ...],
) -> None:
    """Validate cutoff-sensitivity parameters and require unique values."""
    if not radii:
        raise ValueError("--sensitivity-radii must contain at least one value.")
    if len(set(radii)) != len(radii):
        raise ValueError("--sensitivity-radii must not contain duplicates.")
    if tuple(sorted(radii)) != radii:
        raise ValueError("--sensitivity-radii must be strictly increasing.")
    if not seeds:
        raise ValueError("--sensitivity-seeds must contain at least one value.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--sensitivity-seeds must not contain duplicates.")
    for radius in radii:
        validate_configuration(
            image,
            radius=radius,
            steps=FRAME_COUNT,
            noise_level=0.0,
        )


def smoothstep_window(
    progress: np.ndarray,
    *,
    start: float,
    end: float,
) -> np.ndarray:
    """Return a smooth transition from zero to one over ``[start, end]``."""
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Expected 0 <= start < end <= 1.")
    normalized = np.clip((progress - start) / (end - start), 0.0, 1.0)
    return normalized**2 * (3.0 - 2.0 * normalized)


def build_recovery_schedules(progress: np.ndarray) -> tuple[RecoverySchedule, ...]:
    """Create low-band-first, high-band-first, and simultaneous controls."""
    early = smoothstep_window(progress, start=0.05, end=0.55)
    late = smoothstep_window(progress, start=0.45, end=0.95)
    together = smoothstep_window(progress, start=0.10, end=0.90)
    return (
        RecoverySchedule(
            name="low_band_first",
            title="Low Band First",
            low_weight=early,
            high_weight=late,
        ),
        RecoverySchedule(
            name="high_band_first",
            title="High Band First",
            low_weight=late,
            high_weight=early,
        ),
        RecoverySchedule(
            name="together",
            title="Together",
            low_weight=together,
            high_weight=together,
        ),
    )


def _match_l2_norm(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale ``source`` to the L2 norm of ``target``."""
    source_norm = float(np.linalg.norm(np.asarray(source).ravel()))
    target_norm = float(np.linalg.norm(np.asarray(target).ravel()))
    if source_norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero-energy noise component.")
    if target_norm <= np.finfo(np.float64).eps:
        raise ValueError("The target frequency band must contain nonzero energy.")
    return np.asarray(source * (target_norm / source_norm), dtype=np.float64)


def create_band_balanced_noise(
    image_shape: tuple[int, ...],
    *,
    radius: float,
    target_low: np.ndarray,
    target_high: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Create deterministic noise with matched relative energy in both bands."""
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal(image_shape)
    noise_low, noise_high = frequency_band_components(
        white_noise,
        radius=radius,
        exclude_dc=True,
    )
    return _match_l2_norm(noise_low, target_low) + _match_l2_norm(
        noise_high,
        target_high,
    )


def analyze_schedule(
    schedule: RecoverySchedule,
    *,
    progress: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
    target_mean: np.ndarray,
    balanced_noise: np.ndarray,
    evaluation_target_low: np.ndarray,
    evaluation_target_high: np.ndarray,
    evaluation_radius: float,
    noise_level: float,
    frame_indices: np.ndarray,
) -> ScenarioResult:
    """Generate one controlled trajectory and measure both recovery curves."""
    low_scores: list[float] = []
    high_scores: list[float] = []
    low_errors: list[float] = []
    high_errors: list[float] = []
    frames: list[np.ndarray] = []
    selected_indices = set(int(index) for index in frame_indices)
    target_low_eval_low, target_low_eval_high = frequency_band_components(
        target_low,
        radius=evaluation_radius,
        exclude_dc=True,
    )
    target_high_eval_low, target_high_eval_high = frequency_band_components(
        target_high,
        radius=evaluation_radius,
        exclude_dc=True,
    )
    noise_eval_low, noise_eval_high = frequency_band_components(
        balanced_noise,
        radius=evaluation_radius,
        exclude_dc=True,
    )

    for index, trajectory_progress in enumerate(progress):
        noise_weight = noise_level * (1.0 - trajectory_progress)
        low_weight = schedule.low_weight[index]
        high_weight = schedule.high_weight[index]
        prediction_low = (
            low_weight * target_low_eval_low
            + high_weight * target_high_eval_low
            + noise_weight * noise_eval_low
        )
        prediction_high = (
            low_weight * target_low_eval_high
            + high_weight * target_high_eval_high
            + noise_weight * noise_eval_high
        )
        scores: FrequencyBandScores = frequency_band_recovery_scores(
            prediction_low,
            prediction_high,
            evaluation_target_low,
            evaluation_target_high,
        )
        low_scores.append(scores.low_score)
        high_scores.append(scores.high_score)
        low_errors.append(scores.low_relative_error)
        high_errors.append(scores.high_relative_error)
        if index in selected_indices:
            prediction = (
                target_mean
                + low_weight * target_low
                + high_weight * target_high
                + noise_weight * balanced_noise
            )
            frames.append(prepare_image_for_display(prediction))

    return ScenarioResult(
        schedule=schedule,
        progress=progress,
        low_score=np.asarray(low_scores, dtype=np.float64),
        high_score=np.asarray(high_scores, dtype=np.float64),
        low_relative_error=np.asarray(low_errors, dtype=np.float64),
        high_relative_error=np.asarray(high_errors, dtype=np.float64),
        frame_progress=progress[frame_indices],
        frames=tuple(frames),
    )


def _first_threshold_crossing(
    progress: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float = 0.8,
) -> float:
    """Return the first progress value at which a score reaches ``threshold``."""
    matching_indices = np.flatnonzero(score >= threshold)
    if matching_indices.size == 0:
        raise RuntimeError(f"Score never reached the validation threshold {threshold}.")
    return float(progress[matching_indices[0]])


def validate_controlled_ordering(results: tuple[ScenarioResult, ...]) -> None:
    """Fail if measured threshold ordering disagrees with the known schedules."""
    result_by_name = {result.schedule.name: result for result in results}
    low_band_first = result_by_name["low_band_first"]
    high_band_first = result_by_name["high_band_first"]
    together = result_by_name["together"]

    low_band_first_low = _first_threshold_crossing(
        low_band_first.progress,
        low_band_first.low_score,
    )
    low_band_first_high = _first_threshold_crossing(
        low_band_first.progress,
        low_band_first.high_score,
    )
    high_band_first_low = _first_threshold_crossing(
        high_band_first.progress,
        high_band_first.low_score,
    )
    high_band_first_high = _first_threshold_crossing(
        high_band_first.progress,
        high_band_first.high_score,
    )
    together_low = _first_threshold_crossing(
        together.progress,
        together.low_score,
    )
    together_high = _first_threshold_crossing(
        together.progress,
        together.high_score,
    )

    if not low_band_first_low < low_band_first_high:
        raise RuntimeError("Low-band-first control did not recover the low band first.")
    if not high_band_first_high < high_band_first_low:
        raise RuntimeError(
            "High-band-first control did not recover the high band first."
        )
    step_size = 1.0 / (len(together.progress) - 1)
    if abs(together_low - together_high) > step_size + 1e-12:
        raise RuntimeError("Together control did not recover both bands together.")


def save_recovery_curve_figure(
    results: tuple[ScenarioResult, ...],
    output_path: Path,
    *,
    radius: float,
) -> Path:
    """Save the two measured recovery curves for all controlled scenarios."""
    panels = [
        CurvePanel(
            title=result.schedule.title,
            curves=(
                LineCurve(
                    label=r"$S_{low}$: low-frequency band",
                    x=result.progress,
                    y=result.low_score,
                    color=LOW_BAND_COLOR,
                    linewidth=2.8,
                ),
                LineCurve(
                    label=r"$S_{high}$: high-frequency band",
                    x=result.progress,
                    y=result.high_score,
                    color=HIGH_BAND_COLOR,
                    linestyle="--",
                    linewidth=2.8,
                ),
            ),
        )
        for result in results
    ]
    return save_curve_panel_grid(
        panels,
        output_path,
        figure_title=(
            "Controlled Validation of Low- and High-Frequency Recovery "
            f"(Frequency Radius r = {_format_number(radius)})"
        ),
        x_label="Synthetic recovery progress",
        y_label="Band recovery score",
        y_limits=(-0.03, 1.03),
    )


def save_trajectory_grid(
    results: tuple[ScenarioResult, ...],
    output_path: Path,
) -> Path:
    """Save selected image-space frames from each controlled trajectory."""
    panel_rows = [
        [
            ImagePanel(
                title=f"progress = {progress:.2f}" if row_index == 0 else "",
                image=frame,
            )
            for progress, frame in zip(
                result.frame_progress,
                result.frames,
                strict=True,
            )
        ]
        for row_index, result in enumerate(results)
    ]
    return save_panel_grid(
        panel_rows,
        output_path,
        figure_title="Controlled Frequency-Band Recovery Trajectories",
        row_labels=tuple(result.schedule.title for result in results),
        figure_title_size=14.0,
        figure_title_y=0.985,
        column_width=3.2,
        row_height=3.0,
    )


def save_raw_scores(
    results: tuple[ScenarioResult, ...],
    output_path: Path,
    *,
    source_path: Path,
    radius: float,
    seed: int,
    noise_level: float,
) -> Path:
    """Save exact schedule weights, scores, and relative errors as CSV."""
    try:
        source_identifier = str(source_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        source_identifier = source_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            (
                "scenario",
                "source_image",
                "frequency_radius",
                "noise_seed",
                "initial_per_band_noise_level",
                "dc_excluded",
                "progress",
                "low_band_weight",
                "high_band_weight",
                "low_band_score",
                "high_band_score",
                "low_band_relative_error",
                "high_band_relative_error",
            )
        )
        for result in results:
            for index, progress in enumerate(result.progress):
                writer.writerow(
                    (
                        result.schedule.name,
                        source_identifier,
                        f"{radius:.8f}",
                        str(seed),
                        f"{noise_level:.8f}",
                        "true",
                        f"{progress:.8f}",
                        f"{result.schedule.low_weight[index]:.8f}",
                        f"{result.schedule.high_weight[index]:.8f}",
                        f"{result.low_score[index]:.8f}",
                        f"{result.high_score[index]:.8f}",
                        f"{result.low_relative_error[index]:.8f}",
                        f"{result.high_relative_error[index]:.8f}",
                    )
                )
    return output_path


def _format_number(value: float) -> str:
    """Format a numeric parameter compactly for titles and logs."""
    return str(int(value)) if value.is_integer() else f"{value:g}"


def summarize_thresholds(results: tuple[ScenarioResult, ...]) -> list[str]:
    """Return concise threshold-crossing summaries for the terminal log."""
    summaries: list[str] = []
    for result in results:
        low_crossing = _first_threshold_crossing(
            result.progress,
            result.low_score,
        )
        high_crossing = _first_threshold_crossing(
            result.progress,
            result.high_score,
        )
        summaries.append(
            f"- {result.schedule.title}: S_low@0.8={low_crossing:.2f}, "
            f"S_high@0.8={high_crossing:.2f}"
        )
    return summaries


def evaluate_controlled_trajectories(
    image: np.ndarray,
    *,
    progress: np.ndarray,
    construction_radius: float,
    evaluation_radius: float,
    noise_level: float,
    seed: int,
    frame_indices: np.ndarray,
) -> tuple[ScenarioResult, ...]:
    """Construct fixed trajectories and evaluate them at one frequency cutoff."""
    target_low, target_high = frequency_band_components(
        image,
        radius=construction_radius,
        exclude_dc=True,
    )
    evaluation_target_low, evaluation_target_high = frequency_band_components(
        image,
        radius=evaluation_radius,
        exclude_dc=True,
    )
    target_mean = image.mean(axis=(0, 1), keepdims=True)
    balanced_noise = create_band_balanced_noise(
        image.shape,
        radius=construction_radius,
        target_low=target_low,
        target_high=target_high,
        seed=seed,
    )
    return tuple(
        analyze_schedule(
            schedule,
            progress=progress,
            target_low=target_low,
            target_high=target_high,
            target_mean=target_mean,
            balanced_noise=balanced_noise,
            evaluation_target_low=evaluation_target_low,
            evaluation_target_high=evaluation_target_high,
            evaluation_radius=evaluation_radius,
            noise_level=noise_level,
            frame_indices=frame_indices,
        )
        for schedule in build_recovery_schedules(progress)
    )


def run_cutoff_sensitivity(
    image: np.ndarray,
    *,
    progress: np.ndarray,
    construction_radius: float,
    evaluation_radii: tuple[float, ...],
    seeds: tuple[int, ...],
    noise_level: float,
) -> tuple[SensitivityRecord, ...]:
    """Re-evaluate identical controlled trajectories across frequency cutoffs."""
    records: list[SensitivityRecord] = []
    target_low, target_high = frequency_band_components(
        image,
        radius=construction_radius,
        exclude_dc=True,
    )
    schedules = build_recovery_schedules(progress)
    for seed in seeds:
        balanced_noise = create_band_balanced_noise(
            image.shape,
            radius=construction_radius,
            target_low=target_low,
            target_high=target_high,
            seed=seed,
        )
        for evaluation_radius in evaluation_radii:
            evaluation_target_low, evaluation_target_high = frequency_band_components(
                image,
                radius=evaluation_radius,
                exclude_dc=True,
            )
            target_low_eval_low, target_low_eval_high = frequency_band_components(
                target_low,
                radius=evaluation_radius,
                exclude_dc=True,
            )
            target_high_eval_low, target_high_eval_high = frequency_band_components(
                target_high,
                radius=evaluation_radius,
                exclude_dc=True,
            )
            noise_eval_low, noise_eval_high = frequency_band_components(
                balanced_noise,
                radius=evaluation_radius,
                exclude_dc=True,
            )
            for schedule in schedules:
                coefficients = np.column_stack(
                    (
                        schedule.low_weight - 1.0,
                        schedule.high_weight - 1.0,
                        noise_level * (1.0 - progress),
                    )
                )
                low_score = _recovery_curve_from_components(
                    coefficients,
                    (
                        target_low_eval_low,
                        target_high_eval_low,
                        noise_eval_low,
                    ),
                    evaluation_target_low,
                )
                high_score = _recovery_curve_from_components(
                    coefficients,
                    (
                        target_low_eval_high,
                        target_high_eval_high,
                        noise_eval_high,
                    ),
                    evaluation_target_high,
                )
                records.append(
                    SensitivityRecord(
                        scenario_name=schedule.name,
                        scenario_title=schedule.title,
                        evaluation_radius=evaluation_radius,
                        seed=seed,
                        low_crossing=_first_threshold_crossing(
                            progress,
                            low_score,
                            threshold=RECOVERY_THRESHOLD,
                        ),
                        high_crossing=_first_threshold_crossing(
                            progress,
                            high_score,
                            threshold=RECOVERY_THRESHOLD,
                        ),
                    )
                )
    return tuple(records)


def _recovery_curve_from_components(
    coefficients: np.ndarray,
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: np.ndarray,
) -> np.ndarray:
    """Compute a recovery curve from a small Gram matrix of fixed components."""
    component_vectors = tuple(
        np.asarray(component, dtype=np.float64).ravel() for component in components
    )
    gram_matrix = np.asarray(
        [
            [float(np.vdot(left, right).real) for right in component_vectors]
            for left in component_vectors
        ],
        dtype=np.float64,
    )
    squared_error = np.einsum(
        "ti,ij,tj->t",
        coefficients,
        gram_matrix,
        coefficients,
    )
    target_norm = float(np.linalg.norm(np.asarray(target, dtype=np.float64).ravel()))
    if target_norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot score an empty evaluation frequency band.")
    relative_error = np.sqrt(np.maximum(squared_error, 0.0)) / target_norm
    return np.maximum(0.0, 1.0 - relative_error)


def _crossing_summary(
    records: tuple[SensitivityRecord, ...],
    *,
    scenario_name: str,
    radii: tuple[float, ...],
    attribute: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and standard deviation of one crossing statistic by radius."""
    means: list[float] = []
    standard_deviations: list[float] = []
    for radius in radii:
        values = np.asarray(
            [
                getattr(record, attribute)
                for record in records
                if record.scenario_name == scenario_name
                and record.evaluation_radius == radius
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            raise RuntimeError(
                f"No sensitivity records for {scenario_name} at radius {radius}."
            )
        means.append(float(values.mean()))
        standard_deviations.append(float(values.std(ddof=0)))
    return np.asarray(means), np.asarray(standard_deviations)


def save_cutoff_sensitivity_figure(
    records: tuple[SensitivityRecord, ...],
    output_path: Path,
    *,
    construction_radius: float,
    evaluation_radii: tuple[float, ...],
) -> Path:
    """Save threshold timing across cutoffs with uncertainty over noise seeds."""
    radius_values = np.asarray(evaluation_radii, dtype=np.float64)
    scenario_order = (
        ("low_band_first", "Low Band First"),
        ("high_band_first", "High Band First"),
        ("together", "Together"),
    )
    panels: list[CurvePanel] = []
    for scenario_name, scenario_title in scenario_order:
        low_mean, low_std = _crossing_summary(
            records,
            scenario_name=scenario_name,
            radii=evaluation_radii,
            attribute="low_crossing",
        )
        high_mean, high_std = _crossing_summary(
            records,
            scenario_name=scenario_name,
            radii=evaluation_radii,
            attribute="high_crossing",
        )
        panels.append(
            CurvePanel(
                title=scenario_title,
                curves=(
                    LineCurve(
                        label=r"$S_{low}$: low-frequency band",
                        x=radius_values,
                        y=low_mean,
                        y_lower=low_mean - low_std,
                        y_upper=low_mean + low_std,
                        color=LOW_BAND_COLOR,
                        linewidth=2.8,
                        marker="o",
                    ),
                    LineCurve(
                        label=r"$S_{high}$: high-frequency band",
                        x=radius_values,
                        y=high_mean,
                        y_lower=high_mean - high_std,
                        y_upper=high_mean + high_std,
                        color=HIGH_BAND_COLOR,
                        linestyle="--",
                        linewidth=2.8,
                        marker="s",
                    ),
                ),
            )
        )
    return save_curve_panel_grid(
        panels,
        output_path,
        figure_title=(
            "Cutoff Sensitivity of Recovery Timing "
            f"(Trajectories Constructed at r = {_format_number(construction_radius)})"
        ),
        x_label="Evaluation cutoff r (Fourier pixels)",
        y_label=f"First progress reaching score {RECOVERY_THRESHOLD:.1f}",
        y_limits=(0.0, 1.0),
    )


def save_sensitivity_records(
    records: tuple[SensitivityRecord, ...],
    output_path: Path,
    *,
    source_path: Path,
    construction_radius: float,
    noise_level: float,
) -> Path:
    """Save every cutoff and seed threshold crossing as CSV."""
    try:
        source_identifier = str(source_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        source_identifier = source_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            (
                "scenario",
                "source_image",
                "construction_radius",
                "evaluation_radius",
                "noise_seed",
                "initial_per_band_noise_level",
                "dc_excluded",
                "recovery_threshold",
                "low_band_threshold_progress",
                "high_band_threshold_progress",
            )
        )
        for record in records:
            writer.writerow(
                (
                    record.scenario_name,
                    source_identifier,
                    f"{construction_radius:.8f}",
                    f"{record.evaluation_radius:.8f}",
                    str(record.seed),
                    f"{noise_level:.8f}",
                    "true",
                    f"{RECOVERY_THRESHOLD:.8f}",
                    f"{record.low_crossing:.8f}",
                    f"{record.high_crossing:.8f}",
                )
            )
    return output_path


def main() -> int:
    """Run the controlled frequency-band metric validation."""
    args = parse_args()
    image, source_path = load_experiment_image(
        args.image_path,
        default_path=REPO_ROOT / "assets" / "default_fft_reference.png",
    )
    validate_configuration(
        image,
        radius=args.radius,
        steps=args.steps,
        noise_level=args.noise_level,
    )
    sensitivity_radii = tuple(float(radius) for radius in args.sensitivity_radii)
    sensitivity_seeds = tuple(int(seed) for seed in args.sensitivity_seeds)
    validate_sensitivity_configuration(
        image,
        radii=sensitivity_radii,
        seeds=sensitivity_seeds,
    )

    progress = np.linspace(0.0, 1.0, args.steps, dtype=np.float64)
    frame_indices = np.linspace(
        0,
        args.steps - 1,
        FRAME_COUNT,
        dtype=int,
    )
    results = evaluate_controlled_trajectories(
        image,
        progress=progress,
        construction_radius=args.radius,
        evaluation_radius=args.radius,
        noise_level=args.noise_level,
        seed=args.seed,
        frame_indices=frame_indices,
    )
    sensitivity_records = run_cutoff_sensitivity(
        image,
        progress=progress,
        construction_radius=args.radius,
        evaluation_radii=sensitivity_radii,
        seeds=sensitivity_seeds,
        noise_level=args.noise_level,
    )
    validate_controlled_ordering(results)

    output_dir = args.output_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    curve_path = output_dir / "structure_detail_recovery_curves.png"
    trajectory_path = output_dir / "controlled_recovery_trajectories.png"
    sensitivity_path = output_dir / "structure_detail_cutoff_sensitivity.png"
    csv_path = results_dir / "experiment_04_structure_detail_scores.csv"
    sensitivity_csv_path = results_dir / "experiment_04_cutoff_sensitivity.csv"
    save_recovery_curve_figure(results, curve_path, radius=args.radius)
    save_trajectory_grid(results, trajectory_path)
    save_cutoff_sensitivity_figure(
        sensitivity_records,
        sensitivity_path,
        construction_radius=args.radius,
        evaluation_radii=sensitivity_radii,
    )
    save_raw_scores(
        results,
        csv_path,
        source_path=source_path,
        radius=args.radius,
        seed=args.seed,
        noise_level=args.noise_level,
    )
    save_sensitivity_records(
        sensitivity_records,
        sensitivity_csv_path,
        source_path=source_path,
        construction_radius=args.radius,
        noise_level=args.noise_level,
    )

    print(f"Loaded image: {source_path}")
    print(f"Frequency radius: {_format_number(args.radius)} Fourier pixels")
    print(f"Trajectory steps: {args.steps}")
    print(f"Noise seed: {args.seed}")
    print(f"Initial per-band relative noise level: {args.noise_level:.4f}")
    print(f"Saved recovery curves: {curve_path}")
    print(f"Saved trajectory grid: {trajectory_path}")
    print(f"Saved cutoff sensitivity: {sensitivity_path}")
    print(f"Saved raw scores: {csv_path}")
    print(f"Saved sensitivity records: {sensitivity_csv_path}")
    print()
    print("First progress value reaching recovery score 0.8:")
    for summary in summarize_thresholds(results):
        print(summary)
    print()
    print("All three controlled ordering checks passed.")
    print("Construction and evaluation use the same frequency-band definition.")
    print("This validates metric responsiveness, not real denoiser learning dynamics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
