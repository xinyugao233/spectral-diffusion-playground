"""Controlled trajectories and statistics for frequency-band metric calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectral_diffusion_playground.metrics import (
    FrequencyBandScores,
    frequency_band_components,
    frequency_band_recovery_scores,
)
from spectral_diffusion_playground.visualization import prepare_image_for_display


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
class CutoffCurve:
    """Recovery curves for one schedule, evaluation cutoff, and noise seed."""

    schedule: RecoverySchedule
    evaluation_radius: float
    seed: int
    progress: np.ndarray
    low_score: np.ndarray
    high_score: np.ndarray


@dataclass(frozen=True, slots=True)
class SensitivityRecord:
    """Threshold crossings for one scenario, cutoff, and noise seed."""

    scenario_name: str
    scenario_title: str
    evaluation_radius: float
    seed: int
    low_crossing: float
    high_crossing: float


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


def first_threshold_crossing(
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


def validate_controlled_ordering(
    results: tuple[ScenarioResult, ...],
    *,
    threshold: float = 0.8,
) -> None:
    """Fail if measured threshold ordering disagrees with the known schedules."""
    result_by_name = {result.schedule.name: result for result in results}
    low_band_first = result_by_name["low_band_first"]
    high_band_first = result_by_name["high_band_first"]
    together = result_by_name["together"]

    low_band_first_low = first_threshold_crossing(
        low_band_first.progress,
        low_band_first.low_score,
        threshold=threshold,
    )
    low_band_first_high = first_threshold_crossing(
        low_band_first.progress,
        low_band_first.high_score,
        threshold=threshold,
    )
    high_band_first_low = first_threshold_crossing(
        high_band_first.progress,
        high_band_first.low_score,
        threshold=threshold,
    )
    high_band_first_high = first_threshold_crossing(
        high_band_first.progress,
        high_band_first.high_score,
        threshold=threshold,
    )
    together_low = first_threshold_crossing(
        together.progress,
        together.low_score,
        threshold=threshold,
    )
    together_high = first_threshold_crossing(
        together.progress,
        together.high_score,
        threshold=threshold,
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


def evaluate_cutoff_curves(
    image: np.ndarray,
    *,
    progress: np.ndarray,
    construction_radius: float,
    evaluation_radii: tuple[float, ...],
    seeds: tuple[int, ...],
    noise_level: float,
) -> tuple[CutoffCurve, ...]:
    """Evaluate identical controlled trajectories across frequency cutoffs."""
    curves: list[CutoffCurve] = []
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
                curves.append(
                    CutoffCurve(
                        schedule=schedule,
                        evaluation_radius=evaluation_radius,
                        seed=seed,
                        progress=progress,
                        low_score=_recovery_curve_from_components(
                            coefficients,
                            (
                                target_low_eval_low,
                                target_high_eval_low,
                                noise_eval_low,
                            ),
                            evaluation_target_low,
                        ),
                        high_score=_recovery_curve_from_components(
                            coefficients,
                            (
                                target_low_eval_high,
                                target_high_eval_high,
                                noise_eval_high,
                            ),
                            evaluation_target_high,
                        ),
                    )
                )
    return tuple(curves)


def sensitivity_records_from_curves(
    curves: tuple[CutoffCurve, ...],
    *,
    threshold: float = 0.8,
) -> tuple[SensitivityRecord, ...]:
    """Convert cutoff curves into threshold-crossing records."""
    return tuple(
        SensitivityRecord(
            scenario_name=curve.schedule.name,
            scenario_title=curve.schedule.title,
            evaluation_radius=curve.evaluation_radius,
            seed=curve.seed,
            low_crossing=first_threshold_crossing(
                curve.progress,
                curve.low_score,
                threshold=threshold,
            ),
            high_crossing=first_threshold_crossing(
                curve.progress,
                curve.high_score,
                threshold=threshold,
            ),
        )
        for curve in curves
    )


def run_cutoff_sensitivity(
    image: np.ndarray,
    *,
    progress: np.ndarray,
    construction_radius: float,
    evaluation_radii: tuple[float, ...],
    seeds: tuple[int, ...],
    noise_level: float,
    threshold: float = 0.8,
) -> tuple[SensitivityRecord, ...]:
    """Return threshold crossings for fixed trajectories across cutoffs."""
    curves = evaluate_cutoff_curves(
        image,
        progress=progress,
        construction_radius=construction_radius,
        evaluation_radii=evaluation_radii,
        seeds=seeds,
        noise_level=noise_level,
    )
    return sensitivity_records_from_curves(curves, threshold=threshold)


def ordering_matches_control(
    trajectory: str,
    *,
    low_crossing: float,
    high_crossing: float,
    progress_step: float,
) -> bool:
    """Return whether threshold timing matches a known controlled schedule."""
    if trajectory == "low_band_first":
        return low_crossing < high_crossing
    if trajectory == "high_band_first":
        return high_crossing < low_crossing
    if trajectory == "together":
        return abs(low_crossing - high_crossing) <= progress_step + 1e-12
    raise ValueError(f"Unknown controlled trajectory: {trajectory}")


def bootstrap_mean_interval(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap a confidence interval for a mean by resampling image rows."""
    value_array = np.asarray(values, dtype=np.float64)
    index_array = np.asarray(bootstrap_indices, dtype=np.int64)
    if value_array.ndim == 0:
        raise ValueError("values must have an image axis.")
    if index_array.ndim != 2 or index_array.shape[1] != value_array.shape[0]:
        raise ValueError("bootstrap_indices must have shape (resamples, images).")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    bootstrap_means = value_array[index_array].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail), axis=0)
    return np.asarray(lower), np.asarray(upper)
