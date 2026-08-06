"""Frozen condition and analysis utilities for Experiment 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from spectral_diffusion_playground.e006_transition_swaps import SIGMA_GRID

EXPERIMENT_ID: Final[str] = "E010"
RUN_ID: Final[str] = "e010_directional_memorization_transfer"
SAMPLE_SEEDS: Final[tuple[int, ...]] = tuple(range(40000, 40256))
BOOTSTRAP_RESAMPLES: Final[int] = 100_000
BOOTSTRAP_SEED: Final[int] = 0
EXPECTED_RECORDS: Final[int] = 3_584


@dataclass(frozen=True)
class DirectionalCondition:
    """One registered no-swap or directional whole-denoiser condition."""

    condition_id: str
    direction: str
    band: str
    role: str
    recipient: str
    donor: str | None
    swap_indices: tuple[int, ...]

    def model_for_step(self, step_index: int) -> str:
        """Return donor inside the frozen window and recipient otherwise."""
        if step_index in self.swap_indices:
            if self.donor is None:
                raise RuntimeError("A swap condition requires a donor")
            return self.donor
        return self.recipient


def frozen_conditions() -> tuple[DirectionalCondition, ...]:
    """Return the 14 preregistered E010 conditions in manifest order."""
    specs = (
        ("A0", "suppression", "baseline", "baseline", "edm_1k_012000", None, ()),
        ("A1", "suppression", "low", "before", "edm_1k_012000", "edm_50k_040000", (7,)),
        ("A2", "suppression", "low", "target", "edm_1k_012000", "edm_50k_040000", (8,)),
        ("A3", "suppression", "low", "after", "edm_1k_012000", "edm_50k_040000", (9,)),
        (
            "A4",
            "suppression",
            "high",
            "before",
            "edm_1k_012000",
            "edm_50k_040000",
            (7, 8),
        ),
        (
            "A5",
            "suppression",
            "high",
            "target",
            "edm_1k_012000",
            "edm_50k_040000",
            (9, 10),
        ),
        (
            "A6",
            "suppression",
            "high",
            "after",
            "edm_1k_012000",
            "edm_50k_040000",
            (11, 12),
        ),
        ("B0", "induction", "baseline", "baseline", "edm_50k_040000", None, ()),
        ("B1", "induction", "low", "before", "edm_50k_040000", "edm_1k_012000", (7,)),
        ("B2", "induction", "low", "target", "edm_50k_040000", "edm_1k_012000", (8,)),
        ("B3", "induction", "low", "after", "edm_50k_040000", "edm_1k_012000", (9,)),
        (
            "B4",
            "induction",
            "high",
            "before",
            "edm_50k_040000",
            "edm_1k_012000",
            (7, 8),
        ),
        (
            "B5",
            "induction",
            "high",
            "target",
            "edm_50k_040000",
            "edm_1k_012000",
            (9, 10),
        ),
        (
            "B6",
            "induction",
            "high",
            "after",
            "edm_50k_040000",
            "edm_1k_012000",
            (11, 12),
        ),
    )
    return tuple(DirectionalCondition(*spec) for spec in specs)


def validate_condition_registry(conditions: Sequence[DirectionalCondition]) -> None:
    """Fail unless conditions exactly match the frozen registry."""
    expected = frozen_conditions()
    if tuple(conditions) != expected:
        raise ValueError("Condition registry differs from the frozen E010 registry")
    identifiers = [condition.condition_id for condition in conditions]
    if len(set(identifiers)) != 14:
        raise ValueError("E010 requires 14 unique condition identifiers")


def nearest_two_cpu(
    sample: np.ndarray, references: np.ndarray
) -> tuple[float, float, int, int]:
    """Return deterministic nearest-two Euclidean distances on CPU float64."""
    sample64 = np.asarray(sample, dtype=np.float64).reshape(-1)
    refs64 = np.asarray(references, dtype=np.float64).reshape(references.shape[0], -1)
    if refs64.shape[0] < 2 or refs64.shape[1] != sample64.size:
        raise ValueError("Reference bank must contain two compatible samples")
    diff = refs64 - sample64[None, :]
    squared = np.sum(diff * diff, axis=1, dtype=np.float64)
    positions = np.arange(squared.size, dtype=np.int64)
    order = np.lexsort((positions, squared))
    first, second = int(order[0]), int(order[1])
    return (
        float(np.sqrt(squared[first])),
        float(np.sqrt(squared[second])),
        first,
        second,
    )


def transition_category(baseline: bool, swapped: bool) -> str:
    """Classify one paired memorization transition."""
    labels = {
        (True, False): "memorized_to_non_memorized",
        (False, True): "non_memorized_to_memorized",
        (True, True): "memorized_to_memorized",
        (False, False): "non_memorized_to_non_memorized",
    }
    return labels[(bool(baseline), bool(swapped))]


def directional_seed_effect(
    baseline: np.ndarray, swapped: np.ndarray, direction: str
) -> NDArray[np.float64]:
    """Return frozen per-seed suppression or induction effects."""
    base = np.asarray(baseline, dtype=np.float64)
    swap = np.asarray(swapped, dtype=np.float64)
    if base.shape != swap.shape or base.ndim != 1:
        raise ValueError("Baseline and swap must be aligned one-dimensional arrays")
    if direction == "suppression":
        return np.asarray(base - swap, dtype=np.float64)
    if direction == "induction":
        return np.asarray(swap - base, dtype=np.float64)
    raise ValueError(f"Unknown direction: {direction}")


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return deterministic percentile interval for a paired-seed mean."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap input must be a nonempty finite vector")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    offset = 0
    while offset < resamples:
        count = min(1000, resamples - offset)
        indices = generator.integers(0, data.size, size=(count, data.size))
        means[offset : offset + count] = data[indices].mean(axis=1)
        offset += count
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def target_control_summary(
    baseline: np.ndarray,
    before: np.ndarray,
    target: np.ndarray,
    after: np.ndarray,
    *,
    direction: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Apply the frozen target-versus-neighbor influence criterion."""
    before_effects = directional_seed_effect(baseline, before, direction)
    target_effects = directional_seed_effect(baseline, target, direction)
    after_effects = directional_seed_effect(baseline, after, direction)
    contrast = target_effects - (before_effects + after_effects) / 2.0
    ci_low, ci_high = bootstrap_mean_interval(contrast, seed=seed, resamples=resamples)
    values = {
        "before_effect": float(before_effects.mean()),
        "target_effect": float(target_effects.mean()),
        "after_effect": float(after_effects.mean()),
        "contrast": float(contrast.mean()),
        "contrast_ci95_low": ci_low,
        "contrast_ci95_high": ci_high,
    }
    values["criterion_pass"] = bool(
        values["target_effect"] > 0.0
        and values["target_effect"] > values["before_effect"]
        and values["target_effect"] > values["after_effect"]
        and values["contrast"] > 0.0
        and ci_low > 0.0
    )
    return values


def formal_outcomes(results: Mapping[tuple[str, str], bool]) -> list[str]:
    """Return transparent labels for the four frozen primary tests."""
    expected = {
        ("suppression", "low"),
        ("suppression", "high"),
        ("induction", "low"),
        ("induction", "high"),
    }
    if set(results) != expected:
        raise ValueError("Formal outcome requires all four direction-band tests")
    labels = []
    for direction, band in sorted(expected):
        if results[(direction, band)]:
            labels.append(f"{band.upper()}_DERIVED_{direction.upper()}_SUPPORTED")
    if not labels:
        return ["NO_DIRECTIONAL_TARGET_OUTPERFORMS_CONTROLS"]
    if len(labels) not in (1, 4):
        labels.append("MIXED_DIRECTIONAL_EVIDENCE")
    return labels


def sigma_grid() -> tuple[float, ...]:
    """Expose the frozen 18-call schedule without allowing mutation."""
    return tuple(SIGMA_GRID)
