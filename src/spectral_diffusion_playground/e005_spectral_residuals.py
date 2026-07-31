"""Numerical utilities for Experiment 5 spectral residual curves."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import numpy as np
from numpy.typing import NDArray

from spectral_diffusion_playground.filters import decompose_frequency_bands

EXPERIMENT_ID: Final[str] = "experiment_05"
RUN_ID: Final[str] = "e005_spectral_residual_curves"
MASTER_NOISE_SEED: Final[int] = 20260726
BOOTSTRAP_SEED: Final[int] = 20260727
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
REDUCTION_ELEMENTS: Final[int] = 3 * 32 * 32
RECONSTRUCTION_ATOL: Final[float] = 1e-10
RELATIVE_ENERGY_ATOL: Final[float] = 1e-12
CUTOFFS: Final[tuple[int, ...]] = (3, 4, 5, 6)
PRIMARY_CUTOFFS: Final[tuple[int, ...]] = (3, 4, 5)
REFERENCE_CUTOFF: Final[int] = 4
SIGMA_GRID: Final[tuple[float, ...]] = (
    80.0,
    57.58598472124816,
    40.785573796507961,
    28.374584604156844,
    19.352452980325229,
    12.91008238075732,
    8.4009353090998165,
    5.3151945217963821,
    3.2568215197655368,
    1.9233398370400518,
    1.088170636545279,
    0.58534812319454221,
    0.29644228447915727,
    0.13951646873101678,
    0.05994731123547159,
    0.022934518372333384,
    0.0075280199627840785,
    0.0020000000000000031,
)
BANDS: Final[tuple[str, ...]] = (
    "full",
    "low_frequency_residual",
    "high_frequency_residual",
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionEnergy:
    """Squared residual energies and identity diagnostics for one cutoff."""

    full: float
    low: float
    high: float
    full_mse: float
    low_mse: float
    high_mse: float
    reconstruction_max_abs_error: float
    additivity_absolute_error: float
    additivity_relative_error: float
    orthogonality_relative_error: float


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_sequence_material(
    *,
    split_code: int,
    dataset_index: int,
    noise_repeat: int,
    sigma_index: int,
) -> list[int]:
    """Return the frozen seed material for one noise tensor."""
    return [
        MASTER_NOISE_SEED,
        split_code,
        int(dataset_index),
        int(noise_repeat),
        int(sigma_index),
    ]


def deterministic_noise(
    shape: tuple[int, ...],
    *,
    split_code: int,
    dataset_index: int,
    noise_repeat: int,
    sigma_index: int,
) -> tuple[FloatArray, int]:
    """Generate one deterministic float64 Gaussian tensor and derived seed."""
    material = seed_sequence_material(
        split_code=split_code,
        dataset_index=dataset_index,
        noise_repeat=noise_repeat,
        sigma_index=sigma_index,
    )
    sequence = np.random.SeedSequence(material)
    derived_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    generator = np.random.Generator(np.random.PCG64DXSM(sequence))
    return (
        np.asarray(generator.standard_normal(shape, dtype=np.float64)),
        derived_seed,
    )


def compute_projection_energy(
    residual_hwc: np.ndarray, cutoff: int
) -> ProjectionEnergy:
    """Project one HWC residual and return additive spectral energies."""
    residual = np.asarray(residual_hwc, dtype=np.float64)
    low, high = decompose_frequency_bands(residual, radius=float(cutoff))
    full_energy = float(np.sum(residual**2, dtype=np.float64))
    low_energy = float(np.sum(low**2, dtype=np.float64))
    high_energy = float(np.sum(high**2, dtype=np.float64))
    reconstruction_error = float(np.max(np.abs(residual - (low + high))))
    additivity_absolute = float(abs(full_energy - low_energy - high_energy))
    denominator = max(full_energy, float(np.finfo(np.float64).tiny))
    additivity_relative = float(additivity_absolute / denominator)
    orthogonality_relative = float(
        abs(np.sum(low * high, dtype=np.float64)) / denominator
    )
    return ProjectionEnergy(
        full=full_energy,
        low=low_energy,
        high=high_energy,
        full_mse=full_energy / REDUCTION_ELEMENTS,
        low_mse=low_energy / REDUCTION_ELEMENTS,
        high_mse=high_energy / REDUCTION_ELEMENTS,
        reconstruction_max_abs_error=reconstruction_error,
        additivity_absolute_error=additivity_absolute,
        additivity_relative_error=additivity_relative,
        orthogonality_relative_error=orthogonality_relative,
    )


def validate_projection_energy(energy: ProjectionEnergy) -> None:
    """Raise if one projected residual violates the frozen identities."""
    if energy.reconstruction_max_abs_error > RECONSTRUCTION_ATOL:
        raise ValueError(
            "frequency reconstruction identity failed: "
            f"{energy.reconstruction_max_abs_error}"
        )
    if energy.additivity_relative_error > RELATIVE_ENERGY_ATOL:
        raise ValueError(
            "energy additivity identity failed: " f"{energy.additivity_relative_error}"
        )
    if energy.orthogonality_relative_error > RELATIVE_ENERGY_ATOL:
        raise ValueError(
            "low/high orthogonality identity failed: "
            f"{energy.orthogonality_relative_error}"
        )


def raw_csv_header() -> list[str]:
    """Return the frozen per-sample residual CSV schema."""
    return [
        "experiment_id",
        "run_id",
        "model",
        "checkpoint_sha256",
        "split",
        "image_index",
        "image_manifest_position",
        "noise_repeat",
        "noise_seed",
        "sigma_index",
        "sigma",
        "sigma_grid",
        "cutoff",
        "cutoff_normalized",
        "reduction_elements",
        "full_squared_error",
        "low_squared_error",
        "high_squared_error",
        "full_mean_squared_error",
        "low_mean_squared_error",
        "high_mean_squared_error",
        "reconstruction_max_abs_error",
        "energy_additivity_absolute_error",
        "energy_additivity_relative_error",
        "orthogonality_relative_error",
        "status",
        "error",
    ]


def write_csv(path: Path, rows: Iterable[dict[str, object]], header: list[str]) -> None:
    """Write dictionaries to a CSV file with a stable header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bootstrap_mean_ci(
    image_means: FloatArray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    chunk_size: int = 500,
) -> tuple[float, float, float]:
    """Return mean and 95% image-cluster bootstrap interval for one vector."""
    values = np.asarray(image_means, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D image means, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot bootstrap nonfinite values")

    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    n_images = values.shape[0]
    offset = 0
    while offset < resamples:
        size = min(chunk_size, resamples - offset)
        indices = generator.integers(0, n_images, size=(size, n_images))
        means[offset : offset + size] = values[indices].mean(axis=1)
        offset += size
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def aggregate_curves(
    energies: FloatArray,
    *,
    model: str,
    split: str,
    checkpoint_sha256: str,
    cutoff: int,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[dict[str, object]]:
    """Aggregate one model/split/cutoff energy tensor into curve rows.

    ``energies`` has shape ``(n_images, n_repeats, n_sigmas, 3)`` with band
    order ``full``, ``low_frequency_residual``, ``high_frequency_residual``.
    """
    if energies.ndim != 4 or energies.shape[-1] != len(BANDS):
        raise ValueError(f"Unexpected energy tensor shape: {energies.shape}")
    n_images, n_repeats, _, _ = energies.shape
    image_means = energies.mean(axis=1)
    rows: list[dict[str, object]] = []
    for sigma_index, sigma in enumerate(SIGMA_GRID):
        for band_index, band in enumerate(BANDS):
            seed = (
                bootstrap_seed
                + stable_int_hash(model)
                + stable_int_hash(split)
                + cutoff * 100
                + sigma_index * 10
                + band_index
            )
            mean, ci_low, ci_high = bootstrap_mean_ci(
                image_means[:, sigma_index, band_index],
                seed=seed,
                resamples=bootstrap_resamples,
            )
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": RUN_ID,
                    "model": model,
                    "split": split,
                    "sigma_index": sigma_index,
                    "sigma": f"{sigma:.17g}",
                    "sigma_grid": "edm_18_rho7",
                    "cutoff": cutoff,
                    "band": band,
                    "n_images": n_images,
                    "n_repeats": n_repeats,
                    "mean_summed_squared_error": f"{mean:.17g}",
                    "ci95_low_summed_squared_error": f"{ci_low:.17g}",
                    "ci95_high_summed_squared_error": f"{ci_high:.17g}",
                    "mean_per_element_mse": f"{mean / REDUCTION_ELEMENTS:.17g}",
                    "ci95_low_per_element_mse": f"{ci_low / REDUCTION_ELEMENTS:.17g}",
                    "ci95_high_per_element_mse": f"{ci_high / REDUCTION_ELEMENTS:.17g}",
                    "aggregation_status": "ok",
                }
            )
    return rows


def stable_int_hash(value: str) -> int:
    """Return a deterministic small integer hash for bootstrap seed offsets."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def aggregated_csv_header() -> list[str]:
    """Return the frozen aggregated-curve CSV schema."""
    return [
        "experiment_id",
        "run_id",
        "model",
        "split",
        "sigma_index",
        "sigma",
        "sigma_grid",
        "cutoff",
        "band",
        "n_images",
        "n_repeats",
        "mean_summed_squared_error",
        "ci95_low_summed_squared_error",
        "ci95_high_summed_squared_error",
        "mean_per_element_mse",
        "ci95_low_per_element_mse",
        "ci95_high_per_element_mse",
        "aggregation_status",
    ]


def extract_transition_window(curve: FloatArray) -> dict[str, object]:
    """Extract one transition window using the frozen two-point crossing rule."""
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (len(SIGMA_GRID),):
        raise ValueError(f"Expected one 18-point curve, got {values.shape}")
    if not np.all(np.isfinite(values)):
        return {"status": "no_clear_transition", "reason": "nonfinite_curve"}

    high_endpoint = float(np.median(values[[0, 1, 2]]))
    low_endpoint = float(np.median(values[[15, 16, 17]]))
    denominator = high_endpoint - low_endpoint
    if denominator <= 0.0 or not np.isfinite(denominator):
        return {
            "status": "no_clear_transition",
            "reason": "invalid_endpoint_denominator",
            "high_endpoint": high_endpoint,
            "low_endpoint": low_endpoint,
        }

    recovery = (high_endpoint - values) / denominator
    entry = first_two_point_crossing(recovery, threshold=0.20, start=0)
    if entry is None:
        return transition_failure(
            "missing_entry_crossing",
            recovery,
            high_endpoint,
            low_endpoint,
        )
    exit_index = first_two_point_crossing(recovery, threshold=0.80, start=entry)
    if exit_index is None:
        return transition_failure(
            "missing_exit_crossing",
            recovery,
            high_endpoint,
            low_endpoint,
            entry=entry,
        )
    if exit_index < entry:
        return transition_failure(
            "exit_precedes_entry",
            recovery,
            high_endpoint,
            low_endpoint,
            entry=entry,
            exit_index=exit_index,
        )
    if exit_index == entry:
        return transition_failure(
            "minimum_width_failed",
            recovery,
            high_endpoint,
            low_endpoint,
            entry=entry,
            exit_index=exit_index,
        )
    if has_two_point_recrossing(recovery, threshold=0.20, start=entry):
        return transition_failure(
            "entry_threshold_recrossing",
            recovery,
            high_endpoint,
            low_endpoint,
            entry=entry,
            exit_index=exit_index,
        )
    if has_two_point_recrossing(recovery, threshold=0.80, start=exit_index):
        return transition_failure(
            "exit_threshold_recrossing",
            recovery,
            high_endpoint,
            low_endpoint,
            entry=entry,
            exit_index=exit_index,
        )
    return {
        "status": "ok",
        "reason": None,
        "high_endpoint": high_endpoint,
        "low_endpoint": low_endpoint,
        "normalized_recovery": recovery.tolist(),
        "entry_index": int(entry),
        "exit_index": int(exit_index),
        "entry_sigma": SIGMA_GRID[entry],
        "exit_sigma": SIGMA_GRID[exit_index],
        "window_indices": list(range(entry, exit_index + 1)),
        "window_sigmas": [SIGMA_GRID[i] for i in range(entry, exit_index + 1)],
    }


def first_two_point_crossing(
    values: FloatArray,
    *,
    threshold: float,
    start: int,
) -> int | None:
    """Return the first index with two consecutive values above threshold."""
    for index in range(start, len(values) - 1):
        if values[index] >= threshold and values[index + 1] >= threshold:
            return index
    return None


def has_two_point_recrossing(
    values: FloatArray,
    *,
    threshold: float,
    start: int,
) -> bool:
    """Return whether later values fall below threshold for two consecutive points."""
    for index in range(start + 1, len(values) - 1):
        if values[index] < threshold and values[index + 1] < threshold:
            return True
    return False


def transition_failure(
    reason: str,
    recovery: FloatArray,
    high_endpoint: float,
    low_endpoint: float,
    *,
    entry: int | None = None,
    exit_index: int | None = None,
) -> dict[str, object]:
    """Build a no-clear-transition record with diagnostic values."""
    return {
        "status": "no_clear_transition",
        "reason": reason,
        "high_endpoint": high_endpoint,
        "low_endpoint": low_endpoint,
        "normalized_recovery": recovery.tolist(),
        "entry_index": entry,
        "exit_index": exit_index,
    }


def attach_adjacent_cutoff_stability(
    transitions: dict[str, dict[str, dict[str, object]]],
) -> None:
    """Annotate r=4 transitions using the frozen adjacent-cutoff rule."""
    for band in (
        "low_frequency_residual",
        "high_frequency_residual",
    ):
        reference = transitions[band][str(REFERENCE_CUTOFF)]
        if reference["status"] != "ok":
            reference["adjacent_cutoff_stable"] = False
            reference["cutoff_sensitive"] = True
            continue
        stable = True
        for cutoff in (3, 5):
            candidate = transitions[band][str(cutoff)]
            if candidate["status"] != "ok":
                stable = False
                continue
            stable = stable and (
                abs(int(reference["entry_index"]) - int(candidate["entry_index"])) <= 1
            )
            stable = stable and (
                abs(int(reference["exit_index"]) - int(candidate["exit_index"])) <= 1
            )
        reference["adjacent_cutoff_stable"] = stable
        reference["cutoff_sensitive"] = not stable


def dump_json(path: Path, payload: dict[str, object]) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
