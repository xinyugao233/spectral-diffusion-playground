"""Frequency-restricted posterior and Gaussian-shell geometry estimators."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .filters import create_frequency_mask
from .paper_geometry import (
    gaussian_shell_radii,
    normalized_weights_from_logits,
    posterior_weights,
    squared_distances,
)
from .paper_geometry_evaluation import (
    _coverage_draw_values,
    _cross_term,
    _posterior_draw_values,
    hierarchical_bootstrap_interval,
    monte_carlo_standard_error,
)
from .region_definitions import contiguous_components, high_high_indices

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Band = Literal["low", "high"]

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
IMAGE_DIMENSION = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH


def _validate_band(band: str) -> Band:
    """Return a validated frequency-band name."""
    if band not in {"low", "high"}:
        raise ValueError("band must be 'low' or 'high'")
    return band


def _band_mask(radius: float, band: Band) -> FloatArray:
    """Return one centered binary mask from the complementary mask pair."""
    low = create_frequency_mask(IMAGE_HEIGHT, IMAGE_WIDTH, radius)
    return low if band == "low" else np.asarray(1.0 - low, dtype=np.float64)


def mask_is_conjugate_symmetric(mask: np.ndarray) -> bool:
    """Check conjugate symmetry in the unshifted discrete Fourier grid."""
    values = np.asarray(mask)
    if values.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("mask must have shape (32, 32)")
    unshifted = np.fft.ifftshift(values)
    y_conjugate = (-np.arange(IMAGE_HEIGHT)) % IMAGE_HEIGHT
    x_conjugate = (-np.arange(IMAGE_WIDTH)) % IMAGE_WIDTH
    return bool(np.array_equal(unshifted, unshifted[np.ix_(y_conjugate, x_conjugate)]))


def band_projector_rank(radius: float, band: Band, *, channels: int = 3) -> int:
    """Return the exact real rank of one conjugate-symmetric band projector."""
    selected = _band_mask(radius, _validate_band(band))
    if not mask_is_conjugate_symmetric(selected):
        raise ValueError("Frequency mask is not conjugate symmetric")
    if channels < 1:
        raise ValueError("channels must be positive")
    return channels * int(np.count_nonzero(selected))


def project_to_band(
    vectors: np.ndarray,
    radius: float,
    band: Band,
    *,
    return_imaginary_residual: bool = False,
) -> FloatArray | tuple[FloatArray, float]:
    """Project flattened CIFAR-10 vectors into one real Fourier subspace.

    Inputs and outputs use the canonical flattened channel-first CIFAR layout.
    The orthonormal FFT is applied channelwise, and the inverse transform is
    retained in real spatial coordinates for all downstream distances.
    """
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != IMAGE_DIMENSION:
        raise ValueError("vectors must have shape (examples, 3072)")
    selected = _band_mask(radius, _validate_band(band))
    channel_first = values.reshape(-1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    spectrum = np.fft.fftshift(
        np.fft.fft2(channel_first, axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )
    filtered = spectrum * selected[None, None, :, :]
    reconstructed = np.fft.ifft2(
        np.fft.ifftshift(filtered, axes=(-2, -1)),
        axes=(-2, -1),
        norm="ortho",
    )
    maximum_imaginary = float(np.max(np.abs(reconstructed.imag), initial=0.0))
    projected = np.asarray(reconstructed.real.reshape(values.shape), dtype=np.float64)
    if return_imaginary_residual:
        return projected, maximum_imaginary
    return projected


def band_posterior_weights(
    projected_noisy_queries: FloatArray,
    projected_references: FloatArray,
    sigma: float,
) -> FloatArray:
    """Compute normalized posterior weights in a projected real subspace."""
    return posterior_weights(projected_noisy_queries, projected_references, sigma)


def band_maximum_posterior_weight(
    projected_noisy_queries: FloatArray,
    projected_references: FloatArray,
    sigma: float,
) -> float:
    """Average the maximum empirical posterior weight in one band."""
    weights = band_posterior_weights(
        projected_noisy_queries,
        projected_references,
        sigma,
    )
    return float(np.max(weights, axis=1).mean())


def band_gaussian_shell_membership(
    projected_noisy_queries: FloatArray,
    projected_references: FloatArray,
    sigma: float,
    *,
    band_dimension: int,
    c_value: float = 5.0,
    reference_batch_size: int = 256,
) -> BoolArray:
    """Test exact shell-union membership using the projector's real rank."""
    queries = np.asarray(projected_noisy_queries, dtype=np.float64)
    references = np.asarray(projected_references, dtype=np.float64)
    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("Inputs must be two-dimensional matrices")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("Inputs must share a storage dimension")
    if band_dimension < 2 or band_dimension > queries.shape[1]:
        raise ValueError("band_dimension must be a valid real projector rank")
    if reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    inner, outer = gaussian_shell_radii(band_dimension, c_value)
    lower_sq = (sigma * inner) ** 2
    upper_sq = (sigma * outer) ** 2
    covered = np.zeros(len(queries), dtype=bool)
    for start in range(0, len(references), reference_batch_size):
        distance_sq = squared_distances(
            queries,
            references[start : start + reference_batch_size],
        )
        covered |= np.any(
            (distance_sq >= lower_sq) & (distance_sq <= upper_sq),
            axis=1,
        )
    return covered


def band_gaussian_shell_coverage(
    projected_noisy_queries: FloatArray,
    projected_references: FloatArray,
    sigma: float,
    *,
    band_dimension: int,
    c_value: float = 5.0,
    reference_batch_size: int = 256,
) -> float:
    """Average exact band-space Gaussian-shell union membership."""
    membership = band_gaussian_shell_membership(
        projected_noisy_queries,
        projected_references,
        sigma,
        band_dimension=band_dimension,
        c_value=c_value,
        reference_batch_size=reference_batch_size,
    )
    return float(membership.mean())


def _noise_draws(
    shape: tuple[int, int], draws: int, rng: np.random.Generator
) -> list[FloatArray]:
    """Generate E004A-compatible float32 draws represented as float64 arrays."""
    return [
        rng.standard_normal(shape, dtype=np.float32).astype(np.float64)
        for _ in range(draws)
    ]


def _draw_digest(draws: Sequence[FloatArray]) -> str:
    """Hash underlying full-dimensional Gaussian draws in generation order."""
    digest = hashlib.sha256()
    for draw in draws:
        digest.update(np.asarray(draw, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _evaluate_projected_band(
    training: FloatArray,
    test: FloatArray,
    posterior_noise: Sequence[FloatArray],
    coverage_noise: Sequence[FloatArray],
    *,
    radius: int,
    band: Band,
    sigmas: Sequence[float],
    shell_c: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    query_batch_size: int,
    reference_batch_size: int,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Evaluate one band while passing its explicit real rank to shell code."""
    projected_train, train_imaginary = project_to_band(
        training,
        radius,
        band,
        return_imaginary_residual=True,
    )
    projected_test, test_imaginary = project_to_band(
        test,
        radius,
        band,
        return_imaginary_residual=True,
    )
    dimension = band_projector_rank(radius, band)
    train_distances = squared_distances(projected_train, projected_train)
    test_distances = squared_distances(projected_test, projected_train)

    posterior_cross: list[FloatArray] = []
    posterior_energy: list[FloatArray] = []
    coverage_cross: list[FloatArray] = []
    coverage_energy: list[FloatArray] = []
    maximum_imaginary = max(train_imaginary, test_imaginary)
    for noise in posterior_noise:
        projected_noise, imaginary = project_to_band(
            noise,
            radius,
            band,
            return_imaginary_residual=True,
        )
        maximum_imaginary = max(maximum_imaginary, imaginary)
        posterior_cross.append(
            _cross_term(projected_noise, projected_train, projected_train)
        )
        posterior_energy.append(np.square(projected_noise).sum(axis=1))
    for noise in coverage_noise:
        projected_noise, imaginary = project_to_band(
            noise,
            radius,
            band,
            return_imaginary_residual=True,
        )
        maximum_imaginary = max(maximum_imaginary, imaginary)
        coverage_cross.append(
            _cross_term(projected_noise, projected_test, projected_train)
        )
        coverage_energy.append(np.square(projected_noise).sum(axis=1))

    posterior_raw = np.empty(
        (len(sigmas), len(posterior_noise), len(training)), dtype=np.float64
    )
    coverage_raw = np.empty(
        (len(sigmas), len(coverage_noise), len(test)), dtype=np.float64
    )
    normalization_error = 0.0
    for sigma_index, sigma in enumerate(sigmas):
        for draw_index, (cross, energy) in enumerate(
            zip(posterior_cross, posterior_energy)
        ):
            maxima, error = _posterior_draw_values(
                train_distances,
                cross,
                energy,
                float(sigma),
                query_batch_size,
            )
            posterior_raw[sigma_index, draw_index] = maxima
            normalization_error = max(normalization_error, error)
        for draw_index, (cross, energy) in enumerate(
            zip(coverage_cross, coverage_energy)
        ):
            coverage_raw[sigma_index, draw_index] = _coverage_draw_values(
                test_distances,
                cross,
                energy,
                float(sigma),
                dimension,
                shell_c,
                query_batch_size,
                reference_batch_size,
            )

    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    posterior_low, posterior_high = hierarchical_bootstrap_interval(
        posterior_raw,
        bootstrap_replicates,
        bootstrap_rng,
    )
    coverage_low, coverage_high = hierarchical_bootstrap_interval(
        coverage_raw,
        bootstrap_replicates,
        bootstrap_rng,
    )
    curves = {
        "posterior_weight_estimate": posterior_raw.mean(axis=(1, 2)),
        "posterior_weight_ci95_low": posterior_low,
        "posterior_weight_ci95_high": posterior_high,
        "posterior_weight_monte_carlo_se": monte_carlo_standard_error(posterior_raw),
        "coverage_estimate": coverage_raw.mean(axis=(1, 2)),
        "coverage_ci95_low": coverage_low,
        "coverage_ci95_high": coverage_high,
        "coverage_monte_carlo_se": monte_carlo_standard_error(coverage_raw),
    }
    validation = {
        "band_dimension": dimension,
        "maximum_inverse_fft_imaginary_residual": maximum_imaginary,
        "posterior_normalization_error_max": normalization_error,
        "posterior_values_in_unit_interval": bool(
            np.all((posterior_raw >= 0.0) & (posterior_raw <= 1.0))
        ),
        "coverage_values_in_unit_interval": bool(
            np.all((coverage_raw >= 0.0) & (coverage_raw <= 1.0))
        ),
        "nonfinite_values": int(
            sum(np.count_nonzero(~np.isfinite(value)) for value in curves.values())
        ),
    }
    validation["status"] = (
        "pass"
        if validation["maximum_inverse_fft_imaginary_residual"] <= 1e-12
        and validation["posterior_normalization_error_max"] <= 1e-10
        and validation["posterior_values_in_unit_interval"]
        and validation["coverage_values_in_unit_interval"]
        and validation["nonfinite_values"] == 0
        else "fail"
    )
    return curves, validation


def _projector_validation(sample: FloatArray, radius: int) -> dict[str, Any]:
    """Validate complementarity, orthogonality, reconstruction, and Parseval."""
    low_mask = _band_mask(radius, "low")
    high_mask = _band_mask(radius, "high")
    low, low_imaginary = project_to_band(
        sample,
        radius,
        "low",
        return_imaginary_residual=True,
    )
    high, high_imaginary = project_to_band(
        sample,
        radius,
        "high",
        return_imaginary_residual=True,
    )
    low_twice = project_to_band(low, radius, "low")
    high_twice = project_to_band(high, radius, "high")
    cross_inner = np.sum(low * high, axis=1)
    original_energy = np.square(sample).sum(axis=1)
    decomposed_energy = np.square(low).sum(axis=1) + np.square(high).sum(axis=1)
    return {
        "mask_binary": bool(
            np.all(np.isin(low_mask, (0.0, 1.0)))
            and np.all(np.isin(high_mask, (0.0, 1.0)))
        ),
        "mask_exact_complement": bool(
            np.array_equal(low_mask + high_mask, np.ones_like(low_mask))
        ),
        "mask_conjugate_symmetric": bool(
            mask_is_conjugate_symmetric(low_mask)
            and mask_is_conjugate_symmetric(high_mask)
        ),
        "dc_in_low_band": bool(low_mask[IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2] == 1.0),
        "rank_sum": band_projector_rank(radius, "low")
        + band_projector_rank(radius, "high"),
        "reconstruction_error_max": float(np.max(np.abs(sample - low - high))),
        "low_idempotence_error_max": float(np.max(np.abs(low_twice - low))),
        "high_idempotence_error_max": float(np.max(np.abs(high_twice - high))),
        "orthogonality_inner_product_max": float(np.max(np.abs(cross_inner))),
        "parseval_energy_error_max": float(
            np.max(np.abs(original_energy - decomposed_energy))
        ),
        "maximum_inverse_fft_imaginary_residual": max(low_imaginary, high_imaginary),
    }


def evaluate_frequency_restricted_geometry(
    training: FloatArray,
    test: FloatArray,
    *,
    cutoffs: Sequence[int],
    sigmas: Sequence[float],
    shell_c: float,
    posterior_draws: int,
    coverage_draws: int,
    bootstrap_replicates: int,
    seed: int,
    query_batch_size: int,
    reference_batch_size: int,
    q_coverage: float = 0.8,
    q_posterior_weight: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Compute low/high geometry with paired full-dimensional noise draws."""
    train = np.asarray(training, dtype=np.float64)
    held_out = np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or held_out.ndim != 2:
        raise ValueError("training and test must be flattened matrices")
    if train.shape[1] != IMAGE_DIMENSION or held_out.shape[1] != IMAGE_DIMENSION:
        raise ValueError("E004B requires flattened 32x32 RGB inputs")
    sigma_grid = tuple(float(value) for value in sigmas)
    if not sigma_grid or any(
        not np.isfinite(value) or value <= 0 for value in sigma_grid
    ):
        raise ValueError("sigmas must be positive and finite")
    cutoff_values = tuple(int(value) for value in cutoffs)
    if not cutoff_values or len(set(cutoff_values)) != len(cutoff_values):
        raise ValueError("cutoffs must be nonempty and unique")

    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"cutoffs": {}, "status": "pass"}
    targets: dict[str, Any] = {"cutoffs": {}}
    for radius in cutoff_values:
        rng = np.random.default_rng(seed)
        posterior_noise = _noise_draws(train.shape, posterior_draws, rng)
        coverage_noise = _noise_draws(held_out.shape, coverage_draws, rng)
        cutoff_validation = _projector_validation(train[: min(16, len(train))], radius)
        cutoff_validation["posterior_noise_sha256"] = _draw_digest(posterior_noise)
        cutoff_validation["coverage_noise_sha256"] = _draw_digest(coverage_noise)
        cutoff_validation["bands"] = {}
        cutoff_targets: dict[str, Any] = {}
        for band in ("low", "high"):
            curves, band_validation = _evaluate_projected_band(
                train,
                held_out,
                posterior_noise,
                coverage_noise,
                radius=radius,
                band=band,
                sigmas=sigma_grid,
                shell_c=shell_c,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed + 100_000 + 22,
                query_batch_size=query_batch_size,
                reference_batch_size=reference_batch_size,
            )
            cutoff_validation["bands"][band] = band_validation
            band_rows: list[dict[str, Any]] = []
            for sigma_index, sigma in enumerate(sigma_grid):
                coverage = float(curves["coverage_estimate"][sigma_index])
                coverage_low = float(curves["coverage_ci95_low"][sigma_index])
                posterior = float(curves["posterior_weight_estimate"][sigma_index])
                posterior_low = float(curves["posterior_weight_ci95_low"][sigma_index])
                row = {
                    "cutoff": radius,
                    "band": band,
                    "sigma_index": sigma_index,
                    "sigma": sigma,
                    "band_dimension": band_validation["band_dimension"],
                    "coverage_estimate": coverage,
                    "coverage_ci95_low": coverage_low,
                    "coverage_ci95_high": float(
                        curves["coverage_ci95_high"][sigma_index]
                    ),
                    "coverage_monte_carlo_se": float(
                        curves["coverage_monte_carlo_se"][sigma_index]
                    ),
                    "posterior_weight_estimate": posterior,
                    "posterior_weight_ci95_low": posterior_low,
                    "posterior_weight_ci95_high": float(
                        curves["posterior_weight_ci95_high"][sigma_index]
                    ),
                    "posterior_weight_monte_carlo_se": float(
                        curves["posterior_weight_monte_carlo_se"][sigma_index]
                    ),
                    "coverage_ge_q_c": coverage >= q_coverage,
                    "posterior_ge_q_w": posterior >= q_posterior_weight,
                    "high_high_point_estimate": coverage >= q_coverage
                    and posterior >= q_posterior_weight,
                    "high_high_lower_bound": coverage_low >= q_coverage
                    and posterior_low >= q_posterior_weight,
                }
                rows.append(row)
                band_rows.append(row)
            point_indices = high_high_indices(
                [float(row["coverage_estimate"]) for row in band_rows],
                [float(row["posterior_weight_estimate"]) for row in band_rows],
                q_coverage=q_coverage,
                q_posterior_weight=q_posterior_weight,
            )
            lower_indices = high_high_indices(
                [float(row["coverage_ci95_low"]) for row in band_rows],
                [float(row["posterior_weight_ci95_low"]) for row in band_rows],
                q_coverage=q_coverage,
                q_posterior_weight=q_posterior_weight,
            )
            cutoff_targets[band] = {
                "band_dimension": band_validation["band_dimension"],
                "point_estimate_indices": point_indices,
                "point_estimate_components": contiguous_components(point_indices),
                "lower_bound_indices": lower_indices,
                "lower_bound_components": contiguous_components(lower_indices),
            }
        validation["cutoffs"][str(radius)] = cutoff_validation
        targets["cutoffs"][str(radius)] = cutoff_targets
        projector_pass = (
            cutoff_validation["mask_binary"]
            and cutoff_validation["mask_exact_complement"]
            and cutoff_validation["mask_conjugate_symmetric"]
            and cutoff_validation["dc_in_low_band"]
            and cutoff_validation["rank_sum"] == IMAGE_DIMENSION
            and cutoff_validation["reconstruction_error_max"] <= 1e-12
            and cutoff_validation["low_idempotence_error_max"] <= 1e-12
            and cutoff_validation["high_idempotence_error_max"] <= 1e-12
            and cutoff_validation["orthogonality_inner_product_max"] <= 1e-10
            and cutoff_validation["parseval_energy_error_max"] <= 1e-10
            and cutoff_validation["maximum_inverse_fft_imaginary_residual"] <= 1e-12
            and all(
                details["status"] == "pass"
                for details in cutoff_validation["bands"].values()
            )
        )
        cutoff_validation["status"] = "pass" if projector_pass else "fail"
        if not projector_pass:
            validation["status"] = "fail"
    return rows, validation, targets


def summarize_targets(
    targets: Mapping[str, Any],
    *,
    primary_cutoff: int,
    q_coverage: float,
    q_weight: float,
) -> dict[str, Any]:
    """Build the stable primary target and adjacent-cutoff sensitivity summary."""
    primary = targets["cutoffs"][str(primary_cutoff)]
    summary: dict[str, Any] = {
        "primary_cutoff": primary_cutoff,
        "q_C": q_coverage,
        "q_W": q_weight,
        "low_band_dimension": primary["low"]["band_dimension"],
        "high_band_dimension": primary["high"]["band_dimension"],
        "low_point_estimate_indices": primary["low"]["point_estimate_indices"],
        "low_lower_bound_indices": primary["low"]["lower_bound_indices"],
        "high_point_estimate_indices": primary["high"]["point_estimate_indices"],
        "high_lower_bound_indices": primary["high"]["lower_bound_indices"],
        "connected_components": {
            "low_point_estimate": primary["low"]["point_estimate_components"],
            "low_lower_bound": primary["low"]["lower_bound_components"],
            "high_point_estimate": primary["high"]["point_estimate_components"],
            "high_lower_bound": primary["high"]["lower_bound_components"],
        },
        "cutoff_stability": {},
    }
    for cutoff, cutoff_targets in targets["cutoffs"].items():
        summary["cutoff_stability"][cutoff] = {
            band: {
                "lower_bound_indices": details["lower_bound_indices"],
                "matches_primary": details["lower_bound_indices"]
                == primary[band]["lower_bound_indices"],
            }
            for band, details in cutoff_targets.items()
        }
    return summary
