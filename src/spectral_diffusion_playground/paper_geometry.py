"""Clean-room estimators for the paper's full-space CALM geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

IMAGE_DIMENSION = 3 * 32 * 32
SHELL_C = 5.0
SIGMA_GRID = (
    0.02,
    0.05,
    0.1,
    0.14,
    0.2,
    0.3,
    0.4,
    0.6,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    8.0,
    12.0,
    20.0,
    40.0,
    80.0,
)


def squared_distances(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return all pairwise squared Euclidean distances between two matrices."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("Inputs must be two-dimensional matrices")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("Inputs must share a feature dimension")
    distances = (
        np.square(left_array).sum(axis=1, keepdims=True)
        + np.square(right_array).sum(axis=1)[None, :]
        - 2.0 * (left_array @ right_array.T)
    )
    return np.maximum(distances, 0.0)


def normalized_weights_from_logits(logits: FloatArray) -> FloatArray:
    """Normalize log weights rowwise with log-sum-exp stabilization."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("logits must have shape (queries, references)")
    log_weights = values - logsumexp(values, axis=1, keepdims=True)
    return np.exp(log_weights)


def posterior_weights(
    noisy_queries: FloatArray,
    references: FloatArray,
    sigma: float,
) -> FloatArray:
    """Compute empirical posterior weights from paper Eq. (3)."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    distance_sq = squared_distances(noisy_queries, references)
    logits = -distance_sq / (2.0 * sigma * sigma)
    return normalized_weights_from_logits(logits)


def maximum_posterior_weight(
    noisy_queries: FloatArray,
    references: FloatArray,
    sigma: float,
) -> float:
    """Estimate W_sigma(D) by averaging each query's maximum posterior weight."""
    weights = posterior_weights(noisy_queries, references, sigma)
    return float(np.max(weights, axis=1).mean())


def gaussian_shell_radii(
    dimension: int,
    c_value: float = SHELL_C,
) -> tuple[float, float]:
    """Return normalized inner and outer radii from paper Lemma 4.5."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if not np.isfinite(c_value) or c_value <= 0.0:
        raise ValueError("c_value must be positive and finite")
    d = float(dimension)
    inner = math.sqrt(max(d - 2.0 * math.sqrt(c_value * d), 0.0))
    outer = math.sqrt(d + 2.0 * math.sqrt(c_value * d) + 2.0 * c_value)
    return inner, outer


def gaussian_shell_membership(
    noisy_queries: FloatArray,
    references: FloatArray,
    sigma: float,
    *,
    c_value: float = SHELL_C,
    reference_batch_size: int = 256,
) -> BoolArray:
    """Test membership in the exact union of training-centered Gaussian shells."""
    queries = np.asarray(noisy_queries, dtype=np.float64)
    centers = np.asarray(references, dtype=np.float64)
    if queries.ndim != 2 or centers.ndim != 2:
        raise ValueError("Inputs must be two-dimensional matrices")
    if queries.shape[1] != centers.shape[1]:
        raise ValueError("Inputs must share a feature dimension")
    if reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")

    inner, outer = gaussian_shell_radii(queries.shape[1], c_value)
    lower_sq = (sigma * inner) ** 2
    upper_sq = (sigma * outer) ** 2
    covered = np.zeros(len(queries), dtype=bool)
    for start in range(0, len(centers), reference_batch_size):
        distance_sq = squared_distances(
            queries,
            centers[start : start + reference_batch_size],
        )
        covered |= np.any(
            (distance_sq >= lower_sq) & (distance_sq <= upper_sq),
            axis=1,
        )
    return covered


def gaussian_shell_coverage(
    noisy_queries: FloatArray,
    references: FloatArray,
    sigma: float,
    *,
    c_value: float = SHELL_C,
    reference_batch_size: int = 256,
) -> float:
    """Estimate paper Definition 4.6 by averaging exact shell-union events."""
    membership = gaussian_shell_membership(
        noisy_queries,
        references,
        sigma,
        c_value=c_value,
        reference_batch_size=reference_batch_size,
    )
    return float(membership.mean())


def expanded_noisy_distances(
    clean_queries: FloatArray,
    references: FloatArray,
    noise: FloatArray,
    sigma: float,
) -> FloatArray:
    """Evaluate noisy distances using the exact quadratic expansion."""
    clean = np.asarray(clean_queries, dtype=np.float64)
    centers = np.asarray(references, dtype=np.float64)
    perturbation = np.asarray(noise, dtype=np.float64)
    if clean.shape != perturbation.shape:
        raise ValueError("clean_queries and noise must share a shape")
    base = squared_distances(clean, centers)
    cross = np.sum(perturbation * clean, axis=1)[:, None] - perturbation @ centers.T
    noise_energy = np.square(perturbation).sum(axis=1)[:, None]
    return np.maximum(base + 2.0 * sigma * cross + sigma * sigma * noise_energy, 0.0)


def validate_sigma_grid(sigmas: Sequence[float]) -> tuple[float, ...]:
    """Validate and freeze a strictly increasing positive sigma grid."""
    values = tuple(float(value) for value in sigmas)
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Sigma values must be positive and finite")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Sigma values must be strictly increasing")
    return values
