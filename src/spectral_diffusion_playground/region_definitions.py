"""Utilities for classifying evaluated geometric regions without interpolation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def high_high_indices(
    coverage: Sequence[float],
    posterior_weight: Sequence[float],
    *,
    q_coverage: float,
    q_posterior_weight: float,
) -> list[int]:
    """Return qualifying evaluated-point indices without interpolation."""
    coverage_values = np.asarray(coverage, dtype=np.float64)
    posterior_values = np.asarray(posterior_weight, dtype=np.float64)
    if coverage_values.shape != posterior_values.shape or coverage_values.ndim != 1:
        raise ValueError("coverage and posterior_weight must be equal-length vectors")
    if not np.all(np.isfinite(coverage_values)) or not np.all(
        np.isfinite(posterior_values)
    ):
        raise ValueError("geometry values must be finite")
    return np.flatnonzero(
        (coverage_values >= q_coverage) & (posterior_values >= q_posterior_weight)
    ).tolist()


def contiguous_components(indices: Sequence[int]) -> list[list[int]]:
    """Group consecutive indices while preserving every observed gap."""
    values = [int(index) for index in indices]
    if values != sorted(set(values)):
        raise ValueError("indices must be unique and increasing")
    components: list[list[int]] = []
    for index in values:
        if not components or index != components[-1][-1] + 1:
            components.append([index])
        else:
            components[-1].append(index)
    return components
