"""Metrics for measuring low- and high-frequency image recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectral_diffusion_playground.filters import decompose_frequency_bands


@dataclass(frozen=True, slots=True)
class FrequencyBandScores:
    """Frequency-band recovery scores and their underlying relative errors."""

    low_score: float
    high_score: float
    low_relative_error: float
    high_relative_error: float


def relative_l2_error(estimate: np.ndarray, target: np.ndarray) -> float:
    """Return ``||estimate - target||_2 / ||target||_2``.

    The target must contain nonzero energy. The function does not clip either
    input, so callers can distinguish computation from display normalization.
    """
    estimate_array = np.asarray(estimate, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if estimate_array.shape != target_array.shape:
        raise ValueError("estimate and target must have identical shapes.")

    target_norm = float(np.linalg.norm(target_array.ravel()))
    if target_norm <= np.finfo(np.float64).eps:
        raise ValueError("target must contain nonzero L2 energy.")
    error_norm = float(np.linalg.norm((estimate_array - target_array).ravel()))
    return error_norm / target_norm


def recovery_score(estimate: np.ndarray, target: np.ndarray) -> float:
    """Measure recovery relative to the zero-estimate baseline.

    The score is ``max(0, 1 - relative_l2_error(estimate, target))``. Exact
    recovery scores one, a zero estimate scores zero, and estimates worse than
    the zero baseline remain at zero. Use :func:`relative_l2_error` when the
    magnitude of worse-than-baseline errors must remain visible.
    """
    return max(0.0, 1.0 - relative_l2_error(estimate, target))


def frequency_band_components(
    image: np.ndarray,
    *,
    radius: float,
    exclude_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low- and high-frequency bands for metric evaluation.

    The high-frequency component is the complementary high-pass reconstruction.
    The low-frequency component is the circular low-pass reconstruction with
    its per-channel spatial mean removed by default. Removing this DC component
    prevents global brightness or mean color from dominating the low-band score.
    """
    low_frequency, high_frequency = decompose_frequency_bands(image, radius)
    if exclude_dc:
        spatial_mean = np.asarray(image, dtype=np.float64).mean(
            axis=(0, 1),
            keepdims=True,
        )
        low_frequency = low_frequency - spatial_mean
    return low_frequency, high_frequency


def image_frequency_band_recovery_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    radius: float,
    exclude_dc: bool = True,
) -> FrequencyBandScores:
    """Measure low- and high-frequency recovery for two images.

    ``radius`` defines both projections in centered Fourier pixels. By default,
    the centered DC coefficient is removed from the low-pass band so global
    mean color cannot dominate the low-band score. Low-frequency recovery can
    serve as a coarse/global-structure proxy and high-frequency recovery as a
    fine-detail proxy, but the metric itself makes no semantic claim.
    """
    prediction_low, prediction_high = frequency_band_components(
        prediction,
        radius=radius,
        exclude_dc=exclude_dc,
    )
    target_low, target_high = frequency_band_components(
        target,
        radius=radius,
        exclude_dc=exclude_dc,
    )
    return frequency_band_recovery_scores(
        prediction_low,
        prediction_high,
        target_low,
        target_high,
    )


def frequency_band_recovery_scores(
    prediction_low: np.ndarray,
    prediction_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> FrequencyBandScores:
    """Score precomputed low- and high-frequency prediction components."""
    low_error = relative_l2_error(prediction_low, target_low)
    high_error = relative_l2_error(prediction_high, target_high)

    return FrequencyBandScores(
        low_score=max(0.0, 1.0 - low_error),
        high_score=max(0.0, 1.0 - high_error),
        low_relative_error=low_error,
        high_relative_error=high_error,
    )
