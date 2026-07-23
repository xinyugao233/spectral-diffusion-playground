"""Noise-generation helpers for diffusion-focused experiments.

This module is reserved for noise construction, schedule-related helpers, and
small reproducible perturbation utilities used by multiple experiments.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> FloatArray:
    """Add deterministic Gaussian noise to a normalized image.

    Parameters
    ----------
    image:
        Grayscale or RGB image normalized to ``[0, 1]``.
    sigma:
        Standard deviation of the additive Gaussian perturbation.
    seed:
        Random seed used to make the perturbation reproducible.

    Notes
    -----
    The returned array is intentionally not clipped. This keeps the operation
    faithful to ``x_sigma = x + sigma * epsilon``. Display helpers should clip
    only when rendering the noisy image for visualization.
    """
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim not in (2, 3):
        raise ValueError(
            "Expected a grayscale image with shape (H, W) or an RGB image with "
            f"shape (H, W, C), but received shape {image_array.shape}."
        )
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative.")
    if image_array.min() < 0.0 or image_array.max() > 1.0:
        raise ValueError("image must be normalized to the range [0, 1].")

    rng = np.random.default_rng(seed)
    epsilon = rng.normal(loc=0.0, scale=1.0, size=image_array.shape)
    return np.asarray(image_array + sigma * epsilon, dtype=np.float64)
