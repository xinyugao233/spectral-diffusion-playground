"""Reusable filters for centered two-dimensional Fourier spectra."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spectral_diffusion_playground.fft import compute_fft, compute_ifft, shift_fft

FloatArray = NDArray[np.float64]


def create_frequency_mask(height: int, width: int, radius: float) -> FloatArray:
    """Create a circular low-pass mask in centered Fourier coordinates.

    Parameters
    ----------
    height:
        Number of rows in the spatial image and Fourier spectrum.
    width:
        Number of columns in the spatial image and Fourier spectrum.
    radius:
        Inclusive radial cutoff in Fourier pixels. ``radius=0`` retains only
        the centered DC coefficient.

    Returns
    -------
    np.ndarray
        A binary ``float64`` mask with shape ``(height, width)``. Coefficients
        whose Euclidean distance from ``(height // 2, width // 2)`` is at most
        ``radius`` are one; all other coefficients are zero.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers.")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be a finite, nonnegative value.")

    yy, xx = np.ogrid[:height, :width]
    center_y = height // 2
    center_x = width // 2
    squared_distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
    return np.asarray(squared_distance <= radius**2, dtype=np.float64)


def _validate_centered_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Validate a centered grayscale or multichannel Fourier spectrum."""
    spectrum_array = np.asarray(spectrum)
    if spectrum_array.ndim not in (2, 3):
        raise ValueError(
            "Expected a centered spectrum with shape (H, W) or (H, W, C), "
            f"but received shape {spectrum_array.shape}."
        )
    if not np.iscomplexobj(spectrum_array):
        raise ValueError("spectrum must contain complex Fourier coefficients.")
    return spectrum_array


def _broadcast_mask(mask: FloatArray, spectrum: np.ndarray) -> np.ndarray:
    """Add a channel axis when applying a 2D mask to an RGB spectrum."""
    return mask if spectrum.ndim == 2 else mask[..., np.newaxis]


def low_pass_filter(spectrum: np.ndarray, radius: float) -> np.ndarray:
    """Retain frequencies within ``radius`` of a centered spectrum's origin.

    ``spectrum`` must already use centered coordinates, for example the output
    of :func:`spectral_diffusion_playground.fft.shift_fft`.
    """
    spectrum_array = _validate_centered_spectrum(spectrum)
    height, width = spectrum_array.shape[:2]
    mask = create_frequency_mask(height, width, radius)
    return np.asarray(spectrum_array * _broadcast_mask(mask, spectrum_array))


def high_pass_filter(spectrum: np.ndarray, radius: float) -> np.ndarray:
    """Remove frequencies within ``radius`` of a centered spectrum's origin.

    The cutoff is complementary to :func:`low_pass_filter`: coefficients at
    exactly ``radius`` are retained by the low-pass result and removed here.
    """
    spectrum_array = _validate_centered_spectrum(spectrum)
    height, width = spectrum_array.shape[:2]
    high_pass_mask = 1.0 - create_frequency_mask(height, width, radius)
    return np.asarray(spectrum_array * _broadcast_mask(high_pass_mask, spectrum_array))


def decompose_frequency_bands(
    image: np.ndarray,
    radius: float,
) -> tuple[FloatArray, FloatArray]:
    """Decompose an image into complementary low- and high-frequency images.

    The input is transformed with the repository's orthonormal FFT. The
    low-frequency component retains centered coefficients at distances
    ``<= radius``; the high-frequency component contains the exact complement.
    Their sum reconstructs the input up to floating-point precision.
    """
    centered_spectrum = shift_fft(compute_fft(image))
    low_frequency = compute_ifft(
        low_pass_filter(centered_spectrum, radius),
        is_shifted=True,
    )
    high_frequency = compute_ifft(
        high_pass_filter(centered_spectrum, radius),
        is_shifted=True,
    )
    return low_frequency, high_frequency
