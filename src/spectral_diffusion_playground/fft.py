"""FFT helpers shared across spectral analysis experiments."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SPATIAL_AXES: Final[tuple[int, int]] = (0, 1)


def _validate_image_or_spectrum(array: np.ndarray) -> np.ndarray:
    """Validate the array rank used by image-space and frequency-space helpers."""
    validated = np.asarray(array)
    if validated.ndim not in (2, 3):
        raise ValueError(
            "Expected a grayscale image with shape (H, W) or an RGB image with "
            f"shape (H, W, C), but received shape {validated.shape}."
        )
    return validated


def compute_fft(image: np.ndarray) -> ComplexArray:
    """Compute a 2D Fourier transform over the spatial axes of an image.

    The transform is always applied over the first two dimensions so that
    grayscale images and RGB images can share the same implementation.
    """
    image_array = _validate_image_or_spectrum(np.asarray(image, dtype=np.float64))
    return np.asarray(fft2(image_array, axes=SPATIAL_AXES, norm="ortho"))


def shift_fft(spectrum: np.ndarray) -> ComplexArray:
    """Shift the zero-frequency component to the center of the spectrum image."""
    spectrum_array = _validate_image_or_spectrum(np.asarray(spectrum))
    return np.asarray(fftshift(spectrum_array, axes=SPATIAL_AXES))


def unshift_fft(spectrum: np.ndarray) -> ComplexArray:
    """Undo the centering performed by :func:`shift_fft`."""
    spectrum_array = _validate_image_or_spectrum(np.asarray(spectrum))
    return np.asarray(ifftshift(spectrum_array, axes=SPATIAL_AXES))


def magnitude_spectrum(spectrum: np.ndarray) -> FloatArray:
    """Return the magnitude of a complex Fourier spectrum."""
    spectrum_array = _validate_image_or_spectrum(np.asarray(spectrum))
    return np.asarray(np.abs(spectrum_array), dtype=np.float64)


def log_magnitude(magnitude: np.ndarray, *, epsilon: float = 1e-12) -> FloatArray:
    """Compress a magnitude spectrum with a numerically stable logarithm."""
    magnitude_array = _validate_image_or_spectrum(np.asarray(magnitude, dtype=np.float64))
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")
    return np.asarray(np.log1p(np.maximum(magnitude_array, 0.0) + epsilon))


def compute_ifft(spectrum: np.ndarray, *, is_shifted: bool = False) -> FloatArray:
    """Invert a 2D Fourier spectrum back into image space.

    Parameters
    ----------
    spectrum:
        Complex-valued Fourier coefficients.
    is_shifted:
        If ``True``, ``spectrum`` is assumed to be centered with
        :func:`shift_fft` and will be unshifted before inversion.
    """
    spectrum_array = _validate_image_or_spectrum(np.asarray(spectrum))
    if is_shifted:
        spectrum_array = unshift_fft(spectrum_array)

    reconstructed = ifft2(spectrum_array, axes=SPATIAL_AXES, norm="ortho")
    return np.asarray(reconstructed.real, dtype=np.float64)
