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


def power_spectrum(spectrum: np.ndarray) -> FloatArray:
    """Return the squared magnitude of a complex Fourier spectrum."""
    magnitude = magnitude_spectrum(spectrum)
    return np.asarray(magnitude**2, dtype=np.float64)


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


def radial_frequency_energy(
    spectrum: np.ndarray,
    *,
    is_shifted: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Compute annulus-averaged spectral energy as a function of frequency radius.

    Parameters
    ----------
    spectrum:
        Complex-valued Fourier coefficients.
    is_shifted:
        If ``True``, ``spectrum`` is assumed to already be centered with
        :func:`shift_fft`. Otherwise the spectrum is centered before radial
        aggregation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer-valued radial bins and the corresponding annulus-averaged power
        density ``E(r)``. For RGB spectra, channel power is averaged before
        radial aggregation so the curve remains directly comparable across
        grayscale and RGB inputs.
    """
    spectrum_array = _validate_image_or_spectrum(np.asarray(spectrum))
    centered_spectrum = spectrum_array if is_shifted else shift_fft(spectrum_array)
    power = power_spectrum(centered_spectrum)
    if power.ndim == 3:
        power = power.mean(axis=2)

    height, width = power.shape
    center_y = height // 2
    center_x = width // 2
    yy, xx = np.indices((height, width))
    radii = np.rint(np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)).astype(int)

    flat_radii = radii.ravel()
    flat_power = power.ravel()
    summed_energy = np.bincount(flat_radii, weights=flat_power)
    counts = np.bincount(flat_radii)
    valid = counts > 0

    radial_bins = np.arange(counts.size, dtype=np.float64)[valid]
    mean_energy = summed_energy[valid] / counts[valid]
    return np.asarray(radial_bins, dtype=np.float64), np.asarray(mean_energy, dtype=np.float64)


def normalize_radial_energy(radial_energy: np.ndarray) -> FloatArray:
    """Normalize a nonnegative radial energy profile into a unit-sum distribution."""
    energy_array = np.asarray(radial_energy, dtype=np.float64)
    if energy_array.ndim != 1:
        raise ValueError(
            "Expected a 1D radial energy profile, "
            f"but received shape {energy_array.shape}."
        )
    if np.any(energy_array < 0.0):
        raise ValueError("radial_energy must be nonnegative.")

    total_energy = float(np.sum(energy_array))
    if total_energy <= 0.0:
        raise ValueError("radial_energy must contain positive total energy.")
    return np.asarray(energy_array / total_energy, dtype=np.float64)
