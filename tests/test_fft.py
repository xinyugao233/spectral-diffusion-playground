"""Tests for reusable FFT and image utility functions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.fft import (
    compute_fft,
    compute_ifft,
    log_magnitude,
    magnitude_spectrum,
    normalize_radial_energy,
    shift_fft,
)
from spectral_diffusion_playground.utils import (
    create_reference_image,
    ensure_default_reference_image,
    rgb_to_grayscale,
)
from spectral_diffusion_playground.visualization import (
    normalize_signed_fields,
    spectrum_to_display_image,
)


class FFTUtilitiesTest(unittest.TestCase):
    """Check that Fourier helpers are numerically stable and reusable."""

    def test_fft_round_trip_reconstructs_rgb_image(self) -> None:
        image = create_reference_image(size=96)
        shifted_spectrum = shift_fft(compute_fft(image))
        reconstruction = compute_ifft(shifted_spectrum, is_shifted=True)

        np.testing.assert_allclose(reconstruction, image, atol=1e-10)

    def test_log_magnitude_preserves_shape_and_nonnegativity(self) -> None:
        image = create_reference_image(size=64)
        shifted_spectrum = shift_fft(compute_fft(image))
        magnitude = magnitude_spectrum(shifted_spectrum)
        compressed = log_magnitude(magnitude)

        self.assertEqual(compressed.shape, magnitude.shape)
        self.assertTrue(np.all(compressed >= 0.0))

    def test_rgb_to_grayscale_returns_single_channel_image(self) -> None:
        image = create_reference_image(size=64)
        grayscale = rgb_to_grayscale(image)

        self.assertEqual(grayscale.shape, image.shape[:2])
        self.assertTrue(np.all(grayscale >= 0.0))
        self.assertTrue(np.all(grayscale <= 1.0))

    def test_default_reference_image_is_created_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "default_fft_reference.png"
            created_path = ensure_default_reference_image(output_path, size=80)

            self.assertTrue(created_path.exists())

    def test_spectrum_display_image_is_normalized(self) -> None:
        image = create_reference_image(size=72)
        display_image = spectrum_to_display_image(
            log_magnitude(magnitude_spectrum(shift_fft(compute_fft(image)))),
            normalization="max",
        )

        self.assertEqual(display_image.ndim, 2)
        self.assertGreaterEqual(display_image.min(), 0.0)
        self.assertLessEqual(display_image.max(), 1.0)

    def test_max_normalization_preserves_strongest_coefficient(self) -> None:
        scalar_field = np.asarray([[0.0, 2.0], [1.0, 8.0]], dtype=np.float64)
        display_image = spectrum_to_display_image(scalar_field, normalization="max")

        np.testing.assert_allclose(display_image, scalar_field / 8.0)

    def test_normalized_radial_energy_sums_to_one(self) -> None:
        radial_energy = np.asarray([2.0, 3.0, 5.0], dtype=np.float64)
        normalized = normalize_radial_energy(radial_energy)

        self.assertTrue(np.all(normalized >= 0.0))
        self.assertAlmostEqual(float(normalized.sum()), 1.0)

    def test_signed_field_normalization_uses_shared_symmetric_scale(self) -> None:
        fields = [
            np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float64),
            np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float64),
        ]

        normalized = normalize_signed_fields(fields, percentile=100.0)

        np.testing.assert_allclose(normalized[0], [[0.0, 0.5, 1.0]])
        np.testing.assert_allclose(normalized[1], [[0.25, 0.5, 0.75]])


if __name__ == "__main__":
    unittest.main()
