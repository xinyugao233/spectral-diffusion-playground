"""Tests for centered Fourier-domain filtering utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.fft import compute_fft, compute_ifft, shift_fft
from spectral_diffusion_playground.filters import (
    create_frequency_mask,
    decompose_frequency_bands,
    high_pass_filter,
    low_pass_filter,
)
from spectral_diffusion_playground.utils import create_reference_image


class FrequencyFilterTest(unittest.TestCase):
    """Verify mask geometry and reconstruction behavior."""

    def test_mask_contains_center_and_excludes_corner(self) -> None:
        height, width = 32, 48
        mask = create_frequency_mask(height, width, radius=5)

        self.assertEqual(mask[height // 2, width // 2], 1.0)
        self.assertEqual(mask[0, 0], 0.0)
        self.assertTrue(np.all(np.logical_or(mask == 0.0, mask == 1.0)))

    def test_radius_zero_keeps_only_dc_coefficient(self) -> None:
        mask = create_frequency_mask(17, 19, radius=0)

        self.assertEqual(float(mask.sum()), 1.0)
        self.assertEqual(mask[17 // 2, 19 // 2], 1.0)

    def test_large_radius_recovers_original_image(self) -> None:
        image = create_reference_image(size=64)
        centered_spectrum = shift_fft(compute_fft(image))
        full_radius = float(np.hypot(image.shape[0], image.shape[1]))

        filtered = low_pass_filter(centered_spectrum, radius=full_radius)
        reconstruction = compute_ifft(filtered, is_shifted=True)

        np.testing.assert_allclose(reconstruction, image, atol=1e-10)

    def test_low_and_high_pass_filters_are_complementary(self) -> None:
        image = create_reference_image(size=64)
        centered_spectrum = shift_fft(compute_fft(image))

        low_frequency = low_pass_filter(centered_spectrum, radius=12)
        high_frequency = high_pass_filter(centered_spectrum, radius=12)

        np.testing.assert_allclose(
            low_frequency + high_frequency,
            centered_spectrum,
            atol=0.0,
            rtol=0.0,
        )

    def test_high_pass_reconstruction_equals_spatial_residual(self) -> None:
        image = create_reference_image(size=64)
        centered_spectrum = shift_fft(compute_fft(image))

        low_reconstruction = compute_ifft(
            low_pass_filter(centered_spectrum, radius=12),
            is_shifted=True,
        )
        high_reconstruction = compute_ifft(
            high_pass_filter(centered_spectrum, radius=12),
            is_shifted=True,
        )

        np.testing.assert_allclose(
            high_reconstruction,
            image - low_reconstruction,
            atol=1e-10,
        )

    def test_frequency_band_decomposition_reconstructs_image(self) -> None:
        image = create_reference_image(size=64)

        low_frequency, high_frequency = decompose_frequency_bands(image, radius=12)

        np.testing.assert_allclose(
            low_frequency + high_frequency,
            image,
            atol=1e-10,
        )

    def test_invalid_radius_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_frequency_mask(32, 32, radius=-1)


if __name__ == "__main__":
    unittest.main()
