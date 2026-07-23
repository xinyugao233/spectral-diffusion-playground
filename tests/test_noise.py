"""Tests for deterministic Gaussian noise helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.noise import add_gaussian_noise
from spectral_diffusion_playground.utils import create_reference_image


class GaussianNoiseTest(unittest.TestCase):
    """Verify reproducible additive Gaussian perturbations."""

    def test_noise_is_reproducible_for_a_fixed_seed(self) -> None:
        image = create_reference_image(size=64)

        first = add_gaussian_noise(image, sigma=0.2, seed=7)
        second = add_gaussian_noise(image, sigma=0.2, seed=7)

        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)

    def test_sigma_zero_returns_the_original_image(self) -> None:
        image = create_reference_image(size=64)
        noisy = add_gaussian_noise(image, sigma=0.0, seed=123)

        np.testing.assert_allclose(noisy, image, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
