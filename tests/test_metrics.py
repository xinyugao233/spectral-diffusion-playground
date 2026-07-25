"""Tests for low- and high-frequency recovery metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.filters import decompose_frequency_bands
from spectral_diffusion_playground.metrics import (
    image_frequency_band_recovery_scores,
    recovery_score,
    relative_l2_error,
)
from spectral_diffusion_playground.utils import create_reference_image


class FrequencyBandMetricsTest(unittest.TestCase):
    """Verify score endpoints, amplitude sensitivity, and band selectivity."""

    def test_exact_recovery_scores_one(self) -> None:
        target = create_reference_image(size=64)

        self.assertAlmostEqual(recovery_score(target, target), 1.0)

    def test_zero_estimate_scores_zero(self) -> None:
        target = create_reference_image(size=64)

        self.assertAlmostEqual(recovery_score(np.zeros_like(target), target), 0.0)

    def test_recovery_score_tracks_signal_amplitude(self) -> None:
        target = create_reference_image(size=64)

        self.assertAlmostEqual(recovery_score(0.4 * target, target), 0.4)

    def test_low_band_only_prediction_is_band_selective(self) -> None:
        target = create_reference_image(size=64)
        target_low, _ = decompose_frequency_bands(target, radius=12)

        scores = image_frequency_band_recovery_scores(
            target_low,
            target,
            radius=12,
        )

        self.assertAlmostEqual(scores.low_score, 1.0, places=10)
        self.assertAlmostEqual(scores.high_score, 0.0, places=10)

    def test_high_band_only_prediction_is_band_selective(self) -> None:
        target = create_reference_image(size=64)
        _, target_high = decompose_frequency_bands(target, radius=12)

        scores = image_frequency_band_recovery_scores(
            target_high,
            target,
            radius=12,
        )

        self.assertAlmostEqual(scores.low_score, 0.0, places=10)
        self.assertAlmostEqual(scores.high_score, 1.0, places=10)

    def test_scores_ignore_constant_brightness_shift_by_default(self) -> None:
        target = create_reference_image(size=64)
        shifted = target + 0.2

        scores = image_frequency_band_recovery_scores(
            shifted,
            target,
            radius=12,
        )

        self.assertAlmostEqual(scores.low_score, 1.0, places=10)
        self.assertAlmostEqual(scores.high_score, 1.0, places=10)

    def test_relative_error_rejects_zero_energy_target(self) -> None:
        zeros = np.zeros((8, 8), dtype=np.float64)

        with self.assertRaises(ValueError):
            relative_l2_error(zeros, zeros)


if __name__ == "__main__":
    unittest.main()
