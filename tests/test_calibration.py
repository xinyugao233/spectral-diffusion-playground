"""Tests for shared controlled frequency-band calibration utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.calibration import (
    bootstrap_mean_interval,
    evaluate_controlled_trajectories,
    evaluate_cutoff_curves,
    first_threshold_crossing,
    ordering_matches_control,
)
from spectral_diffusion_playground.utils import create_reference_image


class ControlledCalibrationUtilitiesTest(unittest.TestCase):
    """Verify shared calibration logic without changing metric definitions."""

    def test_optimized_cutoff_curves_match_direct_metric_evaluation(self) -> None:
        """The Gram-matrix path should reproduce direct per-step scoring."""
        image = create_reference_image(size=64)
        progress = np.linspace(0.0, 1.0, 21, dtype=np.float64)
        direct_results = evaluate_controlled_trajectories(
            image,
            progress=progress,
            construction_radius=10.0,
            evaluation_radius=10.0,
            noise_level=0.05,
            seed=0,
            frame_indices=np.asarray([0, 10, 20]),
        )
        optimized_results = evaluate_cutoff_curves(
            image,
            progress=progress,
            construction_radius=10.0,
            evaluation_radii=(10.0,),
            seeds=(0,),
            noise_level=0.05,
        )
        optimized_by_name = {
            result.schedule.name: result for result in optimized_results
        }

        for direct in direct_results:
            optimized = optimized_by_name[direct.schedule.name]
            np.testing.assert_allclose(
                optimized.low_score,
                direct.low_score,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                optimized.high_score,
                direct.high_score,
                atol=1e-12,
            )

    def test_first_threshold_crossing_uses_first_reached_progress(self) -> None:
        """Crossing timing should use the first score at or above threshold."""
        progress = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        score = np.asarray([0.0, 0.4, 0.8, 0.9, 1.0])

        crossing = first_threshold_crossing(progress, score, threshold=0.8)

        self.assertEqual(crossing, 0.5)

    def test_ordering_checks_match_all_three_controls(self) -> None:
        """Known crossing patterns should satisfy their corresponding controls."""
        self.assertTrue(
            ordering_matches_control(
                "low_band_first",
                low_crossing=0.4,
                high_crossing=0.8,
                progress_step=0.01,
            )
        )
        self.assertTrue(
            ordering_matches_control(
                "high_band_first",
                low_crossing=0.8,
                high_crossing=0.4,
                progress_step=0.01,
            )
        )
        self.assertTrue(
            ordering_matches_control(
                "together",
                low_crossing=0.68,
                high_crossing=0.69,
                progress_step=0.01,
            )
        )

    def test_bootstrap_interval_resamples_image_axis(self) -> None:
        """Bootstrap indices should index images and preserve trailing axes."""
        values = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
        indices = np.asarray([[0, 0, 0], [2, 2, 2], [0, 1, 2]])

        lower, upper = bootstrap_mean_interval(
            values,
            indices,
            confidence_level=0.5,
        )

        self.assertEqual(lower.shape, (2,))
        self.assertEqual(upper.shape, (2,))
        self.assertTrue(np.all(lower <= upper))


if __name__ == "__main__":
    unittest.main()
