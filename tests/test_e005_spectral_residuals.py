"""Tests for E005 spectral residual numerical utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.e005_spectral_residuals import (
    CUTOFFS,
    REDUCTION_ELEMENTS,
    SIGMA_GRID,
    compute_projection_energy,
    deterministic_noise,
    extract_transition_window,
    seed_sequence_material,
    validate_projection_energy,
)


class E005SpectralResidualTest(unittest.TestCase):
    """Validate frozen E005 numerical behavior without model inference."""

    def test_sigma_grid_is_frozen_18_point_descending_grid(self) -> None:
        self.assertEqual(len(SIGMA_GRID), 18)
        self.assertEqual(SIGMA_GRID[0], 80.0)
        self.assertAlmostEqual(SIGMA_GRID[-1], 0.0020000000000000031)
        self.assertTrue(
            all(left > right for left, right in zip(SIGMA_GRID, SIGMA_GRID[1:]))
        )

    def test_cutoffs_are_frozen(self) -> None:
        self.assertEqual(CUTOFFS, (3, 4, 5, 6))

    def test_seed_material_and_noise_are_deterministic(self) -> None:
        material = seed_sequence_material(
            split_code=1,
            dataset_index=42,
            noise_repeat=3,
            sigma_index=7,
        )
        self.assertEqual(material, [20260726, 1, 42, 3, 7])
        noise_a, seed_a = deterministic_noise(
            (3, 32, 32),
            split_code=1,
            dataset_index=42,
            noise_repeat=3,
            sigma_index=7,
        )
        noise_b, seed_b = deterministic_noise(
            (3, 32, 32),
            split_code=1,
            dataset_index=42,
            noise_repeat=3,
            sigma_index=7,
        )
        self.assertEqual(seed_a, seed_b)
        np.testing.assert_array_equal(noise_a, noise_b)

    def test_projection_energy_is_additive_and_orthogonal(self) -> None:
        yy, xx = np.indices((32, 32))
        residual = np.stack(
            [
                np.sin(2 * np.pi * xx / 32),
                np.cos(2 * np.pi * yy / 32),
                np.sin(2 * np.pi * (xx + yy) / 16),
            ],
            axis=2,
        ).astype(np.float64)

        energy = compute_projection_energy(residual, cutoff=4)
        validate_projection_energy(energy)

        self.assertAlmostEqual(energy.full, energy.low + energy.high, places=10)
        self.assertAlmostEqual(energy.full_mse, energy.full / REDUCTION_ELEMENTS)

    def test_transition_extraction_reports_valid_window(self) -> None:
        curve = np.asarray(
            [10, 10, 9, 8, 7, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dtype=np.float64,
        )
        result = extract_transition_window(curve)

        self.assertEqual(result["status"], "ok")
        self.assertIsInstance(result["entry_index"], int)
        self.assertIsInstance(result["exit_index"], int)

    def test_transition_extraction_reports_no_clear_transition(self) -> None:
        curve = np.ones(18, dtype=np.float64)
        result = extract_transition_window(curve)

        self.assertEqual(result["status"], "no_clear_transition")
        self.assertEqual(result["reason"], "invalid_endpoint_denominator")


if __name__ == "__main__":
    unittest.main()
