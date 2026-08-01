"""Tests for the paper-derived clean-room geometry utilities."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.paper_geometry import (
    SHELL_C,
    SIGMA_GRID,
    expanded_noisy_distances,
    gaussian_shell_coverage,
    gaussian_shell_membership,
    gaussian_shell_radii,
    normalized_weights_from_logits,
    posterior_weights,
    squared_distances,
)


class PaperGeometryTest(unittest.TestCase):
    """Validate exact posterior and Gaussian-shell definitions."""

    def test_posterior_weights_are_finite_normalized_and_bounded(self) -> None:
        queries = np.asarray([[0.1, 0.2], [0.8, 0.7]])
        references = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        weights = posterior_weights(queries, references, sigma=0.4)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        maximum = weights.max(axis=1)
        self.assertTrue(np.all(maximum >= 1.0 / len(references) - 1e-15))
        self.assertTrue(np.all(maximum <= 1.0 + 1e-15))

    def test_stable_and_direct_softmax_agree(self) -> None:
        logits = np.asarray([[0.1, -0.2, 0.4], [1.0, 1.2, -0.5]])
        direct = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(normalized_weights_from_logits(logits), direct)

    def test_weight_normalization_is_invariant_to_logit_shift(self) -> None:
        logits = np.asarray([[-1000.0, -1001.0], [20.0, 19.0]])
        np.testing.assert_allclose(
            normalized_weights_from_logits(logits),
            normalized_weights_from_logits(logits + 12345.0),
        )

    def test_shell_membership_matches_hand_constructed_union(self) -> None:
        inner, outer = gaussian_shell_radii(2, SHELL_C)
        sigma = 1.0
        midpoint = (inner + outer) / 2.0
        references = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        queries = np.asarray([[midpoint, 0.0], [30.0, 0.0]])
        membership = gaussian_shell_membership(queries, references, sigma)
        np.testing.assert_array_equal(membership, [True, False])
        self.assertEqual(gaussian_shell_coverage(queries, references, sigma), 0.5)

    def test_shell_membership_is_batching_invariant_and_deterministic(self) -> None:
        rng = np.random.default_rng(8)
        queries = rng.normal(size=(7, 5))
        references = rng.normal(size=(11, 5))
        first = gaussian_shell_membership(
            queries, references, 0.7, reference_batch_size=2
        )
        second = gaussian_shell_membership(
            queries, references, 0.7, reference_batch_size=11
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first,
            gaussian_shell_membership(queries, references, 0.7, reference_batch_size=2),
        )

    def test_shell_membership_matches_reference_loop(self) -> None:
        rng = np.random.default_rng(21)
        queries = rng.normal(size=(8, 6))
        references = rng.normal(size=(13, 6))
        sigma = 0.45
        inner, outer = gaussian_shell_radii(6, SHELL_C)
        expected = []
        for query in queries:
            distances = [np.linalg.norm(query - reference) for reference in references]
            expected.append(
                any(
                    sigma * inner <= distance <= sigma * outer for distance in distances
                )
            )
        np.testing.assert_array_equal(
            gaussian_shell_membership(queries, references, sigma),
            expected,
        )

    def test_expanded_distance_matches_direct_corruption(self) -> None:
        rng = np.random.default_rng(11)
        clean = rng.normal(size=(4, 7))
        references = rng.normal(size=(6, 7))
        noise = rng.normal(size=(4, 7))
        sigma = 0.37
        expanded = expanded_noisy_distances(clean, references, noise, sigma)
        direct = squared_distances(clean + sigma * noise, references)
        np.testing.assert_allclose(expanded, direct, rtol=2e-14, atol=2e-13)

    def test_committed_curves_have_frozen_grid_and_finite_values(self) -> None:
        result_dir = REPO_ROOT / "results" / "experiment_04a"
        for filename in ("coverage_curve.csv", "max_posterior_weight_curve.csv"):
            with (result_dir / filename).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(float(row["sigma"]) for row in rows), SIGMA_GRID)
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            values = np.asarray(
                [
                    [float(row[key]) for key in ("estimate", "ci95_low", "ci95_high")]
                    for row in rows
                ]
            )
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))

        manifest = json.loads((result_dir / "geometry_manifest.json").read_text())
        self.assertEqual(manifest["shell_c"], SHELL_C)
        self.assertEqual(manifest["sigma_grid"], list(SIGMA_GRID))

    def test_committed_high_high_summary_reproduces_manifest(self) -> None:
        result_dir = REPO_ROOT / "results" / "experiment_04a"
        with (result_dir / "coverage_curve.csv").open(newline="") as handle:
            coverage = list(csv.DictReader(handle))
        with (result_dir / "max_posterior_weight_curve.csv").open(newline="") as handle:
            posterior = list(csv.DictReader(handle))
        manifest = json.loads((result_dir / "geometry_manifest.json").read_text())
        threshold_c = manifest["paper_guided_primary_thresholds"]["C"]
        threshold_w = manifest["paper_guided_primary_thresholds"]["W"]
        high_high = [
            float(coverage_row["sigma"])
            for coverage_row, posterior_row in zip(coverage, posterior)
            if float(coverage_row["estimate"]) >= threshold_c
            and float(posterior_row["estimate"]) >= threshold_w
        ]
        self.assertEqual(high_high, manifest["sampled_primary_high_high_sigmas"])


if __name__ == "__main__":
    unittest.main()
