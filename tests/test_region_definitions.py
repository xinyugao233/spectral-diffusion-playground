"""Consistency gates for E004A, E005, E006, and proposed E007 regions."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.region_definitions import (
    contiguous_components,
    high_high_indices,
)


class RegionDefinitionMathTests(unittest.TestCase):
    """Verify classification uses evaluated points and preserves gaps."""

    def test_high_high_classification_does_not_fill_gaps(self) -> None:
        indices = high_high_indices(
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.9],
            q_coverage=0.8,
            q_posterior_weight=0.8,
        )
        self.assertEqual(indices, [0, 2])
        self.assertEqual(contiguous_components(indices), [[0], [2]])

    def test_nonfinite_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            high_high_indices(
                [0.9, float("nan")],
                [0.9, 0.9],
                q_coverage=0.8,
                q_posterior_weight=0.8,
            )


class RegionDefinitionRepositoryTests(unittest.TestCase):
    """Keep machine-readable definitions aligned with scientific wording."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(
            (REPO_ROOT / "results" / "region_definition_registry.json").read_text()
        )
        cls.manifest = json.loads(
            (
                REPO_ROOT
                / "results"
                / "experiment_04a"
                / "e006_grid_geometry_manifest.json"
            ).read_text()
        )
        with (REPO_ROOT / "results" / "experiment_04a" / "e006_grid_geometry.csv").open(
            newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.readme = (REPO_ROOT / "README.md").read_text()
        cls.e006_protocol = (
            REPO_ROOT / "docs" / "experiment_06_transition_swap_protocol.md"
        ).read_text()
        cls.e007_protocol = (
            REPO_ROOT / "docs" / "experiment_07_geometry_aligned_swap_protocol.md"
        ).read_text()

    def test_exact_e006_grid_schema_and_classification(self) -> None:
        expected_fields = {
            "sigma_index",
            "sigma",
            "coverage_estimate",
            "coverage_ci95_low",
            "coverage_ci95_high",
            "posterior_weight_estimate",
            "posterior_weight_ci95_low",
            "posterior_weight_ci95_high",
            "coverage_ge_q_c",
            "posterior_weight_ge_q_w",
            "high_high_point_estimate",
            "high_high_lower_bound",
        }
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(set(self.rows[0]), expected_fields)
        point = [
            int(row["sigma_index"])
            for row in self.rows
            if row["high_high_point_estimate"] == "True"
        ]
        lower = [
            int(row["sigma_index"])
            for row in self.rows
            if row["high_high_lower_bound"] == "True"
        ]
        self.assertEqual(point, [8, 9])
        self.assertEqual(lower, [8, 9])
        self.assertEqual(point, self.manifest["clean_room_geometry_high_high_indices"])
        self.assertEqual(
            lower,
            self.manifest["clean_room_geometry_high_high_lower_bound_indices"],
        )

    def test_registry_keeps_sources_distinct(self) -> None:
        paper = self.registry["paper_reported_medium_reference"]
        low = self.registry["e005_low_spectral_transition"]
        high = self.registry["e005_high_spectral_transition"]
        geometry = self.registry["e004a_clean_room_geometry_high_high"]
        self.assertEqual(paper["source"], "paper Table 1 / Figure 10")
        self.assertFalse(paper["defines_original_danger_zone"])
        self.assertFalse(low["defines_original_danger_zone"])
        self.assertFalse(high["defines_original_danger_zone"])
        self.assertEqual(geometry["indices"], [8, 9])
        self.assertIn("q_C=q_W=0.8", geometry["source"])
        self.assertFalse(geometry["defines_universal_paper_boundary"])

    def test_readme_preserves_inconclusive_historical_account(self) -> None:
        self.assertIn("formally `INCONCLUSIVE`", self.readme)
        self.assertIn("did not identify a\nmemorization danger zone", self.readme)
        self.assertNotIn("causal model-swap intervention", self.readme)
        self.assertIn("q_C=q_W=0.8", self.readme)

    def test_paper_reference_is_not_described_as_e004a_derived(self) -> None:
        required = (
            "E006 tested spectral-aligned windows and a literature-derived paper "
            "medium\nreference. It did not preregister a window from locally "
            "computed coverage and\nposterior-weight curves, because E004A did "
            "not yet exist."
        )
        self.assertIn(required, self.e006_protocol)

    def test_e006_formal_outcome_remains_inconclusive(self) -> None:
        outcome = json.loads(
            (
                REPO_ROOT / "results" / "experiment_06" / "experiment_06_outcome.json"
            ).read_text()
        )
        self.assertEqual(outcome["outcome"], "INCONCLUSIVE")

    def test_e007_is_proposed_and_unexecuted(self) -> None:
        self.assertIn("PROPOSED — NOT EXECUTED", self.e007_protocol)
        self.assertIn("has not been executed", self.e007_protocol)
        self.assertNotIn("Status: completed", self.e007_protocol)

    def test_alignment_figure_exists(self) -> None:
        path = (
            REPO_ROOT
            / "figures"
            / "experiment_04a"
            / "e006_grid_geometry_alignment.png"
        )
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
