"""Contract tests for committed Experiment 5 calibration outputs."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


class Experiment05OutputContractTest(unittest.TestCase):
    """Keep E005 artifacts aligned with the frozen protocol."""

    def test_raw_score_schema_and_row_count_are_frozen(self) -> None:
        """Raw scores should use the future-compatible schema at every grid point."""
        scores_path = RESULTS_DIR / "experiment_05_scores.csv"
        with scores_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            rows = list(reader)

        self.assertEqual(
            tuple(reader.fieldnames or ()),
            (
                "experiment_id",
                "image_id",
                "split",
                "checkpoint",
                "trajectory",
                "axis_name",
                "axis_value",
                "cutoff",
                "seed",
                "S_low",
                "S_high",
            ),
        )
        self.assertEqual(len(rows), 6 * 3 * 3 * 1 * 101)
        self.assertEqual({row["experiment_id"] for row in rows}, {"experiment_05"})
        self.assertEqual({row["split"] for row in rows}, {"calibration"})
        self.assertEqual(
            {row["checkpoint"] for row in rows},
            {"not_applicable"},
        )
        self.assertTrue(
            all(
                0.0 <= float(row[score_name]) <= 1.0
                for row in rows
                for score_name in ("S_low", "S_high")
            )
        )

    def test_crossing_table_contains_every_image_control_and_cutoff(self) -> None:
        """Crossing analysis should contain one row per frozen combination."""
        crossings_path = RESULTS_DIR / "experiment_05_crossings.csv"
        with crossings_path.open(encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))

        self.assertEqual(len(rows), 6 * 3 * 3 * 1)
        self.assertEqual({float(row["cutoff"]) for row in rows}, {20.0, 40.0, 80.0})
        self.assertEqual(
            {row["trajectory"] for row in rows},
            {"low_band_first", "high_band_first", "together"},
        )

    def test_summary_records_frozen_configuration_and_failure_analysis(self) -> None:
        """The summary should identify settings and prohibit post hoc tuning."""
        summary_path = RESULTS_DIR / "experiment_05_summary.json"
        with summary_path.open(encoding="utf-8") as input_file:
            summary = json.load(input_file)

        configuration = summary["configuration"]
        self.assertEqual(configuration["num_images"], 6)
        self.assertEqual(configuration["construction_radius"], 40.0)
        self.assertEqual(configuration["evaluation_radii"], [20.0, 40.0, 80.0])
        self.assertEqual(configuration["recovery_threshold"], 0.8)
        self.assertEqual(configuration["recovery_seeds"], [0])
        self.assertEqual(configuration["bootstrap_unit"], "image")
        self.assertFalse(summary["failure_analysis"]["metric_tuning_performed"])


if __name__ == "__main__":
    unittest.main()
