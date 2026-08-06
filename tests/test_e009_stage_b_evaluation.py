"""Tests for the frozen E009 Stage B baseline-only evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))


def load_evaluation():
    """Load the Stage B evaluator from its tracked entry point."""
    path = EXPERIMENTS_ROOT / "09_stage_b_baseline_evaluation.py"
    spec = importlib.util.spec_from_file_location("e009_stage_b_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Stage B evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E009StageBBaselineTests(unittest.TestCase):
    """Protect the candidate cohort, seeds, evaluator, and stopping rule."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation = load_evaluation()
        cls.config = cls.evaluation.load_config()

    def test_exact_checkpoint_cohort_is_frozen(self) -> None:
        candidates = self.config["candidates"]
        self.assertEqual(len(candidates["edm_5k"]), 18)
        self.assertEqual(len(candidates["edm_1k"]), 6)
        self.assertEqual(
            [row["training_kimg"] for row in candidates["edm_5k"]],
            list(range(13000, 30001, 1000)),
        )
        self.assertEqual(
            [row["training_kimg"] for row in candidates["edm_1k"]],
            [2000, 4000, 6000, 8000, 10000, 12000],
        )
        hashes = [
            row["checkpoint_sha256"]
            for role in self.evaluation.ROLES
            for row in candidates[role]
        ]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertTrue(all(len(value) == 64 for value in hashes))

    def test_seed_and_eligibility_contract_is_unchanged(self) -> None:
        self.assertEqual(self.evaluation.PILOT_SEEDS, tuple(range(20000, 20128)))
        self.assertFalse(
            set(self.evaluation.PILOT_SEEDS).intersection(
                self.evaluation.CONFIRMATORY_SEEDS
            )
        )
        self.assertEqual(
            self.config["eligibility"]["count_interval_inclusive"], [13, 115]
        )
        self.assertEqual(self.config["memorization"]["criterion"], "d1nn < d2nn / 3")
        self.assertEqual(self.config["execution"]["expected_record_count"], 3072)

    def test_pair_selection_uses_rate_then_sha(self) -> None:
        new = [
            {
                "eligible": True,
                "memorization_rate": 0.25,
                "checkpoint_sha256": "b" * 64,
            },
            {
                "eligible": True,
                "memorization_rate": 0.25,
                "checkpoint_sha256": "a" * 64,
            },
        ]
        small = [
            {
                "eligible": True,
                "memorization_rate": 0.25,
                "checkpoint_sha256": "c" * 64,
            }
        ]
        pair = self.evaluation.select_pair(new, small)
        self.assertIsNotNone(pair)
        self.assertEqual(pair["edm_5k_checkpoint"]["checkpoint_sha256"], "a" * 64)

    def test_launchers_are_baseline_only_and_preserve_confirmatory_seeds(self) -> None:
        paths = (
            "scripts/e009_stage_b_baseline.slurm",
            "scripts/e009_stage_b_baseline_smoke.slurm",
            "scripts/e009_stage_b_baseline_pilot.slurm",
        )
        combined = "\n".join((REPO_ROOT / path).read_text() for path in paths)
        self.assertIn("pilot_seeds=20000..20127", combined)
        self.assertIn("e008_swaps=false", combined)
        self.assertIn("E009_REMOTE_BRANCH_COMMIT", combined)
        self.assertNotIn("0..255", combined)
        self.assertNotIn("--donor", combined)
        self.assertNotIn("--swap", combined)
        pilot = (REPO_ROOT / paths[-1]).read_text()
        self.assertIn("#SBATCH --array=0-1%2", pilot)

    def test_outcomes_include_preregistered_negative_stop(self) -> None:
        source = (
            REPO_ROOT / "experiments/09_stage_b_baseline_evaluation.py"
        ).read_text()
        self.assertIn("ELIGIBLE_5K_PAIR_FROZEN", source)
        self.assertIn("BLOCKED_NO_ELIGIBLE_5K_THROUGH_30K", source)
        self.assertIn('"automatic_extension_started": False', source)
        self.assertIn('"e008_executed": False', source)

    def test_protocol_records_completed_training_and_evaluation(self) -> None:
        protocol = json.loads(
            (REPO_ROOT / "configs/e009_stage_b_protocol.json").read_text()
        )
        scope = protocol["scientific_scope"]
        self.assertTrue(scope["full_stage_b_continuation_completed"])
        self.assertTrue(scope["baseline_evaluation_started"])
        self.assertTrue(scope["baseline_evaluation_completed"])
        execution = protocol["continuation_execution"]
        self.assertEqual(
            execution["execution_commit"], "a109a86c540dbd57be4d1f4110e47607e937bc65"
        )
        self.assertEqual(execution["slurm_job_id"], "15723871")
        self.assertEqual(execution["validation_status"], "pass")
        evaluation = protocol["evaluation_execution"]
        self.assertEqual(evaluation["observed_record_count"], 3072)
        self.assertEqual(evaluation["failed_record_count"], 0)
        self.assertEqual(evaluation["eligible_edm_5k_count"], 0)
        self.assertIsNone(evaluation["selected_pair"])
        self.assertEqual(evaluation["outcome"], "BLOCKED_NO_ELIGIBLE_5K_THROUGH_30K")
        self.assertFalse(evaluation["automatic_extension_started"])
        self.assertFalse(evaluation["e008_executed"])


if __name__ == "__main__":
    unittest.main()
