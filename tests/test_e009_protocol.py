"""Tests for the frozen E009 staged training protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.e009_subsets import load_indices


def load_preflight():
    """Load the E009 preflight module from its tracked script."""
    path = REPO_ROOT / "scripts/e009_preflight.py"
    spec = importlib.util.spec_from_file_location("e009_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load E009 preflight")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_smoke_validator():
    """Load the E009 smoke-checkpoint validator from its tracked script."""
    path = REPO_ROOT / "scripts/e009_validate_smoke_checkpoint.py"
    spec = importlib.util.spec_from_file_location("e009_smoke_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load E009 smoke validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E009ProtocolTests(unittest.TestCase):
    """Protect inputs, training parity, seeds, budgets, and stop rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = load_preflight()
        cls.smoke_validator = load_smoke_validator()
        cls.protocol = json.loads(
            (REPO_ROOT / "configs/e009_stage_a_protocol.json").read_text()
        )
        cls.manifest = json.loads(
            (REPO_ROOT / "data/e009_nested_subsets_manifest.json").read_text()
        )

    def test_nested_manifests_are_exact_and_preserve_anchor(self) -> None:
        anchor = set(
            load_indices(REPO_ROOT / "data/e005_cifar10_subset_1k_indices.txt")
        )
        previous = anchor
        for size in (2000, 5000, 10000):
            record = self.manifest["subsets"][str(size)]
            path = REPO_ROOT / "data" / record["index_manifest"]
            values = load_indices(path)
            self.assertEqual(len(values), size)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["index_manifest_sha256"],
            )
            self.assertEqual(set(record["class_distribution"].values()), {size // 10})
            self.assertTrue(previous.issubset(set(values)))
            previous = set(values)

    def test_stage_a_grid_and_training_seeds_are_frozen(self) -> None:
        stage = self.protocol["stage_a"]
        self.assertEqual(stage["dataset_sizes"], [2000, 5000, 10000])
        self.assertEqual(stage["duration_kimg"], 12000)
        self.assertEqual(stage["checkpoint_kimg"], list(range(0, 12001, 1000)))
        self.assertEqual(
            stage["training_seed_by_size"], {"2000": 0, "5000": 0, "10000": 0}
        )
        self.assertEqual(stage["parallel_array"], "0-2%3")

    def test_pilot_seeds_and_eligibility_are_exact(self) -> None:
        pilot = self.protocol["pilot"]
        self.assertEqual(
            pilot["seeds"], {"start": 20000, "stop_inclusive": 20127, "count": 128}
        )
        self.assertEqual(pilot["excluded_seed_ranges"], [[0, 255], [10000, 10127]])
        self.assertEqual(pilot["expected_checkpoint_count"], 39)
        self.assertEqual(pilot["expected_record_count"], 4992)
        self.assertEqual(pilot["eligible_count_interval_inclusive"], [13, 115])

    def test_pair_rule_prefers_rate_then_size_then_hash(self) -> None:
        selection = self.protocol["pair_selection"]
        self.assertEqual(selection["candidate_large_role_minimum_dataset_size"], 5000)
        self.assertEqual(
            selection["primary"],
            "minimum absolute pilot memorization-rate difference",
        )
        self.assertEqual(
            selection["tie_breaks"],
            [
                "prefer larger new dataset size",
                "lexicographic (new_checkpoint_sha256, edm_1k_checkpoint_sha256)",
            ],
        )

    def test_stage_b_is_conditional_and_requires_review(self) -> None:
        decision = self.protocol["stage_a_decision"]
        stage_b = self.protocol["stage_b_if_triggered"]
        self.assertIn("trigger Stage B", decision["only_2k_eligible"])
        self.assertEqual(decision["no_new_eligible_checkpoint"], "trigger Stage B")
        self.assertFalse(stage_b["automatic_submission"])
        self.assertTrue(stage_b["requires_separate_review"])
        self.assertEqual(stage_b["add_dataset_size"], 20000)
        self.assertEqual(stage_b["extend_one_stage_a_size_to_kimg"], 20000)

    def test_configs_match_frozen_semantics_and_hashes(self) -> None:
        protocol, manifest = self.preflight.validate_protocol(REPO_ROOT)
        for relative_path in list(self.protocol["stage_a"]["configs"].values()) + [
            "configs/e009_smoke_edm2k_1kimg.yaml"
        ]:
            with self.subTest(config=relative_path):
                path = REPO_ROOT / relative_path
                config = self.preflight.validate_config(
                    REPO_ROOT, path, protocol, manifest
                )
                training = config["training"]
                for key, expected in self.preflight.EXPECTED_TRAINING_BASE.items():
                    self.assertEqual(training[key], expected)

    def test_launchers_preserve_slurm_and_no_swap_boundaries(self) -> None:
        array = (REPO_ROOT / "scripts/e009_train_stage_a.slurm").read_text()
        smoke = (REPO_ROOT / "scripts/e009_train_smoke.slurm").read_text()
        entrypoint = (REPO_ROOT / "scripts/e009_training_entrypoint.sh").read_text()
        self.assertIn("#SBATCH --array=0-2%3", array)
        self.assertIn("#SBATCH --gres=gpu:L40S:1", array)
        self.assertIn("#SBATCH --time=1-00:00:00", array)
        self.assertIn("#SBATCH --time=00:30:00", smoke)
        self.assertIn(
            'bash "${E009_REPO_ROOT}/scripts/e009_training_entrypoint.sh"', smoke
        )
        self.assertIn(
            'exec bash "${E009_REPO_ROOT}/scripts/e009_training_entrypoint.sh"',
            array,
        )
        self.assertIn("SLURM_JOB_ID is required", entrypoint)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", entrypoint)
        self.assertIn("e009_validate_smoke_checkpoint.py", smoke)
        combined = array + smoke + entrypoint
        for forbidden in ("donor", "swap_window", "confirmatory"):
            self.assertNotIn(forbidden, combined)

    def test_smoke_snapshot_selection_is_highest_kimg(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "network-snapshot-000000.pkl",
                "network-snapshot-000001.pkl",
            ):
                (root / name).write_bytes(b"snapshot")
            selected = self.smoke_validator.find_final_snapshot([root])
            self.assertEqual(selected.name, "network-snapshot-000001.pkl")

    def test_smoke_ema_requires_unconditional_finite_state(self) -> None:
        import torch

        model = torch.nn.Linear(2, 2)
        model.label_dim = 0
        result = self.smoke_validator.validate_ema(model)
        self.assertEqual(result["label_dim"], 0)
        model.label_dim = 10
        with self.assertRaisesRegex(ValueError, "not unconditional"):
            self.smoke_validator.validate_ema(model)

    def test_e008_remains_blocked_and_unexecuted(self) -> None:
        outcome = json.loads(
            (
                REPO_ROOT / "results/experiment_08_preflight/preflight_outcome.json"
            ).read_text()
        )
        self.assertEqual(outcome["outcome"], "BLOCKED_NO_ELIGIBLE_PAIR")
        self.assertFalse(outcome["e008_executed"])

    def test_stage_a_evaluation_is_pre_staged_without_swaps(self) -> None:
        config = json.loads(
            (REPO_ROOT / "configs/e009_stage_a_evaluation.json").read_text()
        )
        self.assertEqual(
            config["pilot_seeds"],
            {"start": 20000, "stop_inclusive": 20127, "count": 128},
        )
        self.assertEqual(config["expected_training_kimg"], list(range(0, 12001, 1000)))
        self.assertEqual(config["eligibility"]["count_interval_inclusive"], [13, 115])
        self.assertFalse(config["scientific_scope"]["swap_windows_allowed"])
        launcher = (REPO_ROOT / "scripts/e009_stage_a_evaluation.slurm").read_text()
        array = (REPO_ROOT / "scripts/e009_stage_a_pilot_array.slurm").read_text()
        self.assertIn("pilot_seeds=20000..20127", launcher)
        self.assertIn("swap_execution_available=false", launcher)
        self.assertIn("#SBATCH --array=0-2%3", array)
        self.assertIn("pilot_seeds=20000..20127", array)
        self.assertIn("swap_execution_available=false", array)

    def test_stage_a_pair_rule_prefers_larger_dataset_on_rate_tie(self) -> None:
        path = REPO_ROOT / "experiments/09_stage_a_baseline_evaluation.py"
        experiments_root = str(path.parent)
        if experiments_root not in sys.path:
            sys.path.insert(0, experiments_root)
        spec = importlib.util.spec_from_file_location("e009_eval", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        new_rows = [
            {
                "model_role": "edm_5k",
                "eligible": True,
                "memorization_rate": 0.5,
                "checkpoint_sha256": "a",
            },
            {
                "model_role": "edm_10k",
                "eligible": True,
                "memorization_rate": 0.5,
                "checkpoint_sha256": "b",
            },
        ]
        small_rows = [
            {
                "model_role": "edm_1k",
                "eligible": True,
                "memorization_rate": 0.5,
                "checkpoint_sha256": "c",
            }
        ]
        selected = module.select_pair(new_rows, small_rows)
        assert selected is not None
        self.assertEqual(selected["larger_data_checkpoint"]["model_role"], "edm_10k")


if __name__ == "__main__":
    unittest.main()
