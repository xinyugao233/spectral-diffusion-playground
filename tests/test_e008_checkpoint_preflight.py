"""Tests for the E008 no-swap checkpoint baseline preflight."""

from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import io
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.e008_checkpoint_preflight import (
    CONFIRMATORY_SEEDS,
    ELIGIBLE_COUNT_MAX,
    ELIGIBLE_COUNT_MIN,
    PILOT_SEEDS,
    candidate_is_eligible,
    discover_checkpoint_paths,
    independent_seed_latents,
    merge_resume_rows,
    parse_training_kimg,
    select_model_pair,
    sha256_file,
    validate_no_swap_only,
)


def load_entrypoint():
    """Load the numeric experiment entrypoint for parser and schema tests."""
    path = REPO_ROOT / "experiments" / "08_checkpoint_baseline_preflight.py"
    experiments_root = str(path.parent)
    if experiments_root not in sys.path:
        sys.path.insert(0, experiments_root)
    spec = importlib.util.spec_from_file_location("experiment_08_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load E008 preflight entrypoint")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointInventoryTests(unittest.TestCase):
    """Verify complete deterministic checkpoint discovery and hashing."""

    def test_training_kimg_parser_is_exact(self) -> None:
        self.assertEqual(parse_training_kimg("network-snapshot-002000.pkl"), 2000)
        for malformed in (
            "network-snapshot-2000.pkl",
            "network-snapshot-002000.pt",
            "snapshot-002000.pkl",
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_training_kimg(malformed)

    def test_discovery_keeps_malformed_candidates_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "network-snapshot-000000.pkl").write_bytes(b"zero")
            (root / "network-snapshot-002000.pkl").write_bytes(b"two")
            (root / "network-snapshot-bad.pkl").write_bytes(b"bad")
            (root / "notes.txt").write_text("ignored")
            accepted, rejected = discover_checkpoint_paths(root)
            self.assertEqual(
                [path.name for path in accepted],
                ["network-snapshot-000000.pkl", "network-snapshot-002000.pkl"],
            )
            self.assertEqual(
                [path.name for path in rejected], ["network-snapshot-bad.pkl"]
            )

    def test_sha256_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pkl"
            path.write_bytes(b"frozen checkpoint")
            expected = hashlib.sha256(b"frozen checkpoint").hexdigest()
            self.assertEqual(sha256_file(path, chunk_size=3), expected)
            self.assertEqual(sha256_file(path, chunk_size=11), expected)


class FrozenPilotRuleTests(unittest.TestCase):
    """Lock the prospective seeds, eligibility, and pair-selection rules."""

    def test_seed_sets_are_exact_and_disjoint(self) -> None:
        self.assertEqual(PILOT_SEEDS, tuple(range(10000, 10128)))
        self.assertEqual(CONFIRMATORY_SEEDS, tuple(range(256)))
        self.assertFalse(set(PILOT_SEEDS).intersection(CONFIRMATORY_SEEDS))

    def test_seeded_latents_are_order_and_batch_invariant(self) -> None:
        first = independent_seed_latents(PILOT_SEEDS[:4], (3, 4, 4))
        second = independent_seed_latents(reversed(PILOT_SEEDS[:4]), (3, 4, 4))
        for seed in PILOT_SEEDS[:4]:
            np.testing.assert_array_equal(first[seed], second[seed])

    def test_eligibility_is_exactly_13_through_115(self) -> None:
        self.assertEqual((ELIGIBLE_COUNT_MIN, ELIGIBLE_COUNT_MAX), (13, 115))
        self.assertFalse(candidate_is_eligible(12, 128))
        self.assertTrue(candidate_is_eligible(13, 128))
        self.assertTrue(candidate_is_eligible(115, 128))
        self.assertFalse(candidate_is_eligible(116, 128))
        with self.assertRaises(ValueError):
            candidate_is_eligible(13, 127)

    def test_pair_selection_minimizes_rate_difference(self) -> None:
        rows = [
            summary("edm_1k", "b", 0.40),
            summary("edm_1k", "a", 0.60),
            summary("edm_50k", "y", 0.43),
            summary("edm_50k", "z", 0.80),
        ]
        selected = select_model_pair(rows)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["edm_1k"]["checkpoint_sha256"], "b")
        self.assertEqual(selected["edm_50k"]["checkpoint_sha256"], "y")
        self.assertAlmostEqual(selected["absolute_pilot_rate_difference"], 0.03)

    def test_pair_selection_uses_sha_tie_break(self) -> None:
        rows = [
            summary("edm_1k", "b", 0.5),
            summary("edm_1k", "a", 0.5),
            summary("edm_50k", "d", 0.5),
            summary("edm_50k", "c", 0.5),
        ]
        selected = select_model_pair(rows)
        assert selected is not None
        self.assertEqual(selected["edm_1k"]["checkpoint_sha256"], "a")
        self.assertEqual(selected["edm_50k"]["checkpoint_sha256"], "c")

    def test_no_pair_is_selected_when_one_role_is_ineligible(self) -> None:
        rows = [summary("edm_1k", "a", 0.5)]
        self.assertIsNone(select_model_pair(rows))

    def test_no_swap_guard_rejects_donor_and_window(self) -> None:
        validate_no_swap_only()
        with self.assertRaisesRegex(ValueError, "donor"):
            validate_no_swap_only(donor_checkpoint="donor.pkl")
        with self.assertRaisesRegex(ValueError, "window"):
            validate_no_swap_only(swap_window="8..8")


class ResumeAndRepositoryTests(unittest.TestCase):
    """Protect resume records and prior frozen experiment artifacts."""

    def test_resume_keeps_failed_rows_and_rejects_conflicts(self) -> None:
        failed = pilot_row("a", 10000, status="failed")
        complete = pilot_row("a", 10001, status="ok")
        merged = merge_resume_rows([failed], [complete, dict(failed)])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["status"], "failed")
        conflicting = dict(failed)
        conflicting["error"] = "different"
        with self.assertRaises(ValueError):
            merge_resume_rows([failed], [conflicting])

    def test_entrypoint_has_no_donor_or_window_arguments(self) -> None:
        module = load_entrypoint()
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        parse_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        )
        strings = {
            node.value
            for node in ast.walk(parse_node)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn("--donor-checkpoint", strings)
        self.assertNotIn("--swap-window", strings)

    def test_help_is_available_outside_slurm(self) -> None:
        module = load_entrypoint()
        stdout = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [str(module.__file__), "--help"]),
            mock.patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(stdout),
            self.assertRaises(SystemExit) as raised,
        ):
            module.main()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--inventory-only", stdout.getvalue())

    def test_per_sample_schema_contains_no_swap_fields(self) -> None:
        module = load_entrypoint()
        header = module.per_sample_header()
        self.assertNotIn("donor_model", header)
        self.assertNotIn("window_name", header)
        self.assertIn("d1nn_over_d2nn", header)

    def test_config_freezes_baseline_only_scope(self) -> None:
        config = json.loads(
            (REPO_ROOT / "configs/e008_checkpoint_preflight.json").read_text()
        )
        self.assertTrue(config["scientific_scope"]["baseline_only"])
        self.assertFalse(config["scientific_scope"]["donor_models_allowed"])
        self.assertFalse(config["scientific_scope"]["swap_windows_allowed"])
        self.assertEqual(config["eligibility"]["count_interval_inclusive"], [13, 115])

    def test_prior_hash_guards_remain_active(self) -> None:
        region_test = (REPO_ROOT / "tests/test_region_definitions.py").read_text()
        frequency_test = (
            REPO_ROOT / "tests/test_frequency_restricted_geometry.py"
        ).read_text()
        self.assertIn("FROZEN_E005_E006_ARTIFACTS", region_test)
        self.assertIn("FROZEN_PRIOR_ARTIFACTS", frequency_test)

    def test_e008_remains_unexecuted_in_canonical_pipeline(self) -> None:
        pipeline = json.loads(
            (REPO_ROOT / "results/canonical_experiment_pipeline.json").read_text()
        )
        self.assertEqual(
            pipeline["stage_7_frequency_geometry_swap"]["status"],
            "proposed_not_executed",
        )

    def test_zero_confirmatory_overlap_passes_output_validation(self) -> None:
        module = load_entrypoint()
        config = module.load_and_validate_config(module.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            inventory_path = output_dir / module.INVENTORY_FILENAME
            candidate = {field: "" for field in module.inventory_header()}
            candidate.update(
                {
                    "model_role": "edm_1k",
                    "checkpoint_sha256": "a" * 64,
                    "training_kimg": 0,
                    "inventory_status": "accepted",
                }
            )
            module.write_csv(inventory_path, [candidate], module.inventory_header())
            module.dump_json(
                output_dir / module.POOL_MANIFEST_FILENAME,
                {
                    "pool_frozen_before_pilot": True,
                    "pilot_started": False,
                    "inventory": {"sha256": sha256_file(inventory_path)},
                    "config_sha256": sha256_file(module.DEFAULT_CONFIG),
                    "scientific_scope": config["scientific_scope"],
                    "pilot_seeds": list(PILOT_SEEDS),
                },
            )
            rows = []
            for seed in PILOT_SEEDS:
                row = {field: "" for field in module.per_sample_header()}
                row.update(
                    {
                        "model_role": "edm_1k",
                        "checkpoint_sha256": "a" * 64,
                        "training_kimg": 0,
                        "sample_seed": seed,
                        "memorized": 0,
                        "status": "ok",
                    }
                )
                rows.append(row)
            module.write_csv(
                output_dir / module.PER_SAMPLE_FILENAME,
                rows,
                module.per_sample_header(),
            )
            validation = module.validate_outputs(
                output_dir, config, require_complete=True
            )
        self.assertEqual(validation["status"], "pass")
        self.assertTrue(validation["checks"]["confirmatory_seed_overlap_absent"])


def summary(role: str, digest: str, rate: float) -> dict[str, object]:
    """Build one synthetic eligible checkpoint summary."""
    return {
        "model_role": role,
        "checkpoint_sha256": digest,
        "memorization_rate": rate,
        "eligible": True,
    }


def pilot_row(digest: str, seed: int, *, status: str) -> dict[str, object]:
    """Build one minimal stable resume row."""
    return {
        "model_role": "edm_1k",
        "checkpoint_sha256": digest,
        "training_kimg": 0,
        "sample_seed": seed,
        "status": status,
        "error": "failure" if status != "ok" else "",
    }


if __name__ == "__main__":
    unittest.main()
