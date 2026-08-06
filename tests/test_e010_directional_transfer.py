"""Focused tests for the frozen E010 directional transfer protocol."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from spectral_diffusion_playground.e010_directional_transfer import (
    EXPECTED_RECORDS,
    SAMPLE_SEEDS,
    bootstrap_mean_interval,
    formal_outcomes,
    frozen_conditions,
    nearest_two_cpu,
    target_control_summary,
    transition_category,
    validate_condition_registry,
)

ROOT = Path(__file__).resolve().parents[1]


class E010ProtocolTests(unittest.TestCase):
    """Protect preregistered E010 conditions, seeds, and analysis."""

    def test_condition_registry_is_complete_and_unique(self) -> None:
        conditions = frozen_conditions()
        self.assertEqual(len(conditions), 14)
        self.assertEqual(len({item.condition_id for item in conditions}), 14)
        validate_condition_registry(conditions)

    def test_directional_recipients_and_donors(self) -> None:
        for condition in frozen_conditions():
            if condition.direction == "suppression":
                self.assertEqual(condition.recipient, "edm_1k_012000")
                if condition.role != "baseline":
                    self.assertEqual(condition.donor, "edm_50k_040000")
            else:
                self.assertEqual(condition.recipient, "edm_50k_040000")
                if condition.role != "baseline":
                    self.assertEqual(condition.donor, "edm_1k_012000")

    def test_geometry_targets_and_controls_are_frozen(self) -> None:
        actual = {item.condition_id: item.swap_indices for item in frozen_conditions()}
        self.assertEqual(
            actual,
            {
                "A0": (),
                "A1": (7,),
                "A2": (8,),
                "A3": (9,),
                "A4": (7, 8),
                "A5": (9, 10),
                "A6": (11, 12),
                "B0": (),
                "B1": (7,),
                "B2": (8,),
                "B3": (9,),
                "B4": (7, 8),
                "B5": (9, 10),
                "B6": (11, 12),
            },
        )

    def test_whole_denoiser_returns_to_recipient(self) -> None:
        condition = next(
            item for item in frozen_conditions() if item.condition_id == "B5"
        )
        self.assertEqual(condition.model_for_step(8), condition.recipient)
        self.assertEqual(condition.model_for_step(9), condition.donor)
        self.assertEqual(condition.model_for_step(10), condition.donor)
        self.assertEqual(condition.model_for_step(11), condition.recipient)

    def test_seed_range_and_record_count(self) -> None:
        self.assertEqual(SAMPLE_SEEDS, tuple(range(40000, 40256)))
        self.assertEqual(len(frozen_conditions()) * len(SAMPLE_SEEDS), EXPECTED_RECORDS)

    def test_seed_range_does_not_overlap_prior_ranges(self) -> None:
        current = set(SAMPLE_SEEDS)
        for prior in (
            range(256),
            range(10000, 10128),
            range(20000, 20128),
            range(30000, 30128),
        ):
            self.assertFalse(current.intersection(prior))

    def test_nearest_two_cpu_uses_stable_reference_tie_break(self) -> None:
        references = np.asarray([[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]])
        d1, d2, first, second = nearest_two_cpu(np.asarray([0.0, 0.0]), references)
        self.assertEqual((d1, d2), (1.0, 1.0))
        self.assertEqual((first, second), (0, 1))

    def test_nearest_two_cpu_is_batching_independent_by_construction(self) -> None:
        rng = np.random.default_rng(4)
        references = rng.normal(size=(8, 2, 2))
        sample = rng.normal(size=(2, 2))
        self.assertEqual(
            nearest_two_cpu(sample, references),
            nearest_two_cpu(sample.copy(), references.copy()),
        )

    def test_transition_categories(self) -> None:
        self.assertEqual(transition_category(True, False), "memorized_to_non_memorized")
        self.assertEqual(transition_category(False, True), "non_memorized_to_memorized")
        self.assertEqual(transition_category(True, True), "memorized_to_memorized")
        self.assertEqual(
            transition_category(False, False), "non_memorized_to_non_memorized"
        )

    def test_suppression_contrast_calculation(self) -> None:
        baseline = np.ones(8, dtype=np.int8)
        before = np.asarray([0, 1, 1, 1, 1, 1, 1, 1])
        target = np.zeros(8, dtype=np.int8)
        after = np.asarray([1, 1, 1, 1, 1, 1, 1, 0])
        result = target_control_summary(
            baseline, before, target, after, direction="suppression", resamples=500
        )
        self.assertEqual(result["target_effect"], 1.0)
        self.assertEqual(result["before_effect"], 0.125)
        self.assertEqual(result["after_effect"], 0.125)
        self.assertEqual(result["contrast"], 0.875)
        self.assertTrue(result["criterion_pass"])

    def test_induction_contrast_calculation(self) -> None:
        baseline = np.zeros(8, dtype=np.int8)
        before = np.zeros(8, dtype=np.int8)
        target = np.ones(8, dtype=np.int8)
        after = np.zeros(8, dtype=np.int8)
        result = target_control_summary(
            baseline, before, target, after, direction="induction", resamples=500
        )
        self.assertEqual(result["contrast"], 1.0)
        self.assertTrue(result["criterion_pass"])

    def test_bootstrap_is_reproducible(self) -> None:
        values = np.asarray([0.0, 1.0, 1.0, -1.0])
        self.assertEqual(
            bootstrap_mean_interval(values, seed=0, resamples=1000),
            bootstrap_mean_interval(values, seed=0, resamples=1000),
        )

    def test_formal_outcomes_preserve_multiple_results(self) -> None:
        outcomes = formal_outcomes(
            {
                ("suppression", "low"): True,
                ("suppression", "high"): False,
                ("induction", "low"): True,
                ("induction", "high"): False,
            }
        )
        self.assertIn("LOW_DERIVED_SUPPRESSION_SUPPORTED", outcomes)
        self.assertIn("LOW_DERIVED_INDUCTION_SUPPORTED", outcomes)
        self.assertIn("MIXED_DIRECTIONAL_EVIDENCE", outcomes)

    def test_committed_manifests_match_config_hashes(self) -> None:
        config = json.loads(
            (ROOT / "configs/e010_directional_memorization_transfer.json").read_text()
        )
        for path_key, hash_key in (
            ("model_pair_manifest", "model_pair_manifest_sha256"),
            ("condition_manifest", "condition_manifest_sha256"),
            ("seed_manifest", "seed_manifest_sha256"),
            ("geometry_target", "geometry_target_sha256"),
        ):
            path = ROOT / config["inputs"][path_key]
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, config["inputs"][hash_key])

    def test_model_selection_and_historical_evidence_are_frozen(self) -> None:
        pair = json.loads((ROOT / "data/e010_model_pair_manifest.json").read_text())
        memorizing = pair["memorizing_model"]
        self.assertEqual(memorizing["training_kimg"], 12000)
        self.assertEqual(memorizing["historical_baseline"]["memorized_count"], 113)
        self.assertEqual(
            pair["generalizing_model"]["historical_baselines"][0]["memorized_count"], 0
        )
        self.assertFalse(pair["baseline_matched"])

    def test_e008_remains_blocked_and_unexecuted(self) -> None:
        protocol = (
            ROOT / "docs/experiment_08_frequency_geometry_swap_protocol.md"
        ).read_text()
        self.assertIn("BLOCKED", protocol)
        config = json.loads(
            (ROOT / "configs/e010_directional_memorization_transfer.json").read_text()
        )
        self.assertEqual(config["experiment_id"], "E010")
        self.assertFalse(
            any("train" in path.name for path in (ROOT / "scripts").glob("e010*"))
        )


if __name__ == "__main__":
    unittest.main()
