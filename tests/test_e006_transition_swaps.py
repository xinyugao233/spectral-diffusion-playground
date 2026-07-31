"""Focused tests for the frozen E006 transition-window swap protocol."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.e006_transition_swaps import (
    EFFECT_THRESHOLD,
    SAMPLE_SEEDS,
    SIGMA_GRID,
    WINDOW_BY_NAME,
    classify_outcome,
    clopper_pearson_interval,
    deterministic_qualitative_selection,
    euler_update,
    frozen_conditions,
    generated_sample_hash,
    memorization_flags,
    nearest_two,
    paired_effect_summary,
    pure_euler_sample_numpy,
    transition_influence,
)


class CountingModel:
    """Small deterministic denoiser double for exact call bookkeeping."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[float] = []

    def __call__(self, state: np.ndarray, sigma: float) -> np.ndarray:
        self.calls.append(sigma)
        return np.full_like(state, self.value)


class E006TransitionSwapTest(unittest.TestCase):
    """Validate frozen E006 sampling, metric, statistics, and outcome logic."""

    def test_sigma_grid_and_seed_set_are_frozen(self) -> None:
        self.assertEqual(len(SIGMA_GRID), 18)
        self.assertEqual(SIGMA_GRID[0], 80.0)
        self.assertAlmostEqual(SIGMA_GRID[-1], 0.002)
        self.assertEqual(SAMPLE_SEEDS, tuple(range(256)))

    def test_windows_use_exact_inclusive_boundaries(self) -> None:
        expected = {
            "low_transition": (5, 11),
            "high_transition": (11, 14),
            "combined_transition": (5, 14),
            "paper_medium_reference": (6, 13),
            "low_pre_control": (0, 6),
            "low_post_control": (11, 17),
            "high_pre_control": (7, 10),
            "high_post_control": (14, 17),
        }
        self.assertEqual(
            {
                name: (window.start_index, window.end_index)
                for name, window in WINDOW_BY_NAME.items()
            },
            expected,
        )

    def test_frozen_condition_table_has_two_baselines_and_sixteen_swaps(self) -> None:
        conditions = frozen_conditions()
        self.assertEqual(len(conditions), 18)
        self.assertEqual(conditions[0].name, "edm_1k_no_swap")
        self.assertEqual(conditions[1].name, "edm_50k_no_swap")

    def test_base_and_donor_switch_only_inside_window(self) -> None:
        condition = next(
            item
            for item in frozen_conditions()
            if item.name
            == "edm_1k_base__edm_50k_donor__high_transition"
        )
        self.assertEqual(condition.model_for_step(10), "edm_1k")
        self.assertEqual(condition.model_for_step(11), "edm_50k")
        self.assertEqual(condition.model_for_step(14), "edm_50k")
        self.assertEqual(condition.model_for_step(15), "edm_1k")

    def test_euler_update_matches_closed_form(self) -> None:
        state = np.asarray([2.0, 4.0])
        denoised = np.asarray([1.0, 3.0])
        actual = euler_update(state, denoised, sigma=2.0, sigma_next=1.0)
        np.testing.assert_array_equal(actual, np.asarray([1.5, 3.5]))

    def test_sampler_makes_exactly_one_call_per_step_and_no_heun_call(self) -> None:
        base = CountingModel(0.0)
        donor = CountingModel(1.0)
        condition = next(
            item
            for item in frozen_conditions()
            if item.name
            == "edm_1k_base__edm_50k_donor__high_transition"
        )
        output = pure_euler_sample_numpy(
            np.ones((1, 1, 1)),
            condition,
            {"edm_1k": base, "edm_50k": donor},
        )
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(len(base.calls), 14)
        self.assertEqual(len(donor.calls), 4)
        self.assertEqual(len(base.calls) + len(donor.calls), 18)

    def test_deterministic_hash_is_layout_and_dtype_canonical(self) -> None:
        sample = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
        copied = np.asarray(sample, dtype="<f8", order="C")
        expected = hashlib.sha256(copied.tobytes(order="C")).hexdigest()
        self.assertEqual(generated_sample_hash(sample), expected)
        self.assertEqual(generated_sample_hash(sample.copy()), expected)

    def test_nearest_neighbor_metric_uses_flat_euclidean_distance(self) -> None:
        reference = np.asarray([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
        query = np.asarray([[2.9, 4.1]])
        indices, distances = nearest_two(query, reference)
        np.testing.assert_array_equal(indices, np.asarray([[1, 0]]))
        self.assertAlmostEqual(distances[0, 0], np.sqrt(0.02))
        self.assertAlmostEqual(distances[0, 1], np.sqrt(25.22))

    def test_memorization_inequality_is_strict(self) -> None:
        distances = np.asarray([[1.0, 3.1], [1.0, 3.0], [0.0, 1.0]])
        np.testing.assert_array_equal(
            memorization_flags(distances),
            np.asarray([True, False, True]),
        )

    def test_clopper_pearson_matches_known_values(self) -> None:
        low, high = clopper_pearson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.3084971078, places=8)
        low, high = clopper_pearson_interval(5, 10)
        self.assertAlmostEqual(low, 0.1870860284, places=8)
        self.assertAlmostEqual(high, 0.8129139716, places=8)

    def test_paired_comparison_records_discordant_counts(self) -> None:
        baseline = np.asarray([0, 0, 1, 1], dtype=np.int8)
        swapped = np.asarray([1, 0, 0, 1], dtype=np.int8)
        summary = paired_effect_summary(
            baseline, swapped, seed=1, resamples=200
        )
        self.assertEqual(summary["discordant_positive"], 1)
        self.assertEqual(summary["discordant_negative"], 1)
        self.assertEqual(summary["discordant_zero"], 2)
        self.assertEqual(summary["paired_mean_delta"], 0.0)

    def test_transition_threshold_requires_both_controls_and_uncertainty(self) -> None:
        baseline = np.zeros(256, dtype=np.int8)
        transition = baseline.copy()
        transition[:96] = 1
        pre = baseline.copy()
        pre[:16] = 1
        post = baseline.copy()
        post[:12] = 1
        result = transition_influence(
            baseline,
            transition,
            pre,
            post,
            seed=10,
            resamples=500,
        )
        self.assertGreaterEqual(
            result["transition_effect_magnitude"], EFFECT_THRESHOLD
        )
        self.assertTrue(result["passes_point_threshold"])
        self.assertTrue(result["uncertainty_support"])
        self.assertTrue(result["influential"])

    def test_outcome_classification_all_labels(self) -> None:
        keys = [
            ("edm_1k_to_edm_50k", "low_transition"),
            ("edm_1k_to_edm_50k", "high_transition"),
            ("edm_50k_to_edm_1k", "low_transition"),
            ("edm_50k_to_edm_1k", "high_transition"),
        ]
        self.assertEqual(classify_outcome(dict.fromkeys(keys, True)), "YES")
        self.assertEqual(classify_outcome(dict.fromkeys(keys, False)), "NO")
        one_direction = dict.fromkeys(keys, False)
        one_direction[keys[0]] = True
        one_direction[keys[1]] = True
        self.assertEqual(classify_outcome(one_direction), "PARTIAL")
        diagonal = dict.fromkeys(keys, False)
        diagonal[keys[0]] = True
        diagonal[keys[3]] = True
        self.assertEqual(classify_outcome(diagonal), "MIXED")
        self.assertEqual(
            classify_outcome(dict.fromkeys(keys, True), invalid=True),
            "INCONCLUSIVE",
        )

    def test_qualitative_selection_uses_first_two_ascending_seeds(self) -> None:
        baseline = [False, False, True, True, False, False]
        swapped = [True, True, False, True, False, False]
        selected = deterministic_qualitative_selection(
            baseline, swapped, list(range(6))
        )
        self.assertEqual(selected["newly_memorized"], [0, 1])
        self.assertEqual(selected["no_longer_memorized"], [2])
        self.assertEqual(selected["unchanged_memorized"], [3])
        self.assertEqual(selected["unchanged_non_memorized"], [4, 5])

    def test_frozen_config_and_subset_hashes(self) -> None:
        config_path = REPO_ROOT / "configs/e006_transition_window_swaps.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["statistics"]["effect_threshold"], 0.10)
        self.assertFalse(config["sampler"]["heun_correction"])
        subset_path = REPO_ROOT / "data/e005_cifar10_subset_1k_indices.txt"
        self.assertEqual(
            hashlib.sha256(subset_path.read_bytes()).hexdigest(),
            "33bb509c48144464a48d3b945cc44c14f880a1e6c6470c283dc0ed65e22b1f29",
        )

    def test_launcher_has_explicit_root_and_collision_guards(self) -> None:
        launcher = (
            REPO_ROOT / "scripts/e006_eval_transition_swaps.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("E006_REPO_ROOT", launcher)
        self.assertIn("E006_REPO_COMMIT", launcher)
        self.assertNotIn("BASH_SOURCE", launcher)
        self.assertIn('if [ -n "$OUTPUT_DIR" ] && [ -e "$OUTPUT_DIR" ]', launcher)
        self.assertIn("--preflight-only", launcher)


if __name__ == "__main__":
    unittest.main()
