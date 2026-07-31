"""Tests for the frozen Experiment 4 measurement and review contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.filters import (
    create_frequency_mask,
    high_pass_filter,
    low_pass_filter,
)
from spectral_diffusion_playground.frequency_cutoff import (
    CANDIDATE_CUTOFFS,
    MANIFEST_CSV_FIELDS,
    MEASUREMENT_CSV_FIELDS,
    REVIEW_CSV_FIELDS,
    RELATIVE_ENERGY_TOLERANCE,
    RECONSTRUCTION_TOLERANCE,
    analyze_frequency_cutoffs,
    apply_frozen_cutoff_rubric,
    build_blank_reviewer_rows,
    preprocess_cifar10_image,
    reviewer_template_is_blank,
    select_balanced_indices,
    validate_completed_reviewer_rows,
)
from spectral_diffusion_playground.fft import compute_fft, shift_fft

EXPECTED_CANONICAL_INDICES = (
    3,
    10,
    6,
    9,
    25,
    35,
    0,
    8,
    22,
    26,
    12,
    16,
    4,
    5,
    13,
    17,
    1,
    2,
    11,
    14,
)


def build_image_records() -> list[dict[str, object]]:
    """Build the frozen two-images-per-class metadata shape for unit tests."""
    records: list[dict[str, object]] = []
    for class_id in range(10):
        for occurrence in range(2):
            dataset_index = class_id * 2 + occurrence
            records.append(
                {
                    "image_id": f"image-{dataset_index:02d}",
                    "dataset_index": dataset_index,
                    "class_id": class_id,
                    "class_name": f"class-{class_id}",
                }
            )
    return records


def complete_rows(
    reviewer_id: str,
    *,
    score: int,
) -> list[dict[str, object]]:
    """Create synthetically completed reviewer rows with one uniform score."""
    rows = build_blank_reviewer_rows(build_image_records(), reviewer_id)
    for row in rows:
        row["layout_score"] = score
        row["identity_score"] = score
        row["high_localization_score"] = score
        row["ambiguous"] = "false"
        row["failure_category"] = "" if score == 2 else "synthetic_failure"
        row["comment"] = "" if score == 2 else "Synthetic low-score case."
    return rows


class FrequencyCutoffMeasurementTest(unittest.TestCase):
    """Verify the frozen image selection and numerical decomposition."""

    def test_canonical_selection_rule_returns_frozen_indices(self) -> None:
        labels = [-1] * 36
        class_positions = {
            0: (3, 10),
            1: (6, 9),
            2: (25, 35),
            3: (0, 8),
            4: (22, 26),
            5: (12, 16),
            6: (4, 5),
            7: (13, 17),
            8: (1, 2),
            9: (11, 14),
        }
        for class_id, positions in class_positions.items():
            for position in positions:
                labels[position] = class_id

        self.assertEqual(select_balanced_indices(labels), EXPECTED_CANONICAL_INDICES)

    def test_cifar_preprocessing_is_float64_in_minus_one_to_one(self) -> None:
        raw = np.zeros((32, 32, 3), dtype=np.uint8)
        raw[0, 0] = 255

        image = preprocess_cifar10_image(raw)

        self.assertEqual(image.dtype, np.float64)
        self.assertEqual(float(image.min()), -1.0)
        self.assertEqual(float(image.max()), 1.0)

    def test_masks_are_exact_complements_for_all_candidates(self) -> None:
        rng = np.random.default_rng(4)
        image = rng.normal(size=(32, 32, 3))
        spectrum = shift_fft(compute_fft(image))

        for radius in CANDIDATE_CUTOFFS:
            mask = create_frequency_mask(32, 32, radius)
            self.assertEqual(mask[16, 16], 1.0)
            np.testing.assert_array_equal(
                low_pass_filter(spectrum, radius) + high_pass_filter(spectrum, radius),
                spectrum,
            )

    def test_reconstruction_parseval_and_fraction_gates_pass(self) -> None:
        raw = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3) % 256
        image = preprocess_cifar10_image(raw.astype(np.uint8))

        analyses = analyze_frequency_cutoffs(image)

        self.assertEqual(
            tuple(analysis.radius for analysis in analyses),
            CANDIDATE_CUTOFFS,
        )
        for analysis in analyses:
            self.assertLessEqual(
                analysis.reconstruction_max_abs_error,
                RECONSTRUCTION_TOLERANCE,
            )
            self.assertLessEqual(
                analysis.energy_decomposition_relative_error,
                RELATIVE_ENERGY_TOLERANCE,
            )
            self.assertLessEqual(
                analysis.orthogonality_relative_error,
                RELATIVE_ENERGY_TOLERANCE,
            )
            self.assertAlmostEqual(
                analysis.low_energy_fraction + analysis.high_energy_fraction,
                1.0,
                places=14,
            )

    def test_output_schemas_match_frozen_protocol(self) -> None:
        self.assertEqual(len(MANIFEST_CSV_FIELDS), 9)
        self.assertIn("source_integrity_id", MANIFEST_CSV_FIELDS)
        self.assertEqual(len(MEASUREMENT_CSV_FIELDS), 16)
        self.assertIn(
            "energy_decomposition_relative_error",
            MEASUREMENT_CSV_FIELDS,
        )
        self.assertEqual(len(REVIEW_CSV_FIELDS), 13)
        self.assertIn("high_localization_score", REVIEW_CSV_FIELDS)


class FrequencyCutoffReviewTest(unittest.TestCase):
    """Verify blank templates and frozen two-reviewer selection behavior."""

    def setUp(self) -> None:
        self.image_records = build_image_records()
        self.image_class_ids = {
            str(record["image_id"]): int(record["class_id"])
            for record in self.image_records
        }

    def test_blank_reviewer_template_is_complete_and_unscored(self) -> None:
        rows = build_blank_reviewer_rows(self.image_records, "A")

        self.assertEqual(len(rows), 20 * len(CANDIDATE_CUTOFFS))
        self.assertEqual(
            len({(str(row["image_id"]), int(row["cutoff_radius"])) for row in rows}),
            len(rows),
        )
        self.assertTrue(reviewer_template_is_blank(rows))

    def test_blank_template_is_not_a_completed_review(self) -> None:
        rows = build_blank_reviewer_rows(self.image_records, "A")

        with self.assertRaises(ValueError):
            validate_completed_reviewer_rows(
                rows,
                reviewer_id="A",
                image_class_ids=self.image_class_ids,
            )

    def test_low_score_requires_failure_category(self) -> None:
        rows = complete_rows("A", score=0)
        rows[0]["failure_category"] = ""

        with self.assertRaises(ValueError):
            validate_completed_reviewer_rows(
                rows,
                reviewer_id="A",
                image_class_ids=self.image_class_ids,
            )

    def test_frozen_rule_selects_smallest_stable_interior_cutoff(self) -> None:
        selection = apply_frozen_cutoff_rubric(
            {
                "A": complete_rows("A", score=2),
                "B": complete_rows("B", score=2),
            },
            image_class_ids=self.image_class_ids,
            numerical_gates_passed=True,
        )

        self.assertEqual(selection.status, "selected")
        self.assertEqual(selection.reference_cutoff, 3)
        self.assertEqual(selection.adjacent_lower_cutoff, 2)
        self.assertEqual(selection.adjacent_higher_cutoff, 4)

    def test_low_scores_return_no_selection(self) -> None:
        selection = apply_frozen_cutoff_rubric(
            {
                "A": complete_rows("A", score=0),
                "B": complete_rows("B", score=0),
            },
            image_class_ids=self.image_class_ids,
            numerical_gates_passed=True,
        )

        self.assertEqual(selection.status, "no_selection")
        self.assertIsNone(selection.reference_cutoff)

    def test_reviewer_disagreement_returns_no_selection(self) -> None:
        selection = apply_frozen_cutoff_rubric(
            {
                "A": complete_rows("A", score=2),
                "B": complete_rows("B", score=0),
            },
            image_class_ids=self.image_class_ids,
            numerical_gates_passed=True,
        )

        self.assertEqual(selection.status, "no_selection")
        self.assertEqual(
            selection.qualifying_by_reviewer["A"],
            CANDIDATE_CUTOFFS,
        )
        self.assertEqual(selection.qualifying_by_reviewer["B"], ())


if __name__ == "__main__":
    unittest.main()
