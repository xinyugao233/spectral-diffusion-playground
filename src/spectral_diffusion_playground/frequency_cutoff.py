"""Reusable measurement and review logic for Experiment 4."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from spectral_diffusion_playground.filters import decompose_frequency_bands

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]

EXPERIMENT_ID: Final[str] = "E004"
CANDIDATE_CUTOFFS: Final[tuple[int, ...]] = (2, 3, 4, 5, 6, 8)
INTERIOR_CUTOFFS: Final[tuple[int, ...]] = (3, 4, 5, 6)
RECONSTRUCTION_TOLERANCE: Final[float] = 1e-10
RELATIVE_ENERGY_TOLERANCE: Final[float] = 1e-12
DISPLAY_PERCENTILE: Final[float] = 99.5
FLOAT64_TINY: Final[float] = float(np.finfo(np.float64).tiny)

MANIFEST_CSV_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "dataset_name",
    "dataset_split",
    "dataset_index",
    "image_id",
    "class_id",
    "class_name",
    "dataset_version",
    "source_integrity_id",
)
MEASUREMENT_CSV_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "image_id",
    "dataset_index",
    "class_id",
    "class_name",
    "cutoff_radius",
    "cutoff_normalized",
    "total_energy",
    "low_energy",
    "high_energy",
    "low_energy_fraction",
    "high_energy_fraction",
    "reconstruction_max_abs_error",
    "energy_decomposition_relative_error",
    "orthogonality_relative_error",
    "high_display_scale",
)
REVIEW_CSV_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "reviewer_id",
    "image_id",
    "dataset_index",
    "class_id",
    "class_name",
    "cutoff_radius",
    "layout_score",
    "identity_score",
    "high_localization_score",
    "ambiguous",
    "failure_category",
    "comment",
)
SCORE_FIELDS: Final[tuple[str, ...]] = (
    "layout_score",
    "identity_score",
    "high_localization_score",
)


@dataclass(frozen=True, slots=True)
class CutoffAnalysis:
    """Numerical decomposition for one image and one cutoff."""

    radius: int
    low_frequency: FloatArray
    high_frequency: FloatArray
    total_energy: float
    low_energy: float
    high_energy: float
    low_energy_fraction: float
    high_energy_fraction: float
    reconstruction_max_abs_error: float
    energy_decomposition_relative_error: float
    orthogonality_relative_error: float


@dataclass(frozen=True, slots=True)
class CutoffSelection:
    """Outcome of applying the frozen rubric to completed reviewer scores."""

    status: str
    reference_cutoff: int | None
    adjacent_lower_cutoff: int | None
    adjacent_higher_cutoff: int | None
    qualifying_by_reviewer: Mapping[str, tuple[int, ...]]


def select_balanced_indices(
    labels: Sequence[int],
    *,
    class_ids: Sequence[int] = tuple(range(10)),
    examples_per_class: int = 2,
) -> tuple[int, ...]:
    """Select the first examples per class in class-ID order.

    Dataset indices are scanned in ascending order. The returned ordering is
    class ID first and encounter order second, matching the frozen protocol.
    """
    if examples_per_class <= 0:
        raise ValueError("examples_per_class must be positive.")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be unique.")

    selected: list[int] = []
    for class_id in class_ids:
        class_indices = [
            index for index, label in enumerate(labels) if int(label) == class_id
        ][:examples_per_class]
        if len(class_indices) != examples_per_class:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} examples; "
                f"{examples_per_class} are required."
            )
        selected.extend(class_indices)
    return tuple(selected)


def preprocess_cifar10_image(image: np.ndarray) -> FloatArray:
    """Convert one uint8 CIFAR-10 RGB image to float64 in ``[-1, 1]``."""
    image_array = np.asarray(image)
    if image_array.shape != (32, 32, 3):
        raise ValueError(
            "Expected a CIFAR-10 RGB image with shape (32, 32, 3), "
            f"but received {image_array.shape}."
        )
    if image_array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels, but received {image_array.dtype}.")
    return np.asarray(2.0 * (image_array.astype(np.float64) / 255.0) - 1.0)


def image_sha256(image: np.ndarray) -> str:
    """Return a stable SHA-256 digest of one raw uint8 RGB image."""
    image_array = np.asarray(image)
    if image_array.dtype != np.uint8:
        raise ValueError("Image identity hashes must be computed from uint8 pixels.")
    contiguous = np.ascontiguousarray(image_array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def analyze_frequency_cutoffs(
    image: np.ndarray,
    *,
    cutoffs: Sequence[int] = CANDIDATE_CUTOFFS,
) -> list[CutoffAnalysis]:
    """Measure complementary Fourier decompositions for every cutoff."""
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.shape != (32, 32, 3):
        raise ValueError(
            "Expected a preprocessed CIFAR-10 image with shape (32, 32, 3), "
            f"but received {image_array.shape}."
        )
    if not np.all(np.isfinite(image_array)):
        raise ValueError("image must contain only finite values.")

    total_energy = float(np.vdot(image_array, image_array).real)
    if total_energy <= 0.0:
        raise ValueError("image must have positive energy.")

    analyses: list[CutoffAnalysis] = []
    for radius in cutoffs:
        low_frequency, high_frequency = decompose_frequency_bands(
            image_array,
            radius=float(radius),
        )
        reconstruction_error = float(
            np.max(np.abs(image_array - low_frequency - high_frequency))
        )
        low_energy = float(np.vdot(low_frequency, low_frequency).real)
        high_energy = float(np.vdot(high_frequency, high_frequency).real)
        energy_error = abs(total_energy - low_energy - high_energy) / max(
            total_energy, FLOAT64_TINY
        )
        orthogonality_error = abs(
            float(np.vdot(low_frequency, high_frequency).real)
        ) / max(total_energy, FLOAT64_TINY)
        low_fraction = low_energy / total_energy
        high_fraction = high_energy / total_energy

        if reconstruction_error > RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(
                f"Reconstruction gate failed at r={radius}: "
                f"{reconstruction_error:.3e}."
            )
        if energy_error > RELATIVE_ENERGY_TOLERANCE:
            raise RuntimeError(f"Energy gate failed at r={radius}: {energy_error:.3e}.")
        if orthogonality_error > RELATIVE_ENERGY_TOLERANCE:
            raise RuntimeError(
                f"Orthogonality gate failed at r={radius}: "
                f"{orthogonality_error:.3e}."
            )
        if abs(low_fraction + high_fraction - 1.0) > RELATIVE_ENERGY_TOLERANCE:
            raise RuntimeError(
                f"Energy-fraction gate failed at r={radius}: "
                f"{low_fraction + high_fraction:.17g}."
            )

        analyses.append(
            CutoffAnalysis(
                radius=int(radius),
                low_frequency=low_frequency,
                high_frequency=high_frequency,
                total_energy=total_energy,
                low_energy=low_energy,
                high_energy=high_energy,
                low_energy_fraction=low_fraction,
                high_energy_fraction=high_fraction,
                reconstruction_max_abs_error=reconstruction_error,
                energy_decomposition_relative_error=energy_error,
                orthogonality_relative_error=orthogonality_error,
            )
        )
    return analyses


def high_frequency_display_scale(
    analyses: Sequence[CutoffAnalysis],
    *,
    percentile: float = DISPLAY_PERCENTILE,
) -> float:
    """Compute one per-image signed display scale shared across all cutoffs."""
    if not analyses:
        raise ValueError("analyses must contain at least one cutoff.")
    absolute_values = np.concatenate(
        [np.abs(analysis.high_frequency).ravel() for analysis in analyses]
    )
    scale = float(np.percentile(absolute_values, percentile))
    return scale if scale > 0.0 else 1.0


def image_to_display(image: np.ndarray) -> FloatArray:
    """Map a raw ``[-1, 1]`` image to clipped display RGB in ``[0, 1]``."""
    image_array = np.asarray(image, dtype=np.float64)
    return np.asarray(np.clip((image_array + 1.0) / 2.0, 0.0, 1.0))


def signed_component_to_display(component: np.ndarray, scale: float) -> FloatArray:
    """Map one signed high-pass component to display space around neutral gray."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")
    component_array = np.asarray(component, dtype=np.float64)
    return np.asarray(np.clip(0.5 + 0.5 * component_array / scale, 0.0, 1.0))


def build_blank_reviewer_rows(
    image_records: Sequence[Mapping[str, object]],
    reviewer_id: str,
    *,
    cutoffs: Sequence[int] = CANDIDATE_CUTOFFS,
) -> list[dict[str, object]]:
    """Create a complete reviewer template with every score field blank."""
    if not reviewer_id:
        raise ValueError("reviewer_id must be nonempty.")

    rows: list[dict[str, object]] = []
    for record in image_records:
        for radius in cutoffs:
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "reviewer_id": reviewer_id,
                    "image_id": record["image_id"],
                    "dataset_index": record["dataset_index"],
                    "class_id": record["class_id"],
                    "class_name": record["class_name"],
                    "cutoff_radius": int(radius),
                    "layout_score": "",
                    "identity_score": "",
                    "high_localization_score": "",
                    "ambiguous": "",
                    "failure_category": "",
                    "comment": "",
                }
            )
    return rows


def reviewer_template_is_blank(rows: Sequence[Mapping[str, object]]) -> bool:
    """Return whether all human-entered reviewer fields remain empty."""
    editable_fields = (
        *SCORE_FIELDS,
        "ambiguous",
        "failure_category",
        "comment",
    )
    return all(
        str(row.get(field, "")).strip() == ""
        for row in rows
        for field in editable_fields
    )


def validate_completed_reviewer_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    reviewer_id: str,
    image_class_ids: Mapping[str, int],
    cutoffs: Sequence[int] = CANDIDATE_CUTOFFS,
) -> None:
    """Validate completeness and allowed values in one completed review."""
    if len(image_class_ids) != 20:
        raise ValueError("The frozen review requires exactly 20 image IDs.")
    class_counts = {
        class_id: sum(value == class_id for value in image_class_ids.values())
        for class_id in set(image_class_ids.values())
    }
    if len(class_counts) != 10 or any(count != 2 for count in class_counts.values()):
        raise ValueError(
            "The frozen review requires two images from each of 10 classes."
        )

    expected_pairs = {
        (image_id, int(radius)) for image_id in image_class_ids for radius in cutoffs
    }
    observed_pairs: set[tuple[str, int]] = set()

    for row in rows:
        if str(row.get("experiment_id", "")) != EXPERIMENT_ID:
            raise ValueError(f"Every row must have experiment_id={EXPERIMENT_ID!r}.")
        if str(row.get("reviewer_id", "")) != reviewer_id:
            raise ValueError(f"Every row must have reviewer_id={reviewer_id!r}.")
        image_id = str(row.get("image_id", ""))
        if image_id not in image_class_ids:
            raise ValueError(f"Unexpected image_id {image_id!r}.")
        if int(row["class_id"]) != image_class_ids[image_id]:
            raise ValueError(f"class_id does not match {image_id!r}.")
        radius = int(row["cutoff_radius"])
        pair = (image_id, radius)
        if pair in observed_pairs:
            raise ValueError(f"Duplicate review row for {pair}.")
        observed_pairs.add(pair)

        scores: list[int] = []
        for field in SCORE_FIELDS:
            raw_value = str(row.get(field, "")).strip()
            if raw_value == "":
                raise ValueError(f"{field} is blank for {pair}.")
            score = int(raw_value)
            if score not in (0, 1, 2):
                raise ValueError(f"{field} must be 0, 1, or 2 for {pair}.")
            scores.append(score)

        ambiguous = str(row.get("ambiguous", "")).strip().lower()
        if ambiguous not in ("false", "true"):
            raise ValueError(f"ambiguous must be true or false for {pair}.")
        comment = str(row.get("comment", "")).strip()
        failure_category = str(row.get("failure_category", "")).strip()
        requires_failure_record = min(scores) < 2 or ambiguous == "true"
        if requires_failure_record and not comment:
            raise ValueError(
                f"A comment is required for ambiguous/low scores at {pair}."
            )
        if requires_failure_record and not failure_category:
            raise ValueError(
                f"A failure_category is required for ambiguous/low scores at {pair}."
            )

    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        unexpected = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            f"Reviewer rows do not match the frozen grid; "
            f"missing={missing}, unexpected={unexpected}."
        )


def apply_frozen_cutoff_rubric(
    reviewer_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    image_class_ids: Mapping[str, int],
    numerical_gates_passed: bool,
) -> CutoffSelection:
    """Apply the frozen two-reviewer qualification and stability rule."""
    if set(reviewer_rows) != {"A", "B"}:
        raise ValueError(
            "Completed reviews from exactly reviewers A and B are required."
        )

    qualifying_by_reviewer: dict[str, tuple[int, ...]] = {}
    for reviewer_id in ("A", "B"):
        rows = reviewer_rows[reviewer_id]
        validate_completed_reviewer_rows(
            rows,
            reviewer_id=reviewer_id,
            image_class_ids=image_class_ids,
        )
        qualifying: list[int] = []
        for radius in CANDIDATE_CUTOFFS:
            cutoff_rows = [row for row in rows if int(row["cutoff_radius"]) == radius]
            layout_count = sum(int(row["layout_score"]) >= 1 for row in cutoff_rows)
            identity_count = sum(int(row["identity_score"]) >= 1 for row in cutoff_rows)
            localization_count = sum(
                int(row["high_localization_score"]) >= 1 for row in cutoff_rows
            )
            class_coverage = all(
                any(
                    image_class_ids[str(row["image_id"])] == class_id
                    and int(row["layout_score"]) >= 1
                    and int(row["identity_score"]) >= 1
                    for row in cutoff_rows
                )
                for class_id in sorted(set(image_class_ids.values()))
            )
            if (
                layout_count >= 16
                and identity_count >= 14
                and localization_count >= 16
                and class_coverage
            ):
                qualifying.append(radius)
        qualifying_by_reviewer[reviewer_id] = tuple(qualifying)

    if not numerical_gates_passed:
        return CutoffSelection(
            status="no_selection",
            reference_cutoff=None,
            adjacent_lower_cutoff=None,
            adjacent_higher_cutoff=None,
            qualifying_by_reviewer=qualifying_by_reviewer,
        )

    overall = set(qualifying_by_reviewer["A"]) & set(qualifying_by_reviewer["B"])
    reference_cutoff: int | None = None
    for radius in INTERIOR_CUTOFFS:
        next_higher = CANDIDATE_CUTOFFS[CANDIDATE_CUTOFFS.index(radius) + 1]
        if radius in overall and next_higher in overall:
            reference_cutoff = radius
            break

    if reference_cutoff is None:
        return CutoffSelection(
            status="no_selection",
            reference_cutoff=None,
            adjacent_lower_cutoff=None,
            adjacent_higher_cutoff=None,
            qualifying_by_reviewer=qualifying_by_reviewer,
        )

    cutoff_index = CANDIDATE_CUTOFFS.index(reference_cutoff)
    return CutoffSelection(
        status="selected",
        reference_cutoff=reference_cutoff,
        adjacent_lower_cutoff=CANDIDATE_CUTOFFS[cutoff_index - 1],
        adjacent_higher_cutoff=CANDIDATE_CUTOFFS[cutoff_index + 1],
        qualifying_by_reviewer=qualifying_by_reviewer,
    )
