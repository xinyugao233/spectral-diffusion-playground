"""Generate the frozen CIFAR-10 frequency-cutoff reviewer packet.

This experiment measures complementary low/high Fourier decompositions for a
deterministic 20-image CIFAR-10 test-set montage. It creates numerical outputs
and blank human-review templates, but it does not score images or select a
reference cutoff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

# Import the shared bootstrap so this script can be run directly from the repo root.
import _bootstrap  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("CIFAR10_ROOT", REPO_ROOT / "data" / "cifar10")
)
CLASS_GROUP_SIZE: Final[int] = 2

from spectral_diffusion_playground.frequency_cutoff import (
    CANDIDATE_CUTOFFS,
    DISPLAY_PERCENTILE,
    EXPERIMENT_ID,
    MANIFEST_CSV_FIELDS,
    MEASUREMENT_CSV_FIELDS,
    REVIEW_CSV_FIELDS,
    CutoffAnalysis,
    analyze_frequency_cutoffs,
    build_blank_reviewer_rows,
    high_frequency_display_scale,
    image_sha256,
    image_to_display,
    preprocess_cifar10_image,
    reviewer_template_is_blank,
    select_balanced_indices,
    signed_component_to_display,
)
from spectral_diffusion_playground.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for E004 packet generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic CIFAR-10 cutoff measurements, reviewer "
            "montages, and blank reviewer templates."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root containing torchvision's cifar-10-batches-py directory. "
            "Defaults to $CIFAR10_ROOT or data/cifar10."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download CIFAR-10 when it is absent.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository-like root containing results/ and figures/ outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    """Write deterministic UTF-8 CSV with a stable field order."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, value: Mapping[str, object]) -> Path:
    """Write deterministic, timestamp-free JSON."""
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def format_float(value: float) -> str:
    """Format a float64 value for reproducible round-trippable CSV output."""
    return format(float(value), ".17g")


def build_manifest_and_analyses(
    dataset_root: Path,
    *,
    download: bool,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
    dict[str, list[CutoffAnalysis]],
]:
    """Load the frozen CIFAR-10 subset and compute all numerical analyses."""
    import torchvision
    from torchvision.datasets import CIFAR10

    resolved_root = dataset_root.expanduser().resolve()
    dataset = CIFAR10(root=resolved_root, train=False, download=download)
    selected_indices = select_balanced_indices(dataset.targets)
    class_names = tuple(str(name) for name in dataset.classes)
    test_batch_path = resolved_root / dataset.base_folder / "test_batch"
    metadata_path = resolved_root / dataset.base_folder / "batches.meta"
    if not test_batch_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Expected CIFAR-10 Python files under "
            f"{resolved_root / dataset.base_folder}."
        )

    test_batch_sha256 = sha256_file(test_batch_path)
    metadata_sha256 = sha256_file(metadata_path)
    dataset_version = (
        f"torchvision={torchvision.__version__};" "archive=cifar-10-python.tar.gz"
    )
    source_integrity_id = f"sha256:test_batch:{test_batch_sha256}"

    manifest_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    images: dict[str, np.ndarray] = {}
    analyses_by_image: dict[str, list[CutoffAnalysis]] = {}
    manifest_images: list[dict[str, object]] = []

    for dataset_index in selected_indices:
        raw_image = np.asarray(dataset.data[dataset_index])
        class_id = int(dataset.targets[dataset_index])
        class_name = class_names[class_id]
        image_id = f"cifar10-test-{dataset_index:05d}"
        raw_sha256 = image_sha256(raw_image)
        image = preprocess_cifar10_image(raw_image)
        analyses = analyze_frequency_cutoffs(image)
        display_scale = high_frequency_display_scale(analyses)

        images[image_id] = image
        analyses_by_image[image_id] = analyses
        manifest_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_name": "CIFAR-10",
                "dataset_split": "test",
                "dataset_index": dataset_index,
                "image_id": image_id,
                "class_id": class_id,
                "class_name": class_name,
                "dataset_version": dataset_version,
                "source_integrity_id": source_integrity_id,
            }
        )
        manifest_images.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "dataset_index": dataset_index,
                "image_id": image_id,
                "raw_rgb_sha256": raw_sha256,
            }
        )

        for analysis in analyses:
            measurement_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "image_id": image_id,
                    "dataset_index": dataset_index,
                    "class_id": class_id,
                    "class_name": class_name,
                    "cutoff_radius": analysis.radius,
                    "cutoff_normalized": format_float(analysis.radius / 16.0),
                    "total_energy": format_float(analysis.total_energy),
                    "low_energy": format_float(analysis.low_energy),
                    "high_energy": format_float(analysis.high_energy),
                    "low_energy_fraction": format_float(analysis.low_energy_fraction),
                    "high_energy_fraction": format_float(analysis.high_energy_fraction),
                    "reconstruction_max_abs_error": format_float(
                        analysis.reconstruction_max_abs_error
                    ),
                    "energy_decomposition_relative_error": format_float(
                        analysis.energy_decomposition_relative_error
                    ),
                    "orthogonality_relative_error": format_float(
                        analysis.orthogonality_relative_error
                    ),
                    "high_display_scale": format_float(display_scale),
                }
            )

    implementation_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src" / "spectral_diffusion_playground" / "frequency_cutoff.py",
        REPO_ROOT / "src" / "spectral_diffusion_playground" / "fft.py",
        REPO_ROOT / "src" / "spectral_diffusion_playground" / "filters.py",
    )
    manifest: dict[str, object] = {
        "dataset": {
            "class_names": list(class_names),
            "download_flag": download,
            "length": len(dataset),
            "metadata_sha256": metadata_sha256,
            "name": "CIFAR-10",
            "root": str(resolved_root),
            "split": "test",
            "test_batch_sha256": test_batch_sha256,
            "torchvision_version": torchvision.__version__,
        },
        "experiment_id": EXPERIMENT_ID,
        "fourier_convention": {
            "boundary": "inclusive radius <= r",
            "candidate_cutoffs": list(CANDIDATE_CUTOFFS),
            "channelwise": True,
            "coordinates": "centered radial frequency bins",
            "dc_band": "low",
            "fft_norm": "ortho",
            "high_mask": "exact complement of low mask",
        },
        "implementation": {
            "provenance": "paper-derived clean-room reimplementation",
            "source_sha256": {
                str(path.relative_to(REPO_ROOT)): sha256_file(path)
                for path in implementation_paths
            },
        },
        "outputs": {
            "timestamps_included": False,
            "reviewer_scores_generated": False,
            "reference_cutoff_selected": False,
        },
        "preprocessing": {
            "channel_order": "RGB",
            "formula": "x = 2 * (uint8 / 255) - 1",
            "input_dtype": "uint8",
            "output_dtype": "float64",
            "output_range": "[-1,1]",
            "resize_crop_augmentation": "none",
            "shape": [32, 32, 3],
        },
        "selection": {
            "count": len(selected_indices),
            "examples_per_class": 2,
            "images": manifest_images,
            "indices": list(selected_indices),
            "rule": (
                "For class IDs 0..9, scan canonical test indices ascending "
                "and retain the first two examples."
            ),
        },
    }
    return (
        manifest,
        manifest_rows,
        measurement_rows,
        images,
        analyses_by_image,
    )


def save_review_montages(
    output_dir: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    images: Mapping[str, np.ndarray],
    analyses_by_image: Mapping[str, Sequence[CutoffAnalysis]],
) -> list[Path]:
    """Save deterministic two-class reviewer montages for all frozen images."""
    from spectral_diffusion_playground.visualization import _apply_publication_style

    import matplotlib.pyplot as plt

    _apply_publication_style()
    ensure_directory(output_dir)
    class_ids = sorted({int(row["class_id"]) for row in manifest_rows})
    output_paths: list[Path] = []

    for group_start in range(0, len(class_ids), CLASS_GROUP_SIZE):
        group_class_ids = class_ids[group_start : group_start + CLASS_GROUP_SIZE]
        group_rows = [
            row for row in manifest_rows if int(row["class_id"]) in group_class_ids
        ]
        row_count = 2 * len(group_rows)
        column_count = 1 + len(CANDIDATE_CUTOFFS)
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(16.8, 2.35 * row_count),
            squeeze=False,
        )

        for image_offset, record in enumerate(group_rows):
            image_id = str(record["image_id"])
            image = images[image_id]
            analyses = list(analyses_by_image[image_id])
            display_scale = high_frequency_display_scale(analyses)
            low_row = 2 * image_offset
            high_row = low_row + 1
            row_identity = (
                f"{record['class_name']} | test index {record['dataset_index']}"
            )

            axes[low_row, 0].imshow(image_to_display(image))
            axes[low_row, 0].set_title(f"{row_identity}\nOriginal", fontsize=10)
            axes[high_row, 0].imshow(np.full((32, 32, 3), 0.5, dtype=np.float64))
            axes[high_row, 0].set_title(
                "Signed high-pass\nzero = neutral gray",
                fontsize=10,
            )

            for column_index, analysis in enumerate(analyses, start=1):
                axes[low_row, column_index].imshow(
                    image_to_display(analysis.low_frequency)
                )
                axes[low_row, column_index].set_title(
                    f"Low-pass r = {analysis.radius}",
                    fontsize=10,
                )
                axes[high_row, column_index].imshow(
                    signed_component_to_display(
                        analysis.high_frequency,
                        display_scale,
                    )
                )
                axes[high_row, column_index].set_title(
                    f"High-pass r = {analysis.radius}",
                    fontsize=10,
                )

            for axis in axes[low_row, :]:
                axis.axis("off")
            for axis in axes[high_row, :]:
                axis.axis("off")
            axes[high_row, 0].text(
                0.5,
                -0.08,
                f"Per-image display scale: {display_scale:.4g}",
                transform=axes[high_row, 0].transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
            )

        first_class = group_class_ids[0]
        last_class = group_class_ids[-1]
        figure.suptitle(
            (
                "E004 CIFAR-10 Frequency Cutoff Review "
                f"(classes {first_class}-{last_class})\n"
                "Raw float64 measurements; high-pass contrast is display-only "
                f"at the per-image {DISPLAY_PERCENTILE:g}th percentile"
            ),
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        output_path = (
            output_dir
            / f"experiment_04_cutoff_montage_classes_{first_class}_{last_class}.png"
        )
        with plt.rc_context({"savefig.bbox": None}):
            figure.savefig(
                output_path,
                dpi=180,
                facecolor="white",
                bbox_inches=None,
            )
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def generate_review_packet(
    dataset_root: Path,
    output_root: Path,
    *,
    download: bool,
) -> dict[str, object]:
    """Generate all E004 measurements, manifests, montages, and blank reviews."""
    (
        manifest,
        manifest_rows,
        measurement_rows,
        images,
        analyses_by_image,
    ) = build_manifest_and_analyses(dataset_root, download=download)

    output_root = output_root.expanduser().resolve()
    results_dir = ensure_directory(output_root / "results")
    figures_dir = ensure_directory(output_root / "figures")

    manifest_csv_path = write_csv(
        results_dir / "experiment_04_image_manifest.csv",
        manifest_rows,
        MANIFEST_CSV_FIELDS,
    )
    manifest_json_path = write_json(
        results_dir / "experiment_04_manifest.json",
        manifest,
    )
    energy_path = write_csv(
        results_dir / "experiment_04_cutoff_energy.csv",
        measurement_rows,
        MEASUREMENT_CSV_FIELDS,
    )
    measurement_path = write_csv(
        results_dir / "experiment_04_cutoff_measurements.csv",
        measurement_rows,
        MEASUREMENT_CSV_FIELDS,
    )

    reviewer_a_rows = build_blank_reviewer_rows(manifest_rows, "A")
    reviewer_b_rows = build_blank_reviewer_rows(manifest_rows, "B")
    if not reviewer_template_is_blank(reviewer_a_rows):
        raise RuntimeError("Reviewer A template unexpectedly contains scores.")
    if not reviewer_template_is_blank(reviewer_b_rows):
        raise RuntimeError("Reviewer B template unexpectedly contains scores.")
    reviewer_a_path = write_csv(
        results_dir / "experiment_04_reviewer_A.csv",
        reviewer_a_rows,
        REVIEW_CSV_FIELDS,
    )
    reviewer_b_path = write_csv(
        results_dir / "experiment_04_reviewer_B.csv",
        reviewer_b_rows,
        REVIEW_CSV_FIELDS,
    )
    combined_review_path = write_csv(
        results_dir / "experiment_04_cutoff_review.csv",
        [*reviewer_a_rows, *reviewer_b_rows],
        REVIEW_CSV_FIELDS,
    )
    montage_paths = save_review_montages(
        figures_dir,
        manifest_rows,
        images,
        analyses_by_image,
    )

    reconstruction_errors = [
        analysis.reconstruction_max_abs_error
        for analyses in analyses_by_image.values()
        for analysis in analyses
    ]
    energy_errors = [
        analysis.energy_decomposition_relative_error
        for analyses in analyses_by_image.values()
        for analysis in analyses
    ]
    return {
        "energy_path": energy_path,
        "manifest_csv_path": manifest_csv_path,
        "manifest_json_path": manifest_json_path,
        "maximum_energy_relative_error": max(energy_errors),
        "maximum_reconstruction_error": max(reconstruction_errors),
        "measurement_path": measurement_path,
        "montage_paths": montage_paths,
        "reviewer_a_path": reviewer_a_path,
        "reviewer_b_path": reviewer_b_path,
        "combined_review_path": combined_review_path,
        "selected_indices": manifest["selection"]["indices"],
    }


def main() -> int:
    """Generate the E004 review packet without selecting a cutoff."""
    args = parse_args()
    outputs = generate_review_packet(
        args.dataset_root,
        args.output_root,
        download=args.download,
    )

    print("Experiment 4 review packet generated.")
    print(f"Selected CIFAR-10 test indices: {outputs['selected_indices']}")
    print(
        "Maximum reconstruction error: "
        f"{outputs['maximum_reconstruction_error']:.3e}"
    )
    print(
        "Maximum relative Parseval-energy discrepancy: "
        f"{outputs['maximum_energy_relative_error']:.3e}"
    )
    print(f"Manifest JSON: {outputs['manifest_json_path']}")
    print(f"Manifest CSV: {outputs['manifest_csv_path']}")
    print(f"Measurements: {outputs['measurement_path']}")
    print(f"Protocol energy CSV: {outputs['energy_path']}")
    print(f"Reviewer A template: {outputs['reviewer_a_path']}")
    print(f"Reviewer B template: {outputs['reviewer_b_path']}")
    print(f"Combined review template: {outputs['combined_review_path']}")
    for montage_path in outputs["montage_paths"]:
        print(f"Reviewer montage: {montage_path}")
    print("Reviewer templates are blank. No reference cutoff was selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
