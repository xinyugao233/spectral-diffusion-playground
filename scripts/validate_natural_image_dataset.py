"""Validate the frozen Experiment 5 natural-image dataset contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.dataset import (
    DatasetValidationError,
    validate_natural_image_dataset,
    write_validation_summary,
)

DEFAULT_METADATA_PATH = REPO_ROOT / "assets" / "examples" / "metadata.csv"
DEFAULT_IMAGE_DIRECTORY = REPO_ROOT / "assets" / "examples" / "natural"
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "assets" / "examples" / "natural_dataset_validation.json"
)


def parse_args() -> argparse.Namespace:
    """Parse dataset-validation command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate metadata, provenance fields, source images, and frozen "
            "preprocessing for the Experiment 5 natural-image dataset."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the frozen provenance metadata CSV.",
    )
    parser.add_argument(
        "--image-directory",
        type=Path,
        default=DEFAULT_IMAGE_DIRECTORY,
        help="Directory containing the original natural-image files.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the dataset integrity summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate the dataset, write its integrity record, and return a status code."""
    args = parse_args()
    try:
        summary = validate_natural_image_dataset(
            args.metadata_path.resolve(),
            args.image_directory.resolve(),
        )
        summary_path = write_validation_summary(summary, args.summary_path.resolve())
    except DatasetValidationError as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Dataset validation passed: {summary.num_images} images, "
        f"formats={list(summary.formats)}."
    )
    print(f"Integrity summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
