"""Tests for the frozen natural-image dataset validator."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.dataset import (
    FROZEN_PREPROCESSING,
    REQUIRED_METADATA_COLUMNS,
    DatasetValidationError,
    validate_natural_image_dataset,
)


class NaturalImageDatasetValidationTest(unittest.TestCase):
    """Verify that invalid calibration datasets fail loudly."""

    def setUp(self) -> None:
        """Create a minimal valid six-image provenance-recorded JPEG dataset."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_directory = self.root / "natural"
        self.image_directory.mkdir()
        self.metadata_path = self.root / "metadata.csv"
        self.rows = []
        for index in range(1, 7):
            image_id = f"image_{index:03d}"
            row = {
                "image_id": image_id,
                "filename": f"{image_id}.jpg",
                "source": "Example source",
                "creator": "Example creator",
                "license": "CC BY 4.0",
                "url": f"https://example.com/{image_id}",
                "download_date": "2026-07-25",
                "original_resolution": "12x8",
                "preprocessing": FROZEN_PREPROCESSING,
            }
            Image.new("RGB", (12, 8), color=(32, 128, 224)).save(
                self.image_directory / row["filename"], format="JPEG"
            )
            self.rows.append(row)
        self.row = self.rows[0]
        self._write_rows(self.rows)

    def tearDown(self) -> None:
        """Remove the temporary dataset."""
        self.temporary_directory.cleanup()

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        """Write test metadata using the frozen column order."""
        with self.metadata_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=REQUIRED_METADATA_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_dataset_produces_integrity_summary(self) -> None:
        """A complete dataset should validate and report deterministic facts."""
        summary = validate_natural_image_dataset(
            self.metadata_path,
            self.image_directory,
            validation_timestamp="2026-07-25T00:00:00Z",
        )

        self.assertEqual(summary.num_images, 6)
        self.assertEqual(summary.formats, ("JPEG",))
        self.assertTrue(summary.all_rgb_convertible)
        self.assertTrue(summary.metadata_complete)
        self.assertEqual(summary.validation_timestamp, "2026-07-25T00:00:00Z")

    def test_missing_image_file_fails(self) -> None:
        """A metadata row without its source file should fail."""
        (self.image_directory / self.row["filename"]).unlink()

        with self.assertRaisesRegex(DatasetValidationError, "Missing files"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_undocumented_image_file_fails(self) -> None:
        """An image without a metadata row should fail."""
        Image.new("RGB", (8, 8)).save(
            self.image_directory / "image_999.jpg", format="JPEG"
        )

        with self.assertRaisesRegex(DatasetValidationError, "undocumented files"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_non_jpeg_image_file_fails(self) -> None:
        """An unsupported image format should not be silently ignored."""
        Image.new("RGB", (8, 8)).save(
            self.image_directory / "image_002.png", format="PNG"
        )

        with self.assertRaisesRegex(DatasetValidationError, "unsupported non-JPEG"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_duplicate_image_id_fails(self) -> None:
        """Duplicate identifiers should fail even when filenames differ."""
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[-1]["image_id"] = self.row["image_id"]
        self._write_rows(changed_rows)

        with self.assertRaisesRegex(DatasetValidationError, "duplicate image_id"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_empty_provenance_field_fails(self) -> None:
        """Every frozen provenance field must be populated."""
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["license"] = ""
        self._write_rows(changed_rows)

        with self.assertRaisesRegex(DatasetValidationError, "empty fields"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_changed_preprocessing_declaration_fails(self) -> None:
        """The validator should reject changes to the frozen pipeline."""
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["preprocessing"] = "RGB conversion;resize 256x256"
        self._write_rows(changed_rows)

        with self.assertRaisesRegex(DatasetValidationError, "frozen protocol"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_recorded_resolution_must_match_decoded_image(self) -> None:
        """Recorded source dimensions should match the downloaded file."""
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["original_resolution"] = "8x12"
        self._write_rows(changed_rows)

        with self.assertRaisesRegex(DatasetValidationError, "does not match"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_corrupt_jpeg_fails(self) -> None:
        """A source file that cannot be decoded should fail."""
        (self.image_directory / self.row["filename"]).write_bytes(b"not a jpeg")

        with self.assertRaisesRegex(DatasetValidationError, "cannot be decoded"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)

    def test_dataset_below_frozen_size_range_fails(self) -> None:
        """A partial calibration set should not pass the gate."""
        for row in self.rows[4:]:
            (self.image_directory / row["filename"]).unlink()
        self._write_rows(self.rows[:4])

        with self.assertRaisesRegex(DatasetValidationError, "5-10 images"):
            validate_natural_image_dataset(self.metadata_path, self.image_directory)


if __name__ == "__main__":
    unittest.main()
