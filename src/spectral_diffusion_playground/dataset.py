"""Validation utilities for provenance-recorded natural image datasets."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError

REQUIRED_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "image_id",
    "filename",
    "source",
    "creator",
    "license",
    "url",
    "download_date",
    "original_resolution",
    "preprocessing",
)
FROZEN_PREPROCESSING: Final[str] = ";".join(
    (
        "RGB conversion",
        "center crop to square",
        "bicubic resize 256x256",
        "float32",
        "scale [0,1]",
    )
)
MIN_DATASET_IMAGES: Final[int] = 5
MAX_DATASET_IMAGES: Final[int] = 10
SUPPORTED_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg"})
KNOWN_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
)
EXPECTED_IMAGE_FORMAT: Final[str] = "JPEG"
RESOLUTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^([1-9]\d*)x([1-9]\d*)$")
ISO_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DatasetValidationError(ValueError):
    """Raised when the natural-image dataset violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    """Machine-readable summary of a successful dataset validation."""

    num_images: int
    formats: tuple[str, ...]
    all_rgb_convertible: bool
    metadata_complete: bool
    validation_timestamp: str


def load_natural_image_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """Read metadata and enforce the frozen column schema."""
    if not metadata_path.is_file():
        raise DatasetValidationError(f"Metadata file does not exist: {metadata_path}")

    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        columns = tuple(reader.fieldnames or ())
        if columns != REQUIRED_METADATA_COLUMNS:
            raise DatasetValidationError(
                "Metadata columns must exactly match the frozen schema. "
                f"Expected {REQUIRED_METADATA_COLUMNS}, received {columns}."
            )
        rows = list(reader)

    if not MIN_DATASET_IMAGES <= len(rows) <= MAX_DATASET_IMAGES:
        raise DatasetValidationError(
            f"Metadata must contain {MIN_DATASET_IMAGES}-{MAX_DATASET_IMAGES} "
            f"images, received {len(rows)}."
        )
    return rows


def _validate_unique_values(rows: list[dict[str, str]], field: str) -> None:
    """Require a metadata field to contain one unique value per row."""
    values = [row[field] for row in rows]
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise DatasetValidationError(
            f"Metadata contains duplicate {field} values: {duplicates}"
        )


def _parse_recorded_resolution(value: str, image_id: str) -> tuple[int, int]:
    """Parse a frozen ``WIDTHxHEIGHT`` resolution field."""
    match = RESOLUTION_PATTERN.fullmatch(value)
    if match is None:
        raise DatasetValidationError(
            f"{image_id}: original_resolution must use WIDTHxHEIGHT, received {value!r}."
        )
    return int(match.group(1)), int(match.group(2))


def _validate_provenance(row: dict[str, str]) -> None:
    """Validate nonempty provenance fields and their basic syntax."""
    image_id = row["image_id"]
    empty_fields = [column for column in REQUIRED_METADATA_COLUMNS if not row[column]]
    if empty_fields:
        raise DatasetValidationError(
            f"{image_id or '<missing image_id>'}: empty fields {empty_fields}."
        )

    parsed_url = urlparse(row["url"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise DatasetValidationError(
            f"{image_id}: url must be an absolute HTTP(S) URL."
        )

    try:
        if ISO_DATE_PATTERN.fullmatch(row["download_date"]) is None:
            raise ValueError
        date.fromisoformat(row["download_date"])
    except ValueError as error:
        raise DatasetValidationError(
            f"{image_id}: download_date must use ISO YYYY-MM-DD format."
        ) from error

    if row["preprocessing"] != FROZEN_PREPROCESSING:
        raise DatasetValidationError(
            f"{image_id}: preprocessing does not match the frozen protocol."
        )


def preprocess_rgb_image(image: Image.Image) -> np.ndarray:
    """Apply the frozen in-memory preprocessing pipeline to a decoded image."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped = rgb_image.crop((left, top, left + crop_size, top + crop_size))
    resized = cropped.resize((256, 256), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def load_preprocessed_natural_image(image_path: Path) -> np.ndarray:
    """Decode one source image and apply the frozen preprocessing pipeline."""
    try:
        with Image.open(image_path) as image:
            return preprocess_rgb_image(image)
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetValidationError(
            f"Image cannot be decoded as a supported image: {image_path}"
        ) from error


def _validate_image(image_path: Path, row: dict[str, str]) -> str:
    """Decode one source image and verify format, dimensions, and preprocessing."""
    image_id = row["image_id"]
    try:
        with Image.open(image_path) as image:
            image_format = image.format
            if image_format != EXPECTED_IMAGE_FORMAT:
                raise DatasetValidationError(
                    f"{image_id}: expected JPEG data, received {image_format!r}."
                )

            recorded_resolution = _parse_recorded_resolution(
                row["original_resolution"], image_id
            )
            if image.size != recorded_resolution:
                raise DatasetValidationError(
                    f"{image_id}: decoded resolution {image.size} does not match "
                    f"recorded resolution {recorded_resolution}."
                )
            processed = preprocess_rgb_image(image)
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetValidationError(
            f"{image_id}: image cannot be decoded as a supported image."
        ) from error

    if processed.shape != (256, 256, 3):
        raise DatasetValidationError(
            f"{image_id}: preprocessing produced unexpected shape {processed.shape}."
        )
    if processed.dtype != np.float32:
        raise DatasetValidationError(
            f"{image_id}: preprocessing produced dtype {processed.dtype}, not float32."
        )
    if not np.isfinite(processed).all():
        raise DatasetValidationError(
            f"{image_id}: preprocessing produced non-finite values."
        )
    if float(processed.min()) < 0.0 or float(processed.max()) > 1.0:
        raise DatasetValidationError(
            f"{image_id}: preprocessing produced values outside [0,1]."
        )
    return image_format


def validate_natural_image_dataset(
    metadata_path: Path,
    image_directory: Path,
    *,
    validation_timestamp: str | None = None,
) -> DatasetValidationSummary:
    """Validate the frozen natural-image dataset contract.

    The check is intentionally offline. It validates provenance completeness
    and URL syntax but does not make network requests to external source pages.
    """
    rows = load_natural_image_metadata(metadata_path)
    for row in rows:
        _validate_provenance(row)

    _validate_unique_values(rows, "image_id")
    _validate_unique_values(rows, "filename")

    if not image_directory.is_dir():
        raise DatasetValidationError(
            f"Image directory does not exist: {image_directory}"
        )

    metadata_filenames = {row["filename"] for row in rows}
    for filename in metadata_filenames:
        path = Path(filename)
        if path.name != filename or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise DatasetValidationError(
                f"Metadata filename must be a local JPEG basename: {filename!r}."
            )

    image_paths = {
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in KNOWN_IMAGE_SUFFIXES
    }
    unsupported_files = sorted(
        path.name
        for path in image_paths
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
    )
    if unsupported_files:
        raise DatasetValidationError(
            f"Dataset contains unsupported non-JPEG image files: {unsupported_files}."
        )

    image_filenames = {path.name for path in image_paths}
    missing_files = sorted(metadata_filenames - image_filenames)
    undocumented_files = sorted(image_filenames - metadata_filenames)
    if missing_files or undocumented_files:
        raise DatasetValidationError(
            "Metadata and image files do not match. "
            f"Missing files: {missing_files}; undocumented files: {undocumented_files}."
        )

    formats = {_validate_image(image_directory / row["filename"], row) for row in rows}
    timestamp = validation_timestamp or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return DatasetValidationSummary(
        num_images=len(rows),
        formats=tuple(sorted(formats)),
        all_rgb_convertible=True,
        metadata_complete=True,
        validation_timestamp=timestamp,
    )


def write_validation_summary(
    summary: DatasetValidationSummary, output_path: Path
) -> Path:
    """Write a successful validation summary as stable, readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(summary), output_file, indent=2)
        output_file.write("\n")
    return output_path
