"""Deterministic nested CIFAR-10 subsets for the E009 training design."""

from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
LABEL_FILE_HASHES = {
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
}
ANCHOR_TEXT_SHA256 = "33bb509c48144464a48d3b945cc44c14f880a1e6c6470c283dc0ed65e22b1f29"
SOURCE_ARCHIVE_SHA256 = (
    "795cdc1444465ae4e19e25a0615d05ba0a0e83caa5db6b1b811deaf4c7910dfa"
)


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cifar10_labels(batch_root: Path, *, verify_hashes: bool = True) -> np.ndarray:
    """Load labels in canonical CIFAR-10 training order from Python batches."""
    labels: list[int] = []
    for filename, expected_hash in LABEL_FILE_HASHES.items():
        path = batch_root / filename
        if verify_hashes and sha256_file(path) != expected_hash:
            raise ValueError(f"CIFAR-10 label source hash mismatch: {path}")
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        values = batch.get(b"labels", batch.get("labels"))
        if values is None:
            raise ValueError(f"Missing labels in {path}")
        labels.extend(int(value) for value in values)
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (50000,) or not np.all((0 <= result) & (result < 10)):
        raise ValueError("Expected 50,000 CIFAR-10 labels in [0, 9]")
    return result


def load_indices(path: Path) -> np.ndarray:
    """Load one sorted unique index per line."""
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if values.ndim != 1 or len(np.unique(values)) != len(values):
        raise ValueError(f"Index manifest must be one-dimensional and unique: {path}")
    if len(values) and (int(values[0]) < 0 or int(values[-1]) >= 50000):
        raise ValueError(f"Index manifest is outside CIFAR-10 range: {path}")
    if not np.array_equal(values, np.sort(values)):
        raise ValueError(f"Index manifest must be sorted: {path}")
    return values


def build_nested_class_balanced_subsets(
    labels: np.ndarray,
    anchor_indices: np.ndarray,
    sizes: Iterable[int],
    *,
    expansion_seed: int,
) -> dict[int, np.ndarray]:
    """Expand an anchor into deterministic nested, exactly balanced subsets."""
    labels = np.asarray(labels, dtype=np.int64)
    anchor = np.asarray(anchor_indices, dtype=np.int64)
    requested = sorted(set(int(size) for size in sizes))
    if labels.shape != (50000,):
        raise ValueError("labels must contain exactly 50,000 entries")
    if not requested or any(size <= len(anchor) or size % 10 for size in requested):
        raise ValueError("subset sizes must exceed the anchor and divide evenly by 10")
    if len(np.unique(anchor)) != len(anchor):
        raise ValueError("anchor indices must be unique")

    anchor_set = set(int(index) for index in anchor)
    pools: dict[int, np.ndarray] = {}
    for class_id in range(10):
        candidates = np.flatnonzero(labels == class_id)
        candidates = np.asarray(
            [index for index in candidates if int(index) not in anchor_set],
            dtype=np.int64,
        )
        class_rng = np.random.RandomState(expansion_seed + class_id)
        class_rng.shuffle(candidates)
        pools[class_id] = candidates

    subsets: dict[int, np.ndarray] = {}
    for size in requested:
        target_per_class = size // 10
        selected = list(int(index) for index in anchor)
        for class_id in range(10):
            anchor_count = int(np.count_nonzero(labels[anchor] == class_id))
            needed = target_per_class - anchor_count
            if needed < 0 or needed > len(pools[class_id]):
                raise ValueError(
                    f"Cannot reach class target {target_per_class} for class {class_id}"
                )
            selected.extend(int(index) for index in pools[class_id][:needed])
        values = np.sort(np.asarray(selected, dtype=np.int64))
        if len(values) != size or len(np.unique(values)) != size:
            raise AssertionError("constructed subset has incorrect size or duplicates")
        counts = np.bincount(labels[values], minlength=10)
        if not np.array_equal(counts, np.full(10, target_per_class)):
            raise AssertionError("constructed subset is not exactly class balanced")
        if not set(anchor).issubset(set(values)):
            raise AssertionError("constructed subset lost anchor indices")
        subsets[size] = values

    for smaller, larger in zip(requested, requested[1:]):
        if not set(subsets[smaller]).issubset(set(subsets[larger])):
            raise AssertionError("constructed subsets are not nested")
    return subsets


def write_indices(path: Path, indices: np.ndarray) -> None:
    """Write the canonical newline-terminated decimal index representation."""
    path.write_text("".join(f"{int(index)}\n" for index in indices), encoding="ascii")


def little_endian_int64_sha256(indices: np.ndarray) -> str:
    """Hash indices in the frozen little-endian int64 representation."""
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def materialize_subset_archive(
    source_archive: Path,
    destination: Path,
    indices: np.ndarray,
) -> None:
    """Create a deterministic ZIP containing exact selected source PNG bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = {f"img{int(index):08d}.png" for index in indices}
    with zipfile.ZipFile(source_archive, "r") as source:
        source_names = {
            Path(name).name: name
            for name in source.namelist()
            if name.lower().endswith(".png")
        }
        missing = selected.difference(source_names)
        if missing:
            raise ValueError(f"Source archive is missing {len(missing)} selected PNGs")
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as output:
            for basename in sorted(selected):
                data = source.read(source_names[basename])
                info = zipfile.ZipInfo(
                    f"00000/{basename}", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                output.writestr(info, data, compresslevel=9)


def build_manifest(
    output_dir: Path,
    labels: np.ndarray,
    subsets: dict[int, np.ndarray],
    *,
    expansion_seed: int,
    source_archive: Path,
    anchor_path: Path,
) -> dict[str, object]:
    """Build the machine-readable provenance record for all nested subsets."""
    records: dict[str, object] = {}
    for size, indices in sorted(subsets.items()):
        index_path = output_dir / f"e009_cifar10_subset_{size // 1000}k_indices.txt"
        archive_path = output_dir / f"e009_cifar10_subset_{size // 1000}k.zip"
        counts = np.bincount(labels[indices], minlength=10)
        records[str(size)] = {
            "index_manifest": index_path.name,
            "index_manifest_sha256": sha256_file(index_path),
            "indices_little_endian_int64_sha256": little_endian_int64_sha256(indices),
            "subset_archive": str(archive_path),
            "subset_archive_sha256": sha256_file(archive_path),
            "subset_archive_size_bytes": archive_path.stat().st_size,
            "class_distribution": {
                name: int(count) for name, count in zip(CLASS_NAMES, counts)
            },
        }
    return {
        "schema_version": 1,
        "experiment_id": "E009",
        "construction": {
            "method": "retain complete E005 1K anchor; shuffle each class-specific complement once; take deterministic prefixes to exact class targets; sort final indices",
            "expansion_seed": expansion_seed,
            "class_seed_rule": "expansion_seed + class_id",
            "nested": True,
            "class_balanced": True,
        },
        "anchor": {
            "path": str(anchor_path),
            "count": 1000,
            "text_sha256": sha256_file(anchor_path),
        },
        "source_archive": {
            "path": str(source_archive),
            "sha256": sha256_file(source_archive),
        },
        "label_sources": LABEL_FILE_HASHES,
        "subsets": records,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write stable indented JSON with a trailing newline."""
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
