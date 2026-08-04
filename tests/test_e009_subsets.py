"""Tests for E009 deterministic nested subset construction."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.e009_subsets import (
    build_nested_class_balanced_subsets,
    little_endian_int64_sha256,
    materialize_subset_archive,
    write_indices,
)


class NestedSubsetTests(unittest.TestCase):
    """Require balance, nesting, determinism, and exact source bytes."""

    def setUp(self) -> None:
        self.labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
        self.anchor = np.concatenate(
            [
                np.arange(class_id * 5000, class_id * 5000 + 100)
                for class_id in range(10)
            ]
        )

    def test_subsets_are_balanced_nested_and_retain_anchor(self) -> None:
        subsets = build_nested_class_balanced_subsets(
            self.labels, self.anchor, [2000, 5000, 10000], expansion_seed=9
        )
        previous = set(self.anchor)
        for size, values in subsets.items():
            with self.subTest(size=size):
                self.assertEqual(len(values), size)
                self.assertEqual(len(np.unique(values)), size)
                np.testing.assert_array_equal(
                    np.bincount(self.labels[values], minlength=10),
                    np.full(10, size // 10),
                )
                self.assertTrue(previous.issubset(set(values)))
                previous = set(values)

    def test_same_seed_is_byte_reproducible(self) -> None:
        first = build_nested_class_balanced_subsets(
            self.labels, self.anchor, [2000], expansion_seed=9
        )[2000]
        second = build_nested_class_balanced_subsets(
            self.labels, self.anchor, [2000], expansion_seed=9
        )[2000]
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            little_endian_int64_sha256(first), little_endian_int64_sha256(second)
        )

    def test_text_manifest_has_stable_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "indices.txt"
            write_indices(path, np.asarray([1, 4, 8], dtype=np.int64))
            self.assertEqual(path.read_bytes(), b"1\n4\n8\n")

    def test_archive_contains_only_exact_selected_png_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            destination = root / "subset.zip"
            expected = {0: b"zero", 2: b"two"}
            with zipfile.ZipFile(source, "w") as archive:
                for index, data in {0: b"zero", 1: b"one", 2: b"two"}.items():
                    archive.writestr(f"00000/img{index:08d}.png", data)
            materialize_subset_archive(
                source, destination, np.asarray([0, 2], dtype=np.int64)
            )
            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()
                self.assertEqual(
                    names,
                    ["00000/img00000000.png", "00000/img00000002.png"],
                )
                for index, data in expected.items():
                    self.assertEqual(archive.read(f"00000/img{index:08d}.png"), data)
            first_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
            materialize_subset_archive(
                source, root / "subset_again.zip", np.asarray([0, 2], dtype=np.int64)
            )
            self.assertEqual(
                first_hash,
                hashlib.sha256((root / "subset_again.zip").read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
