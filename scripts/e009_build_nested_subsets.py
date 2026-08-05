#!/usr/bin/env python3
"""Build E009 nested class-balanced manifests and deterministic ZIP archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral_diffusion_playground.e009_subsets import (
    ANCHOR_TEXT_SHA256,
    SOURCE_ARCHIVE_SHA256,
    build_manifest,
    build_nested_class_balanced_subsets,
    load_cifar10_labels,
    load_indices,
    materialize_subset_archive,
    sha256_file,
    write_indices,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    """Parse deterministic subset-construction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--anchor-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000, 10000])
    parser.add_argument("--expansion-seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    """Create all Stage A subset artifacts and their provenance manifest."""
    args = parse_args()
    if sha256_file(args.anchor_indices) != ANCHOR_TEXT_SHA256:
        raise ValueError("Anchor subset hash does not match the frozen E005 1K set")
    if sha256_file(args.source_archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            "Source archive hash does not match the frozen CIFAR-10 archive"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_cifar10_labels(args.label_root)
    anchor = load_indices(args.anchor_indices)
    subsets = build_nested_class_balanced_subsets(
        labels,
        anchor,
        args.sizes,
        expansion_seed=args.expansion_seed,
    )
    for size, indices in sorted(subsets.items()):
        stem = f"e009_cifar10_subset_{size // 1000}k"
        write_indices(args.output_dir / f"{stem}_indices.txt", indices)
        materialize_subset_archive(
            args.source_archive,
            args.output_dir / f"{stem}.zip",
            indices,
        )
    manifest = build_manifest(
        args.output_dir,
        labels,
        subsets,
        expansion_seed=args.expansion_seed,
        source_archive=args.source_archive,
        anchor_path=args.anchor_indices,
    )
    write_manifest(args.output_dir / "e009_nested_subsets_manifest.json", manifest)
    print("subset_preparation=pass")


if __name__ == "__main__":
    main()
