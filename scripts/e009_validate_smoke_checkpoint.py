#!/usr/bin/env python3
"""Validate the E009 smoke run's final unconditional EMA checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch


def find_final_snapshot(run_dirs: list[Path]) -> Path:
    """Return the highest-kimg nonempty snapshot across approved run directories."""
    snapshots = sorted(
        (
            path
            for run_dir in run_dirs
            for path in run_dir.glob("network-snapshot-*.pkl")
            if path.stat().st_size > 0
        ),
        key=lambda path: (int(path.stem.rsplit("-", 1)[1]), str(path)),
    )
    if not snapshots:
        raise ValueError("No nonempty network snapshot was produced")
    return snapshots[-1]


def validate_ema(ema: torch.nn.Module) -> dict[str, Any]:
    """Require an unconditional network with finite floating-point state."""
    label_dim = int(getattr(ema, "label_dim", -1))
    if label_dim != 0:
        raise ValueError(
            f"Smoke checkpoint is not unconditional: label_dim={label_dim}"
        )
    parameter_count = 0
    for name, tensor in ema.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        parameter_count += tensor.numel()
    if parameter_count == 0:
        raise ValueError("EMA checkpoint contains no state")
    return {"label_dim": label_dim, "state_element_count": parameter_count}


def parse_args() -> argparse.Namespace:
    """Parse smoke-validation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--edm-root", type=Path, required=True)
    parser.add_argument("--expected-kimg", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Load and validate the final snapshot, then write a compact record."""
    args = parse_args()
    if not args.edm_root.is_dir():
        raise ValueError(f"EDM source directory is missing: {args.edm_root}")
    sys.path.insert(0, str(args.edm_root.resolve()))
    snapshot = find_final_snapshot([args.scratch_dir, args.final_dir])
    observed_kimg = int(snapshot.stem.rsplit("-", 1)[1])
    if observed_kimg != args.expected_kimg:
        raise ValueError(
            f"Smoke ended at {observed_kimg} kimg, expected {args.expected_kimg}"
        )
    with snapshot.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise ValueError("Snapshot does not contain the expected EMA payload")
    details = validate_ema(payload["ema"])
    record = {
        "status": "pass",
        "snapshot": str(snapshot.resolve()),
        "snapshot_size_bytes": snapshot.stat().st_size,
        "observed_kimg": observed_kimg,
        "finite_state": True,
        **details,
    }
    output_dir = args.final_dir if args.final_dir.is_dir() else args.scratch_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e009_smoke_validation.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
