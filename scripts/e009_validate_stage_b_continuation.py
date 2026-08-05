#!/usr/bin/env python3
"""Validate the completed E009 Stage B 13K-to-30K continuation."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import torch

from e009_stage_b_preflight import directory_manifest, sha256_file


def parse_args() -> argparse.Namespace:
    """Parse frozen continuation validation paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-a-before", type=Path, required=True)
    parser.add_argument("--edm-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Freeze the 13K..30K snapshot inventory and verify completion."""
    args = parse_args()
    sys.path.insert(0, str(args.edm_root.resolve()))
    from spectral_diffusion_playground.e009_warm_start import (
        state_digest,
        validate_extended_state,
    )

    expected = list(range(13000, 30001, 1000))
    expected_names = {f"network-snapshot-{kimg:06d}.pkl" for kimg in expected}
    observed_names = {
        path.name for path in args.output_dir.glob("network-snapshot-*.pkl")
    }
    if observed_names != expected_names:
        raise ValueError("Stage B snapshot inventory differs from 13K..30K")
    records = []
    for kimg in expected:
        path = args.output_dir / f"network-snapshot-{kimg:06d}.pkl"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing Stage B snapshot: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or "ema" not in payload:
            raise ValueError(f"Unreadable Stage B snapshot: {path}")
        if int(getattr(payload["ema"], "label_dim", -1)) != 0:
            raise ValueError(f"Conditional Stage B snapshot: {path}")
        for name, tensor in payload["ema"].state_dict().items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise ValueError(f"Nonfinite EMA tensor {name} in {path}")
        records.append(
            {
                "training_kimg": kimg,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    parent = records[0]
    if parent["sha256"] != (
        "6d181c0102e93cfe1c43005675e7c76e01fae18afd402337a35ebc8b2128371c"
    ):
        raise ValueError("Validated 13K snapshot was overwritten")
    parent_state_path = args.output_dir / "training-state-013000.pt"
    if sha256_file(parent_state_path) != (
        "8bb1aabceee959ce2478a108b27ad6b34313cf8329cba2b048c9446077a7a130"
    ):
        raise ValueError("Validated 13K training state was overwritten")
    state_path = args.output_dir / "training-state-030000.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_extended_state(state)
    if state["progress"]["cur_kimg"] != 30000:
        raise ValueError("Final Stage B state did not reach 30K")
    if state["warm_start"]["resume_kind"] != "extended_stage_b_state":
        raise ValueError("Final state lost exact Stage B resume provenance")

    stats_path = args.output_dir / "stats_stage_b_continuation.jsonl"
    losses = []
    for line in stats_path.read_text().splitlines():
        record = json.loads(line)
        losses.append(float(record["stage_b_loss_mean"]))
    if len(losses) != 17 or not all(math.isfinite(loss) for loss in losses):
        raise ValueError("Continuation loss records are incomplete or nonfinite")

    before = json.loads(args.stage_a_before.read_text())
    after = directory_manifest(args.stage_a_root)
    if before != after:
        raise ValueError("Stage A changed during the Stage B continuation")
    result = {
        "status": "pass",
        "start_kimg": 13000,
        "final_kimg": 30000,
        "expected_new_checkpoint_kimg": expected[1:],
        "snapshot_count": len(records),
        "snapshot_records": records,
        "final_training_state": {
            "path": str(state_path.resolve()),
            "size_bytes": state_path.stat().st_size,
            "sha256": sha256_file(state_path),
            "rng_state_digest": state_digest(state["rng_state"]),
        },
        "finite_loss": True,
        "loss_records": len(losses),
        "parent_13k_preserved": True,
        "stage_a_unchanged": True,
        "stage_a_manifest_sha256": after["manifest_sha256"],
        "evaluation_started": False,
        "e008_swaps_started": False,
    }
    path = args.output_dir / "e009_stage_b_continuation_validation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
