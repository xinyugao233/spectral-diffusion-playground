#!/usr/bin/env python3
"""Validate the completed E009 Stage B 12K-to-13K warm-start smoke."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

from e009_stage_b_preflight import directory_manifest, sha256_file


def finite_module(module: torch.nn.Module, label: str) -> int:
    """Require finite state and return the number of elements."""
    count = 0
    for name, tensor in module.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite {label} tensor: {name}")
        count += tensor.numel()
    if count == 0:
        raise ValueError(f"{label} contains no state")
    return count


def parse_args() -> argparse.Namespace:
    """Parse smoke-validation paths and frozen identities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-a-before", type=Path, required=True)
    parser.add_argument("--edm-root", type=Path, required=True)
    parser.add_argument("--expected-parent-state-sha256", required=True)
    parser.add_argument("--expected-parent-ema-sha256", required=True)
    parser.add_argument("--expected-start-kimg", type=int, default=12000)
    parser.add_argument("--expected-final-kimg", type=int, default=13000)
    return parser.parse_args()


def main() -> None:
    """Validate artifacts, extended state, finite loss, and parent immutability."""
    args = parse_args()
    sys.path.insert(0, str(args.edm_root.resolve()))
    from spectral_diffusion_playground.e009_warm_start import (
        validate_extended_state,
    )

    snapshot = args.final_dir / "network-snapshot-013000.pkl"
    state_path = args.final_dir / "training-state-013000.pt"
    initialization_path = args.final_dir / "warm_start_initialization.json"
    manifest_path = args.final_dir / "run_manifest.json"
    stats_path = args.final_dir / "stats.jsonl"
    for path in (snapshot, state_path, initialization_path, manifest_path, stats_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty Stage B smoke artifact: {path}")

    initialization = json.loads(initialization_path.read_text())
    if initialization["warm_start"] is not True:
        raise ValueError("Initialization does not declare warm_start=true")
    if initialization["exact_stage_a_continuation"] is not False:
        raise ValueError("Initialization incorrectly claims exact continuation")
    if (
        initialization["seed"] != 1
        or not initialization["seed_derivation_reproducible"]
    ):
        raise ValueError("Stage B RNG initialization was not reproducible")
    if initialization["parent_training_state_sha256"] != (
        args.expected_parent_state_sha256
    ):
        raise ValueError("Parent training-state identity changed")
    if initialization["parent_ema_snapshot_sha256"] != (
        args.expected_parent_ema_sha256
    ):
        raise ValueError("Parent EMA identity changed")
    for field in (
        "network_loaded_exactly",
        "optimizer_loaded_exactly",
        "ema_loaded_exactly",
    ):
        if initialization[field] is not True:
            raise ValueError(f"Warm-start loading gate failed: {field}")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_extended_state(state)
    progress = state["progress"]
    if progress["start_kimg"] != args.expected_start_kimg:
        raise ValueError("Extended state start counter changed")
    if progress["cur_kimg"] != args.expected_final_kimg:
        raise ValueError("Extended state did not reach exactly 13K")
    if progress["cur_nimg"] != args.expected_final_kimg * 1000:
        raise ValueError("Extended state image counter is inconsistent")
    if state["warm_start"]["seed"] != 1:
        raise ValueError("Extended state seed changed")
    network_elements = finite_module(state["net"], "training network")
    ema_elements = finite_module(state["ema"], "EMA")
    if not state["optimizer_state"].get("state"):
        raise ValueError("Extended state optimizer is empty")

    with snapshot.open("rb") as handle:
        snapshot_payload = pickle.load(handle)
    if not isinstance(snapshot_payload, dict) or "ema" not in snapshot_payload:
        raise ValueError("13K snapshot has no EMA")
    if int(getattr(snapshot_payload["ema"], "label_dim", -1)) != 0:
        raise ValueError("13K snapshot is not unconditional")
    finite_module(snapshot_payload["ema"], "snapshot EMA")

    losses = []
    for line in stats_path.read_text().splitlines():
        record = json.loads(line)
        if "stage_b_loss_mean" in record:
            losses.append(float(record["stage_b_loss_mean"]))
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("Stage B smoke loss is missing or nonfinite")

    before = json.loads(args.stage_a_before.read_text())
    after = directory_manifest(args.stage_a_root)
    if before != after:
        raise ValueError("Stage A artifacts changed during the Stage B smoke")

    run_manifest = json.loads(manifest_path.read_text())
    if run_manifest["warm_start"] is not True:
        raise ValueError("Run manifest does not declare warm_start=true")
    if run_manifest["result"]["final_kimg"] != args.expected_final_kimg:
        raise ValueError("Run manifest final exposure changed")

    result: dict[str, Any] = {
        "status": "pass",
        "warm_start": True,
        "exact_stage_a_continuation": False,
        "seed": 1,
        "start_kimg": args.expected_start_kimg,
        "final_kimg": args.expected_final_kimg,
        "finite_loss": True,
        "loss_records": len(losses),
        "network_state_elements": network_elements,
        "ema_state_elements": ema_elements,
        "state_schema_version": state["state_schema_version"],
        "serialized_rng_fields": sorted(state["rng_state"]),
        "stage_a_unchanged": True,
        "stage_a_manifest_sha256": after["manifest_sha256"],
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_sha256": sha256_file(snapshot),
        "training_state_path": str(state_path.resolve()),
        "training_state_sha256": sha256_file(state_path),
        "no_evaluation_or_swap": True,
    }
    output_path = args.final_dir / "e009_stage_b_smoke_validation.json"
    (args.final_dir / "stage_a_identity_before.json").write_text(
        json.dumps(before, indent=2, sort_keys=True) + "\n"
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
