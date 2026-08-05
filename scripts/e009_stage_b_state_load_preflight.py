#!/usr/bin/env python3
"""Load and validate the 13K Stage B continuation state without training."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from e009_stage_b_preflight import directory_manifest, sha256_file


def parse_args() -> argparse.Namespace:
    """Parse the frozen state-load-only preflight arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-a-before", type=Path, required=True)
    parser.add_argument("--edm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def zero_module(module: torch.nn.Module) -> None:
    """Erase a destination module before testing parameter restoration."""
    with torch.no_grad():
        for tensor in module.state_dict().values():
            tensor.zero_()


def main() -> None:
    """Restore every serialized component and stop before an optimizer step."""
    args = parse_args()
    sys.path.insert(0, str(args.edm_root.resolve()))
    from torch_utils import misc

    from spectral_diffusion_playground.e009_warm_start import (
        StatefulInfiniteSampler,
        restore_rng_state,
        state_digest,
        validate_extended_state,
    )

    state_before = (args.state.stat().st_size, args.state.stat().st_mtime_ns)
    snapshot_before = (
        args.snapshot.stat().st_size,
        args.snapshot.stat().st_mtime_ns,
    )
    if sha256_file(args.state) != args.state_sha256:
        raise ValueError("13K extended-state hash mismatch")
    if sha256_file(args.snapshot) != args.snapshot_sha256:
        raise ValueError("13K snapshot hash mismatch")

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    validate_extended_state(state)
    progress = state["progress"]
    if progress["cur_kimg"] != 13000 or progress["cur_nimg"] != 13_000_000:
        raise ValueError("13K progress counters are incorrect")
    if state["warm_start"]["seed"] != 1:
        raise ValueError("13K state seed is not 1")
    if progress["lineage_tick"] != 1:
        raise ValueError("13K lineage counter is incorrect")

    with args.snapshot.open("rb") as handle:
        snapshot = pickle.load(handle)
    if not isinstance(snapshot, dict) or "ema" not in snapshot:
        raise ValueError("13K snapshot does not contain EMA")
    if state_digest(state["ema"]) != state_digest(snapshot["ema"]):
        raise ValueError("13K state and snapshot EMA differ")

    network = copy.deepcopy(state["net"])
    zero_module(network)
    misc.copy_params_and_buffers(state["net"], network, require_all=True)
    if state_digest(network) != state_digest(state["net"]):
        raise ValueError("Network restoration failed")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    optimizer.load_state_dict(state["optimizer_state"])
    if state_digest(optimizer.state_dict()) != state_digest(state["optimizer_state"]):
        raise ValueError("Optimizer restoration failed")
    ema = copy.deepcopy(state["ema"])
    zero_module(ema)
    misc.copy_params_and_buffers(state["ema"], ema, require_all=True)
    if state_digest(ema) != state_digest(state["ema"]):
        raise ValueError("EMA restoration failed")

    sampler = StatefulInfiniteSampler(list(range(5000)), seed=1)
    generator = torch.Generator()
    generator.manual_seed(1)
    restore_rng_state(state["rng_state"], sampler, generator)
    restored_rng_digest = state_digest(state["rng_state"])
    if state_digest(np.random.get_state()) != state_digest(state["rng_state"]["numpy"]):
        raise ValueError("NumPy RNG restoration failed")
    if state_digest(torch.get_rng_state()) != state_digest(
        state["rng_state"]["torch_cpu"]
    ):
        raise ValueError("Torch CPU RNG restoration failed")
    if state_digest(torch.cuda.get_rng_state_all()) != state_digest(
        state["rng_state"]["torch_cuda_all"]
    ):
        raise ValueError("Torch CUDA RNG restoration failed")

    if sha256_file(args.state) != args.state_sha256:
        raise ValueError("13K state changed during load-only preflight")
    if sha256_file(args.snapshot) != args.snapshot_sha256:
        raise ValueError("13K snapshot changed during load-only preflight")
    state_after = (args.state.stat().st_size, args.state.stat().st_mtime_ns)
    snapshot_after = (
        args.snapshot.stat().st_size,
        args.snapshot.stat().st_mtime_ns,
    )
    if state_after != state_before or snapshot_after != snapshot_before:
        raise ValueError("13K artifact metadata changed during preflight")
    before = json.loads(args.stage_a_before.read_text())
    after = directory_manifest(args.stage_a_root)
    if before != after:
        raise ValueError("Stage A changed during state-load-only preflight")

    result = {
        "status": "pass",
        "optimizer_steps_taken": 0,
        "state_sha256": args.state_sha256,
        "snapshot_sha256": args.snapshot_sha256,
        "network_restored": True,
        "optimizer_restored": True,
        "ema_restored": True,
        "progress_kimg": 13000,
        "seed": 1,
        "numpy_rng_restored": True,
        "torch_cpu_rng_restored": True,
        "torch_cuda_rng_restored": True,
        "sampler_rng_restored": True,
        "dataloader_generator_restored": True,
        "restored_rng_state_digest": restored_rng_digest,
        "next_output_kimg": 14000,
        "parent_artifacts_unchanged": True,
        "stage_a_unchanged": True,
        "stage_a_manifest_sha256": after["manifest_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
