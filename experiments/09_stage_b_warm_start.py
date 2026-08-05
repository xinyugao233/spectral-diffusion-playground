#!/usr/bin/env python3
"""Run the frozen E009 Stage B warm-start extension under Slurm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the frozen Stage B training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def stage_continuation_outputs(
    run_dir: Path,
    final_dir: Path,
    *,
    start_kimg: int,
) -> None:
    """Stage only post-parent artifacts without rewriting the validated 13K files."""
    from zwlab_edm.common import copy_file, latest_file

    for source in sorted(run_dir.glob("network-snapshot-*.pkl")):
        kimg = int(source.stem.rsplit("-", 1)[1])
        if kimg <= start_kimg:
            continue
        destination = final_dir / source.name
        if destination.exists():
            raise ValueError(
                f"Refusing to overwrite continuation snapshot: {destination}"
            )
        copy_file(source, destination)
    latest_state = latest_file(run_dir, "training-state-*.pt")
    if latest_state is None:
        raise ValueError("Continuation produced no training state")
    if int(latest_state.stem.rsplit("-", 1)[1]) <= start_kimg:
        raise ValueError("Continuation did not advance beyond its parent state")
    destination = final_dir / latest_state.name
    if destination.exists():
        raise ValueError(f"Refusing to overwrite continuation state: {destination}")
    copy_file(latest_state, destination)
    for filename in (
        "config_used_continuation.yaml",
        "training_options_continuation.json",
        "run_manifest_continuation.json",
        "stage_b_resume_initialization.json",
        "stats_stage_b_continuation.jsonl",
    ):
        source = run_dir / filename
        if source.is_file():
            copy_file(source, final_dir / filename)


def main() -> None:
    """Load separate parent states and execute the warm-start loop."""
    args = parse_args()
    from zwlab_edm.common import (
        copy_file,
        ensure_dir,
        final_run_dir,
        load_config,
        require_slurm,
        scratch_run_dir,
        set_job_env,
        write_json,
    )

    require_slurm()
    set_job_env()
    config = load_config(args.config)
    edm_root = Path(config["experiment"]["edm_root"]).resolve()
    sys.path.insert(0, str(edm_root))

    import torch
    from torch_utils import distributed as dist
    from zwlab_edm.train_subset_sigma import (
        build_training_kwargs,
        resolve_sigma_list,
        stage_outputs,
    )

    from spectral_diffusion_playground.e009_warm_start import (
        warm_start_training_loop,
    )

    if config["experiment"].get("warm_start") is not True:
        raise ValueError("Stage B requires experiment.warm_start=true")
    warm = config["warm_start"]
    if warm.get("enabled") is not True:
        raise ValueError("Stage B requires warm_start.enabled=true")

    resume_kind = str(warm.get("resume_kind", "stage_a_warm_start"))
    execution_commit = os.environ.get("E009_REPO_COMMIT")
    if not execution_commit:
        raise ValueError("E009_REPO_COMMIT is required for Stage B provenance")
    scratch_dir = ensure_dir(scratch_run_dir(config))
    final_dir = ensure_dir(final_run_dir(config))
    run_dir = scratch_dir
    continuation = resume_kind == "extended_stage_b_state"
    config_filename = (
        "config_used_continuation.yaml" if continuation else "config_used.yaml"
    )
    options_filename = (
        "training_options_continuation.json"
        if continuation
        else "training_options.json"
    )
    manifest_filename = (
        "run_manifest_continuation.json" if continuation else "run_manifest.json"
    )
    copy_file(args.config, run_dir / config_filename)

    kwargs = build_training_kwargs(config, run_dir)
    kwargs["data_loader_kwargs"]["num_workers"] = int(config["training"]["workers"])
    kwargs["parent_training_state"] = Path(warm["parent_training_state"])
    kwargs["parent_training_state_sha256"] = warm["parent_training_state_sha256"]
    kwargs["parent_ema_snapshot"] = Path(warm["parent_ema_snapshot"])
    kwargs["parent_ema_snapshot_sha256"] = warm["parent_ema_snapshot_sha256"]
    kwargs["resume_kind"] = resume_kind
    kwargs["execution_commit"] = execution_commit
    kwargs["start_kimg"] = int(warm["start_kimg"])
    write_json(
        run_dir / options_filename,
        json.loads(json.dumps(kwargs, default=str)),
    )

    dist.init()
    result = warm_start_training_loop(**kwargs)
    if dist.get_rank() == 0:
        sigma_list = resolve_sigma_list(config)
        write_json(
            run_dir / manifest_filename,
            {
                "name": config["experiment"]["name"],
                "warm_start": True,
                "exact_stage_a_continuation": False,
                "exact_stage_b_resume": continuation,
                "validated_parent_implementation": (
                    "9e5782f09a3e024c298dc5ce8da1c0f44c9b4fbd"
                ),
                "execution_commit": execution_commit,
                "resume_kind": resume_kind,
                "stage_b_seed": int(config["experiment"]["seed"]),
                "start_kimg": int(warm["start_kimg"]),
                "duration_kimg": int(config["training"]["duration_kimg"]),
                "subset_size": int(config["dataset"]["subset_size"]),
                "parent_training_state_sha256": warm["parent_training_state_sha256"],
                "parent_ema_snapshot_sha256": warm["parent_ema_snapshot_sha256"],
                "state_schema_version": int(warm["state_schema_version"]),
                "sigma_count": len(sigma_list),
                "result": result,
            },
        )
        if continuation:
            stage_continuation_outputs(
                run_dir,
                final_dir,
                start_kimg=int(warm["start_kimg"]),
            )
        else:
            stage_outputs(run_dir, scratch_dir, final_dir, config)
            copy_file(
                run_dir / "warm_start_initialization.json",
                final_dir / "warm_start_initialization.json",
            )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
