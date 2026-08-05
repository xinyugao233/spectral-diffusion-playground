#!/usr/bin/env python3
"""Run the frozen E009 Stage B warm-start extension under Slurm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the frozen Stage B training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


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

    scratch_dir = ensure_dir(scratch_run_dir(config))
    final_dir = ensure_dir(final_run_dir(config))
    run_dir = scratch_dir
    copy_file(args.config, run_dir / "config_used.yaml")

    kwargs = build_training_kwargs(config, run_dir)
    kwargs["data_loader_kwargs"]["num_workers"] = int(config["training"]["workers"])
    kwargs["parent_training_state"] = Path(warm["parent_training_state"])
    kwargs["parent_training_state_sha256"] = warm["parent_training_state_sha256"]
    kwargs["parent_ema_snapshot"] = Path(warm["parent_ema_snapshot"])
    kwargs["parent_ema_snapshot_sha256"] = warm["parent_ema_snapshot_sha256"]
    kwargs["start_kimg"] = int(warm["start_kimg"])
    write_json(
        run_dir / "training_options.json",
        json.loads(json.dumps(kwargs, default=str)),
    )

    dist.init()
    result = warm_start_training_loop(**kwargs)
    if dist.get_rank() == 0:
        sigma_list = resolve_sigma_list(config)
        write_json(
            run_dir / "run_manifest.json",
            {
                "name": config["experiment"]["name"],
                "warm_start": True,
                "exact_stage_a_continuation": False,
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
        stage_outputs(run_dir, scratch_dir, final_dir, config)
        copy_file(
            run_dir / "warm_start_initialization.json",
            final_dir / "warm_start_initialization.json",
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
