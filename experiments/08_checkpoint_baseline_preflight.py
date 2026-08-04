#!/usr/bin/env python3
"""E008 baseline-only checkpoint preflight.

This entry point inventories existing matched-training snapshots, evaluates
only complete no-swap Euler trajectories on disjoint pilot seeds, and applies
the prospectively frozen baseline eligibility and model-pair selection rules.
It contains no donor-model or swap-window execution interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

from _bootstrap import REPO_ROOT
from spectral_diffusion_playground.e006_transition_swaps import (
    SIGMA_GRID,
    TERMINAL_SIGMA,
    generated_sample_hash,
)
from spectral_diffusion_playground.e008_checkpoint_preflight import (
    CONFIRMATORY_SEEDS,
    ELIGIBLE_COUNT_MAX,
    ELIGIBLE_COUNT_MIN,
    EXPERIMENT_ID,
    PILOT_SEEDS,
    candidate_is_eligible,
    discover_checkpoint_paths,
    merge_resume_rows,
    nearest_two_cpu,
    parse_training_kimg,
    read_csv_rows,
    select_model_pair,
    sha256_file,
    summarize_candidate,
    validate_no_swap_only,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/e008_checkpoint_preflight.json"
DEFAULT_OUTPUT = Path("/home/xggh8/data/zw-lab/e008_checkpoint_preflight")
INVENTORY_FILENAME = "candidate_checkpoint_inventory.csv"
POOL_MANIFEST_FILENAME = "candidate_pool_manifest.json"
PER_SAMPLE_FILENAME = "pilot_per_sample.csv"
SUMMARY_FILENAME = "pilot_checkpoint_summary.csv"
FAILURES_FILENAME = "pilot_failures.csv"
PAIR_FILENAME = "selected_model_pair.json"
OUTCOME_FILENAME = "preflight_outcome.json"
VALIDATION_FILENAME = "preflight_validation.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one bounded baseline-preflight mode."""
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inventory-only", action="store_true")
    modes.add_argument("--run-pilot", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--edm-1k-root", type=Path)
    parser.add_argument("--edm-50k-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reference-subset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edm-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--nn-batch-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Execute exactly one inventory, pilot, summary, or validation mode."""
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing to run E008 checkpoint work outside Slurm")
    config = load_and_validate_config(args.config)
    resolve_args(args, config)
    validate_args(args)
    if args.inventory_only:
        inventory_candidates(args, config)
    elif args.run_pilot:
        run_pilot(args, config)
    elif args.summarize:
        summarize(args, config)
    else:
        validation = validate_outputs(args.output_dir, config, require_complete=True)
        print(json.dumps(validation, indent=2))


def load_and_validate_config(path: Path) -> dict[str, Any]:
    """Load the frozen config and reject scientific drift."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError("E008 preflight experiment ID mismatch")
    if config["scientific_scope"] != {
        "baseline_only": True,
        "donor_models_allowed": False,
        "swap_windows_allowed": False,
        "e008_executed": False,
    }:
        raise RuntimeError("E008 preflight scientific scope mismatch")
    if config["pilot_seeds"] != {
        "start": 10000,
        "stop_inclusive": 10127,
        "count": 128,
    }:
        raise RuntimeError("Pilot seed contract mismatch")
    if config["future_confirmatory_seeds"]["used_by_preflight"]:
        raise RuntimeError("Confirmatory seeds may not be used by preflight")
    if config["eligibility"]["count_interval_inclusive"] != [13, 115]:
        raise RuntimeError("Eligibility count interval mismatch")
    sampler = config["sampler"]
    if sampler["algorithm"] != "pure_euler" or sampler["heun_correction"]:
        raise RuntimeError("Sampler must remain pure Euler")
    validate_no_swap_only()
    return config


def resolve_args(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Resolve optional execution paths from the frozen config."""
    args.edm_1k_root = args.edm_1k_root or Path(config["candidate_roots"]["edm_1k"])
    args.edm_50k_root = args.edm_50k_root or Path(config["candidate_roots"]["edm_50k"])
    args.dataset_root = args.dataset_root or Path(config["dataset"]["archive"])
    args.reference_subset = args.reference_subset or (
        REPO_ROOT / config["memorization"]["reference_subset"]
    )
    args.edm_root = args.edm_root or Path(config["execution"]["edm_root"])
    args.batch_size = args.batch_size or int(config["execution"]["batch_size"])
    args.nn_batch_size = args.nn_batch_size or int(config["execution"]["nn_batch_size"])


def validate_args(args: argparse.Namespace) -> None:
    """Validate operational arguments without weakening frozen choices."""
    if args.batch_size <= 0 or args.nn_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.smoke and not args.run_pilot:
        raise ValueError("--smoke is valid only with --run-pilot")
    if args.resume and not args.run_pilot:
        raise ValueError("--resume is valid only with --run-pilot")
    if args.run_pilot and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        if not args.resume:
            raise ValueError("Full pilot requires explicit --resume after pool freeze")
    pilot_overlap = set(PILOT_SEEDS).intersection(CONFIRMATORY_SEEDS)
    if pilot_overlap:
        raise RuntimeError(f"Pilot/confirmatory seed overlap: {pilot_overlap}")


def inventory_candidates(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Inventory every candidate and freeze accepted/rejected checkpoint identities."""
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Inventory output directory already exists: {output_dir}")
    provenance = validate_static_provenance(args, config)
    staging_dir = output_dir.with_name(
        f"{output_dir.name}.inventory_{os.environ['SLURM_JOB_ID']}"
    )
    if staging_dir.exists():
        raise RuntimeError(f"Inventory staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for role, root, subset_size in (
        ("edm_1k", args.edm_1k_root.resolve(), 1000),
        ("edm_50k", args.edm_50k_root.resolve(), 50000),
    ):
        accepted_paths, malformed_paths = discover_checkpoint_paths(root)
        config_source = find_training_config(root)
        config_hash = sha256_file(config_source) if config_source else ""
        for path in accepted_paths:
            record = base_inventory_record(
                role, path, subset_size, config_source, config_hash
            )
            try:
                record.update(inspect_checkpoint(path, args.edm_root))
                record["inventory_status"] = "accepted"
                record["rejection_reason"] = ""
            except Exception as error:  # retained explicitly in the frozen pool
                record.update(
                    {
                        "checkpoint_sha256": sha256_file(path),
                        "architecture_identity": "",
                        "ema_status": "unknown",
                        "label_conditioning_status": "unknown",
                        "loadability_status": "failed",
                        "inventory_status": "rejected",
                        "rejection_reason": f"{type(error).__name__}: {error}",
                    }
                )
            records.append(record)
        for path in malformed_paths:
            records.append(
                {
                    **base_inventory_record(
                        role, path, subset_size, config_source, config_hash
                    ),
                    "training_kimg": "",
                    "checkpoint_sha256": sha256_file(path),
                    "architecture_identity": "",
                    "ema_status": "unknown",
                    "label_conditioning_status": "unknown",
                    "loadability_status": "not_attempted",
                    "inventory_status": "rejected",
                    "rejection_reason": "malformed_snapshot_filename",
                }
            )
    records.sort(
        key=lambda row: (
            str(row["model_role"]),
            int(row["training_kimg"]) if row["training_kimg"] != "" else 10**12,
            str(row["checkpoint_filename"]),
        )
    )
    accepted_architectures = {
        str(row["architecture_identity"])
        for row in records
        if row["inventory_status"] == "accepted"
    }
    if len(accepted_architectures) != 1:
        raise RuntimeError(
            "Accepted candidate architectures are not identical: "
            f"{sorted(accepted_architectures)}"
        )
    inventory_path = staging_dir / INVENTORY_FILENAME
    write_csv(inventory_path, records, inventory_header())
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_frozen_before_pilot": True,
        "pilot_started": False,
        "scientific_scope": config["scientific_scope"],
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "repository_commit": git_commit(),
        "edm_commit": repository_commit(args.edm_root),
        "provenance": provenance,
        "inventory": {
            "path": str(output_dir / INVENTORY_FILENAME),
            "sha256": sha256_file(inventory_path),
            "row_count": len(records),
            "accepted_count": sum(
                row["inventory_status"] == "accepted" for row in records
            ),
            "rejected_count": sum(
                row["inventory_status"] == "rejected" for row in records
            ),
            "records": records,
        },
        "pilot_seeds": list(PILOT_SEEDS),
        "future_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
    }
    dump_json(staging_dir / POOL_MANIFEST_FILENAME, manifest)
    staging_dir.replace(output_dir)
    print(json.dumps({"status": "pool_frozen", **manifest["inventory"]}, indent=2))


def base_inventory_record(
    role: str,
    path: Path,
    subset_size: int,
    config_source: Path | None,
    config_hash: str,
) -> dict[str, object]:
    """Build filesystem and training-run fields shared by every candidate."""
    stat = path.stat()
    return {
        "model_role": role,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_filename": path.name,
        "checkpoint_sha256": "",
        "training_kimg": (
            parse_training_kimg(path.name)
            if path.name.endswith(".pkl")
            and path.name.startswith("network-snapshot-")
            and path.name[17:23].isdigit()
            else ""
        ),
        "file_size": stat.st_size,
        "modification_time": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "architecture_identity": "",
        "training_subset_size": subset_size,
        "training_configuration_source": str(config_source) if config_source else "",
        "training_configuration_sha256": config_hash,
        "ema_status": "unknown",
        "label_conditioning_status": "unknown",
        "loadability_status": "not_attempted",
        "inventory_status": "pending",
        "rejection_reason": "",
    }


def inspect_checkpoint(path: Path, edm_root: Path) -> dict[str, object]:
    """Hash and CPU-load one EMA checkpoint to record its model identity."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "ema" not in payload:
        raise RuntimeError("checkpoint payload has no ema network")
    network = payload["ema"].cpu().eval()
    label_dim = int(getattr(network, "label_dim", -1))
    if label_dim != 0:
        raise RuntimeError(f"checkpoint is not unconditional: label_dim={label_dim}")
    identity = {
        "class": f"{type(network).__module__}.{type(network).__qualname__}",
        "img_resolution": int(getattr(network, "img_resolution", -1)),
        "img_channels": int(getattr(network, "img_channels", -1)),
        "label_dim": label_dim,
        "sigma_min": float(getattr(network, "sigma_min", float("nan"))),
        "sigma_max": float(getattr(network, "sigma_max", float("nan"))),
    }
    if identity["img_resolution"] != 32 or identity["img_channels"] != 3:
        raise RuntimeError(f"unexpected architecture identity: {identity}")
    return {
        "checkpoint_sha256": sha256_file(path),
        "architecture_identity": json.dumps(identity, sort_keys=True),
        "ema_status": "present",
        "label_conditioning_status": "unconditional",
        "loadability_status": "pass",
    }


def find_training_config(root: Path) -> Path | None:
    """Return the run's frozen training configuration when present."""
    for name in ("config_used.yaml", "training_options.json", "config.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def validate_static_provenance(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, object]:
    """Verify frozen dataset, subset, repository, and source identities."""
    subset = np.loadtxt(args.reference_subset, dtype=np.int64)
    if subset.shape != (1000,) or len(np.unique(subset)) != 1000:
        raise RuntimeError("Frozen reference subset is invalid")
    subset_int64 = hashlib.sha256(
        np.asarray(subset, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    observed = {
        "dataset_archive_sha256": sha256_file(args.dataset_root),
        "reference_subset_text_sha256": sha256_file(args.reference_subset),
        "reference_subset_int64_sha256": subset_int64,
    }
    expected = {
        "dataset_archive_sha256": config["dataset"]["archive_sha256"],
        "reference_subset_text_sha256": config["memorization"][
            "reference_subset_text_sha256"
        ],
        "reference_subset_int64_sha256": config["memorization"][
            "reference_subset_int64_sha256"
        ],
    }
    if observed != expected:
        raise RuntimeError(f"Static provenance mismatch: {observed} != {expected}")
    return observed


def run_pilot(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Evaluate only no-swap baselines for the frozen checkpoint pool."""
    validate_no_swap_only()
    pool_dir = DEFAULT_OUTPUT.resolve() if args.smoke else args.output_dir.resolve()
    manifest_path = pool_dir / POOL_MANIFEST_FILENAME
    inventory_path = pool_dir / INVENTORY_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = read_csv_rows(inventory_path)
    verify_pool_manifest(manifest, inventory_path, config)
    accepted = [row for row in inventory if row["inventory_status"] == "accepted"]
    if args.smoke:
        accepted = [
            next(row for row in accepted if row["model_role"] == role)
            for role in ("edm_1k", "edm_50k")
        ]
        seeds = PILOT_SEEDS[:2]
    else:
        seeds = PILOT_SEEDS
    output_dir = args.output_dir.resolve()
    if args.smoke:
        if output_dir.exists():
            raise RuntimeError(f"Smoke output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        copy_pool_files(pool_dir, output_dir)
    elif not args.resume:
        raise RuntimeError("Full pilot requires --resume against the frozen pool")
    elif output_dir != pool_dir:
        raise RuntimeError("Full pilot output must be the frozen pool directory")

    device = resolve_device(args.device)
    configure_determinism()
    subset_indices = np.loadtxt(args.reference_subset, dtype=np.int64)
    reference = load_reference_images(args.dataset_root, subset_indices)
    existing = read_csv_rows(output_dir / PER_SAMPLE_FILENAME)
    all_rows: list[dict[str, object]] = [normalize_resume_row(row) for row in existing]
    existing_keys = {
        (str(row["checkpoint_sha256"]), int(row["sample_seed"])) for row in all_rows
    }
    for candidate in accepted:
        missing = [
            seed
            for seed in seeds
            if (candidate["checkpoint_sha256"], seed) not in existing_keys
        ]
        if not missing:
            continue
        candidate_rows = evaluate_candidate(
            candidate=candidate,
            seeds=missing,
            reference=reference,
            reference_indices=subset_indices,
            edm_root=args.edm_root,
            device=device,
            nn_batch_size=args.nn_batch_size,
            orchestration_batch_size=args.batch_size,
        )
        if args.smoke:
            repeated_rows = evaluate_candidate(
                candidate=candidate,
                seeds=missing,
                reference=reference,
                reference_indices=subset_indices,
                edm_root=args.edm_root,
                device=device,
                nn_batch_size=1,
                orchestration_batch_size=1,
            )
            diagnostic = compare_smoke_rows(candidate_rows, repeated_rows)
            dump_json(output_dir / "smoke_batching_diagnostic.json", diagnostic)
            if candidate_rows != repeated_rows:
                raise RuntimeError(
                    "Smoke batching invariance failed for "
                    f"{candidate['checkpoint_sha256']}; diagnostic="
                    f"{output_dir / 'smoke_batching_diagnostic.json'}"
                )
        all_rows = merge_resume_rows(all_rows, candidate_rows)
        write_csv(output_dir / PER_SAMPLE_FILENAME, all_rows, per_sample_header())
        existing_keys.update(
            (str(row["checkpoint_sha256"]), int(row["sample_seed"]))
            for row in candidate_rows
        )
    failures = [row for row in all_rows if row["status"] != "ok"]
    write_csv(output_dir / FAILURES_FILENAME, failures, per_sample_header())
    run_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "mode": "smoke" if args.smoke else "full_pilot",
        "baseline_only": True,
        "swap_conditions_generated": False,
        "pilot_seeds": list(seeds),
        "future_confirmatory_seeds_touched": False,
        "candidate_pool_manifest_sha256": sha256_file(
            output_dir / POOL_MANIFEST_FILENAME
        ),
        "repository_commit": git_commit(),
        "repository_checkout": repository_checkout_provenance(),
        "output_dir": str(output_dir),
        "device": str(device),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm": {
            "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
            "gpu_type": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
            "memory_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE", ""),
            "time_limit": os.environ.get("SLURM_TIMELIMIT", ""),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "nearest_neighbor_evaluator": {
            "device": "cpu",
            "dtype": "float64",
            "distance": "direct_squared_differences_then_euclidean_sqrt",
            "tie_break": "reference_subset_position",
        },
        "pre_execution_and_diagnostic_jobs": [
            {
                "job_id": "15623452",
                "status": "pre_execution_commit_guard_failure",
                "pilot_rows_persisted": 0,
            },
            {
                "job_id": "15623680",
                "status": "smoke_exact_row_comparison_failure",
                "pilot_rows_persisted": 0,
            },
            {
                "job_id": "15623703",
                "status": "intentional_field_level_diagnostic_failure",
                "generated_samples_identical": True,
                "maximum_observed_distance_difference": 3.552713678800501e-15,
                "pilot_rows_persisted": 0,
            },
        ],
    }
    dump_json(output_dir / "pilot_run_manifest.json", run_manifest)
    if not args.smoke:
        summarize(args, config)
    print(
        json.dumps(
            {
                "status": "pilot_complete",
                "mode": run_manifest["mode"],
                "rows": len(all_rows),
                "failures": len(failures),
                "swap_conditions_generated": False,
            },
            indent=2,
        )
    )


def copy_pool_files(source: Path, destination: Path) -> None:
    """Copy immutable pool metadata into an isolated smoke directory."""
    for name in (INVENTORY_FILENAME, POOL_MANIFEST_FILENAME):
        target = destination / name
        target.write_bytes((source / name).read_bytes())


def compare_smoke_rows(
    batched: Sequence[Mapping[str, object]],
    single_item: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report exact and numerical differences between two smoke evaluations."""
    first = {int(row["sample_seed"]): row for row in batched}
    second = {int(row["sample_seed"]): row for row in single_item}
    if set(first) != set(second):
        raise RuntimeError("Smoke diagnostic seed sets differ")
    numeric_fields = ("d1nn", "d2nn", "d1nn_over_d2nn")
    exact_fields = (
        "model_role",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_kimg",
        "sample_seed",
        "generated_sample_hash",
        "d1nn_reference_index",
        "d2nn_reference_index",
        "memorized",
        "status",
        "error",
    )
    records: list[dict[str, object]] = []
    for seed in sorted(first):
        left = first[seed]
        right = second[seed]
        hashes_equal = left["generated_sample_hash"] == right["generated_sample_hash"]
        numeric = {}
        for field in numeric_fields:
            left_value = float(left[field])
            right_value = float(right[field])
            absolute = abs(left_value - right_value)
            scale = max(abs(left_value), abs(right_value))
            numeric[field] = {
                "batched": left_value,
                "single_item": right_value,
                "absolute_difference": absolute,
                "relative_difference": absolute / scale if scale > 0.0 else 0.0,
                "exactly_equal": left[field] == right[field],
            }
        records.append(
            {
                "sample_seed": seed,
                "generated_sample_hash_equal": hashes_equal,
                "maximum_sample_difference": 0.0 if hashes_equal else None,
                "nearest_neighbor_indices_equal": (
                    left["d1nn_reference_index"] == right["d1nn_reference_index"]
                    and left["d2nn_reference_index"] == right["d2nn_reference_index"]
                ),
                "memorized_equal": left["memorized"] == right["memorized"],
                "nonnumeric_metadata_equal": all(
                    left[field] == right[field] for field in exact_fields
                ),
                "numeric_fields": numeric,
                "differing_fields": [
                    field
                    for field in (*exact_fields, *numeric_fields)
                    if left[field] != right[field]
                ],
            }
        )
    return {
        "status": (
            "exact_match"
            if all(not record["differing_fields"] for record in records)
            else "difference_detected"
        ),
        "comparison": "GPU nearest-neighbor batch versus single-item batch",
        "records": records,
    }


def verify_pool_manifest(
    manifest: Mapping[str, Any], inventory_path: Path, config: Mapping[str, Any]
) -> None:
    """Require a complete prospectively frozen pool before pilot inference."""
    if not manifest["pool_frozen_before_pilot"] or manifest["pilot_started"]:
        raise RuntimeError("Candidate pool was not frozen before pilot")
    if manifest["inventory"]["sha256"] != sha256_file(inventory_path):
        raise RuntimeError("Candidate inventory hash mismatch")
    if manifest["config_sha256"] != sha256_file(DEFAULT_CONFIG):
        raise RuntimeError("Candidate pool config hash mismatch")
    if manifest["scientific_scope"] != config["scientific_scope"]:
        raise RuntimeError("Candidate pool scientific scope mismatch")
    if manifest["pilot_seeds"] != list(PILOT_SEEDS):
        raise RuntimeError("Candidate pool pilot seeds mismatch")


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device, requiring CUDA for cluster pilot work."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA is required unless --device cpu is explicit")


def configure_determinism() -> None:
    """Enable deterministic Torch operations for per-seed canonical sampling."""
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_ema_network(
    checkpoint: Path, edm_root: Path, device: torch.device
) -> torch.nn.Module:
    """Load one validated unconditional EMA network."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    network = payload["ema"].to(device).eval()
    if int(getattr(network, "label_dim", -1)) != 0:
        raise RuntimeError("Preflight candidate is not unconditional")
    return network


def seeded_latent(seed: int, device: torch.device) -> torch.Tensor:
    """Generate one latent from its own device-local seeded generator."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (1, 3, 32, 32), generator=generator, device=device, dtype=torch.float32
    )


def sample_baseline(
    network: torch.nn.Module, seed: int, device: torch.device
) -> np.ndarray:
    """Run one complete 18-call no-swap pure-Euler trajectory."""
    state = seeded_latent(seed, device).to(torch.float64) * SIGMA_GRID[0]
    schedule = SIGMA_GRID + (TERMINAL_SIGMA,)
    with torch.no_grad():
        for step_index, (sigma, sigma_next) in enumerate(
            zip(schedule[:-1], schedule[1:])
        ):
            sigma_tensor = torch.full((1,), sigma, device=device, dtype=torch.float64)
            denoised = network(state, sigma_tensor, class_labels=None).to(torch.float64)
            if not torch.isfinite(denoised).all().item():
                raise RuntimeError(
                    f"Nonfinite denoiser output at seed={seed}, step={step_index}"
                )
            state = state + (sigma_next - sigma) * (state - denoised) / sigma
    if not torch.isfinite(state).all().item():
        raise RuntimeError(f"Nonfinite final sample at seed={seed}")
    return state[0].detach().cpu().numpy().astype(np.float64, copy=False)


def evaluate_candidate(
    *,
    candidate: Mapping[str, str],
    seeds: Sequence[int],
    reference: np.ndarray,
    reference_indices: np.ndarray,
    edm_root: Path,
    device: torch.device,
    nn_batch_size: int,
    orchestration_batch_size: int,
) -> list[dict[str, object]]:
    """Generate and evaluate all missing no-swap seeds for one checkpoint."""
    checkpoint = Path(candidate["checkpoint_path"])
    if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash changed after pool freeze: {checkpoint}")
    try:
        network = load_ema_network(checkpoint, edm_root, device)
    except Exception as error:
        return [failed_row(candidate, seed, error) for seed in seeds]
    rows: list[dict[str, object]] = []
    for start in range(0, len(seeds), orchestration_batch_size):
        chunk_seeds = seeds[start : start + orchestration_batch_size]
        samples: list[np.ndarray] = []
        successful_seeds: list[int] = []
        for seed in chunk_seeds:
            try:
                samples.append(sample_baseline(network, seed, device))
                successful_seeds.append(seed)
            except Exception as error:
                rows.append(failed_row(candidate, seed, error))
        if samples:
            sample_array = np.stack(samples)
            indices, distances = nearest_two_cpu(sample_array, reference)
            for position, seed in enumerate(successful_seeds):
                d1 = float(distances[position, 0])
                d2 = float(distances[position, 1])
                rows.append(
                    successful_row(
                        candidate,
                        seed,
                        sample_array[position],
                        d1,
                        d2,
                        int(reference_indices[indices[position, 0]]),
                        int(reference_indices[indices[position, 1]]),
                    )
                )
    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def load_reference_images(archive_path: Path, indices: np.ndarray) -> np.ndarray:
    """Load the frozen 1K reference as NCHW float64 values in [-1,1]."""
    images = np.empty((len(indices), 3, 32, 32), dtype=np.float64)
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".png"))
        for position, dataset_index in enumerate(indices):
            with archive.open(names[int(dataset_index)]) as handle:
                array = np.asarray(
                    PIL.Image.open(handle).convert("RGB"), dtype=np.float64
                )
            images[position] = np.transpose(2.0 * array / 255.0 - 1.0, (2, 0, 1))
    return images


def successful_row(
    candidate: Mapping[str, str],
    seed: int,
    sample: np.ndarray,
    d1: float,
    d2: float,
    d1_index: int,
    d2_index: int,
) -> dict[str, object]:
    """Build one successful baseline-only per-sample record."""
    ratio = d1 / d2 if d2 > 0.0 else float("inf")
    return {
        "model_role": candidate["model_role"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_kimg": int(candidate["training_kimg"]),
        "sample_seed": seed,
        "generated_sample_hash": generated_sample_hash(sample),
        "d1nn": f"{d1:.17g}",
        "d2nn": f"{d2:.17g}",
        "d1nn_reference_index": d1_index,
        "d2nn_reference_index": d2_index,
        "d1nn_over_d2nn": f"{ratio:.17g}",
        "memorized": int(d1 < d2 / 3.0),
        "status": "ok",
        "error": "",
    }


def failed_row(
    candidate: Mapping[str, str], seed: int, error: Exception
) -> dict[str, object]:
    """Retain one failed seed explicitly in the canonical per-sample schema."""
    return {
        "model_role": candidate["model_role"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_kimg": int(candidate["training_kimg"]),
        "sample_seed": seed,
        "generated_sample_hash": "",
        "d1nn": "",
        "d2nn": "",
        "d1nn_reference_index": "",
        "d2nn_reference_index": "",
        "d1nn_over_d2nn": "",
        "memorized": 0,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }


def normalize_resume_row(row: Mapping[str, str]) -> dict[str, object]:
    """Normalize key fields from an existing CSV before stable resume merging."""
    result: dict[str, object] = dict(row)
    for key in ("training_kimg", "sample_seed", "memorized"):
        result[key] = int(row[key])
    for key in ("d1nn_reference_index", "d2nn_reference_index"):
        result[key] = int(row[key]) if row[key] else ""
    return result


def summarize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Aggregate complete pilot rows, select a pair, validate, and plot."""
    output_dir = args.output_dir.resolve()
    inventory = read_csv_rows(output_dir / INVENTORY_FILENAME)
    accepted = [row for row in inventory if row["inventory_status"] == "accepted"]
    rows = [
        normalize_resume_row(row)
        for row in read_csv_rows(output_dir / PER_SAMPLE_FILENAME)
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint_sha256"]), []).append(row)
    summaries = [
        summarize_candidate(
            model_role=candidate["model_role"],
            checkpoint_path=candidate["checkpoint_path"],
            checkpoint_sha256=candidate["checkpoint_sha256"],
            training_kimg=int(candidate["training_kimg"]),
            rows=grouped.get(candidate["checkpoint_sha256"], []),
        )
        for candidate in accepted
    ]
    summaries.sort(
        key=lambda row: (
            str(row["model_role"]),
            int(row["training_kimg"]),
            str(row["checkpoint_sha256"]),
        )
    )
    write_csv(output_dir / SUMMARY_FILENAME, summaries, summary_header())
    pair = select_model_pair(summaries)
    outcome_name = (
        "ELIGIBLE_PAIR_FROZEN" if pair is not None else "BLOCKED_NO_ELIGIBLE_PAIR"
    )
    pair_payload = {
        "experiment_id": EXPERIMENT_ID,
        "preflight_outcome": outcome_name,
        "selected_pair": pair,
        "selection_rule": config["pair_selection"],
        "eligibility": config["eligibility"],
        "pilot_seeds": list(PILOT_SEEDS),
        "future_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "sampler": config["sampler"],
        "dataset": config["dataset"],
        "reference_subset": config["memorization"],
        "candidate_pool_manifest_sha256": sha256_file(
            output_dir / POOL_MANIFEST_FILENAME
        ),
        "code_commit": git_commit(),
        "repository_checkout": repository_checkout_provenance(),
        "e008_executed": False,
        "swap_conditions_generated": False,
    }
    dump_json(output_dir / PAIR_FILENAME, pair_payload)
    outcome = {
        "experiment_id": EXPERIMENT_ID,
        "outcome": outcome_name,
        "eligible_edm_1k_count": sum(
            row["model_role"] == "edm_1k" and row["eligible"] for row in summaries
        ),
        "eligible_edm_50k_count": sum(
            row["model_role"] == "edm_50k" and row["eligible"] for row in summaries
        ),
        "new_training_started": False,
        "e008_executed": False,
    }
    dump_json(output_dir / OUTCOME_FILENAME, outcome)
    generate_figures(summaries, pair, output_dir / "figures")
    validation = validate_outputs(output_dir, config, require_complete=True)
    dump_json(output_dir / VALIDATION_FILENAME, validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"E008 preflight validation failed: {validation}")


def validate_outputs(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    require_complete: bool,
) -> dict[str, object]:
    """Validate frozen seeds, rows, artifact identities, and no-swap scope."""
    inventory_path = output_dir / INVENTORY_FILENAME
    manifest_path = output_dir / POOL_MANIFEST_FILENAME
    inventory = read_csv_rows(inventory_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_pool_manifest(manifest, inventory_path, config)
    accepted = [row for row in inventory if row["inventory_status"] == "accepted"]
    rows = read_csv_rows(output_dir / PER_SAMPLE_FILENAME)
    keys = [(row["checkpoint_sha256"], int(row["sample_seed"])) for row in rows]
    expected_keys = {
        (candidate["checkpoint_sha256"], seed)
        for candidate in accepted
        for seed in PILOT_SEEDS
    }
    observed_keys = set(keys)
    checks = {
        "pool_frozen_before_pilot": bool(manifest["pool_frozen_before_pilot"]),
        "inventory_hash_matches": manifest["inventory"]["sha256"]
        == sha256_file(inventory_path),
        "no_duplicate_checkpoint_seed_keys": len(keys) == len(observed_keys),
        "pilot_seed_set_exact": {seed for _, seed in keys} == set(PILOT_SEEDS),
        "confirmatory_seed_overlap_absent": not {seed for _, seed in keys}.intersection(
            CONFIRMATORY_SEEDS
        ),
        "all_candidates_have_complete_rows": observed_keys == expected_keys,
        "swap_fields_absent": all(
            field not in rows[0] if rows else True
            for field in ("donor_model", "swap_window", "window_name")
        ),
        "criterion_unchanged": config["memorization"]["criterion"] == "d1nn < d2nn / 3",
        "e008_unexecuted": not config["scientific_scope"]["e008_executed"],
    }
    status = "pass" if all(checks.values()) else "fail"
    if not require_complete:
        status = (
            "pass"
            if all(
                value
                for key, value in checks.items()
                if key
                not in {"pilot_seed_set_exact", "all_candidates_have_complete_rows"}
            )
            else "fail"
        )
    artifacts = {}
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != VALIDATION_FILENAME:
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "checks": checks,
        "accepted_candidate_count": len(accepted),
        "per_sample_row_count": len(rows),
        "failed_row_count": sum(row["status"] != "ok" for row in rows),
        "artifacts": artifacts,
    }


def generate_figures(
    summaries: Sequence[Mapping[str, object]],
    selected_pair: Mapping[str, object] | None,
    figures_dir: Path,
) -> None:
    """Generate the three required baseline-only checkpoint figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    roles = ("edm_1k", "edm_50k")
    colors = {"edm_1k": "#C8563A", "edm_50k": "#247B7B"}
    selected_hashes = set()
    if selected_pair:
        selected_hashes = {
            str(selected_pair["edm_1k"]["checkpoint_sha256"]),
            str(selected_pair["edm_50k"]["checkpoint_sha256"]),
        }

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    for axis, role in zip(axes, roles):
        rows = [row for row in summaries if row["model_role"] == role]
        x = np.arange(len(rows))
        rates = np.asarray([float(row["memorization_rate"]) for row in rows])
        eligible = np.asarray([bool(row["eligible"]) for row in rows])
        axis.axhspan(0.10, 0.90, color="#DDE8D5", alpha=0.65, label="eligible")
        axis.scatter(
            x[~eligible], rates[~eligible], color="#777777", s=45, label="rejected"
        )
        axis.scatter(
            x[eligible], rates[eligible], color=colors[role], s=55, label="eligible"
        )
        for index, row in enumerate(rows):
            axis.annotate(
                f"{row['memorized_count']}/128",
                (index, rates[index]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
            if row["checkpoint_sha256"] in selected_hashes:
                axis.scatter(
                    index,
                    rates[index],
                    facecolors="none",
                    edgecolors="#111111",
                    s=150,
                    linewidths=2,
                )
        axis.set_title(role.replace("_", "-").upper())
        axis.set_ylabel("Pilot memorization rate")
        axis.set_xticks(x, [str(row["training_kimg"]) for row in rows], rotation=45)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Checkpoint training duration (kimg)")
    fig.suptitle("E008 baseline-only checkpoint eligibility")
    fig.tight_layout()
    fig.savefig(figures_dir / "pilot_baseline_rate_by_checkpoint.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.axhspan(0.10, 0.90, color="#DDE8D5", alpha=0.65)
    for role in roles:
        rows = [row for row in summaries if row["model_role"] == role]
        axis.plot(
            [int(row["training_kimg"]) for row in rows],
            [float(row["memorization_rate"]) for row in rows],
            marker="o",
            color=colors[role],
            label=role.replace("_", "-").upper(),
        )
    axis.set(xlabel="Training duration (kimg)", ylabel="Pilot memorization rate")
    axis.set_title("No-swap memorization rate across existing snapshots")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pilot_baseline_rate_by_training_duration.png", dpi=180)
    plt.close(fig)

    left = [
        row for row in summaries if row["model_role"] == "edm_1k" and row["eligible"]
    ]
    right = [
        row for row in summaries if row["model_role"] == "edm_50k" and row["eligible"]
    ]
    fig, axis = plt.subplots(figsize=(8, 7))
    for first in left:
        for second in right:
            chosen = selected_pair and (
                first["checkpoint_sha256"]
                == selected_pair["edm_1k"]["checkpoint_sha256"]
                and second["checkpoint_sha256"]
                == selected_pair["edm_50k"]["checkpoint_sha256"]
            )
            axis.scatter(
                float(first["memorization_rate"]),
                float(second["memorization_rate"]),
                s=120 if chosen else 35,
                facecolors="#D9A441" if chosen else "#7A9E9F",
                edgecolors="#111111" if chosen else "none",
            )
    axis.plot([0.1, 0.9], [0.1, 0.9], linestyle="--", color="#777777")
    axis.set(
        xlim=(0.08, 0.92),
        ylim=(0.08, 0.92),
        xlabel="EDM-1K pilot memorization rate",
        ylabel="EDM-50K pilot memorization rate",
        title="Eligible cross-role checkpoint pairs",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "eligible_checkpoint_pairs.png", dpi=180)
    plt.close(fig)


def per_sample_header() -> list[str]:
    """Return the frozen per-sample schema."""
    return [
        "model_role",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_kimg",
        "sample_seed",
        "generated_sample_hash",
        "d1nn",
        "d2nn",
        "d1nn_reference_index",
        "d2nn_reference_index",
        "d1nn_over_d2nn",
        "memorized",
        "status",
        "error",
    ]


def inventory_header() -> list[str]:
    """Return the complete accepted/rejected inventory schema."""
    return [
        "model_role",
        "checkpoint_path",
        "checkpoint_filename",
        "checkpoint_sha256",
        "training_kimg",
        "file_size",
        "modification_time",
        "architecture_identity",
        "training_subset_size",
        "training_configuration_source",
        "training_configuration_sha256",
        "ema_status",
        "label_conditioning_status",
        "loadability_status",
        "inventory_status",
        "rejection_reason",
    ]


def summary_header() -> list[str]:
    """Return the frozen per-checkpoint pilot summary schema."""
    return [
        "model_role",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_kimg",
        "n_samples",
        "memorized_count",
        "memorization_rate",
        "ci95_low",
        "ci95_high",
        "eligible",
        "eligibility_reason",
        "n_failures",
        "status",
    ]


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    """Atomically write stable CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def dump_json(path: Path, payload: object) -> None:
    """Atomically write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_commit() -> str:
    """Return the current repository commit."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def repository_commit(path: Path) -> str:
    """Return one external source repository commit."""
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def repository_checkout_provenance() -> dict[str, str]:
    """Record both the submitted checkout path and its canonical resolution."""
    return {
        "user_facing_path": os.environ.get("E008_REPO_ROOT", str(REPO_ROOT)),
        "resolved_path": str(REPO_ROOT.resolve()),
    }


if __name__ == "__main__":
    main()
