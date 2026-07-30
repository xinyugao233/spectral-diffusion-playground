#!/usr/bin/env python3
"""Experiment 6: whole-denoiser swaps over frozen E005 transition windows.

This is a paper-derived clean-room experiment. It uses matched clean-room
EDM-1K and EDM-50K checkpoints, pure Euler sampling, fixed swap windows, and
the paper's stated pixel-space nearest-neighbor memorization criterion.
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
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from _bootstrap import REPO_ROOT
from spectral_diffusion_playground.e005_spectral_residuals import sha256_file
from spectral_diffusion_playground.e006_transition_swaps import (
    EFFECT_THRESHOLD,
    EXPERIMENT_ID,
    PAIR_BOOTSTRAP_RESAMPLES,
    PAIR_BOOTSTRAP_SEED,
    PROTOCOL_COMMIT,
    RUN_ID,
    SAMPLE_SEEDS,
    SIGMA_GRID,
    TERMINAL_SIGMA,
    ConditionSpec,
    classify_outcome,
    clopper_pearson_interval,
    deterministic_qualitative_selection,
    frozen_conditions,
    generated_sample_hash,
    paired_effect_summary,
    transition_influence,
)

EDM_ROOT = Path("/home/xggh8/edm")
DEFAULT_CONFIG = REPO_ROOT / "configs/e006_transition_window_swaps.json"
DEFAULT_OUTPUT = Path("/home/xggh8/data/zw-lab/e006_transition_window_swaps")
PAPER_SHA256 = "1dd6b436878c74327dab0c289e57335a915cdb35845fb75967882ba62375d2d8"
CORRECTED_PLAN_SHA256 = (
    "0317dc62391cd28bfbd784f270f279f149fe25f844a7f52379ce4c5e769be317"
)
TRAIN_ARCHIVE = Path("/home/xggh8/datasets/edm/cifar10-32x32-train50k.zip")
TRAIN_ARCHIVE_SHA256 = (
    "795cdc1444465ae4e19e25a0615d05ba0a0e83caa5db6b1b811deaf4c7910dfa"
)
SUBSET_TEXT_SHA256 = (
    "33bb509c48144464a48d3b945cc44c14f880a1e6c6470c283dc0ed65e22b1f29"
)
SUBSET_INT64_SHA256 = (
    "f97076ea6db59a96dc81a59d1b573bc8aaecdb8efa1e93c0d79928bfbf8a43f8"
)
EDM_1K_CHECKPOINT = Path(
    "/home/xggh8/data/zw-lab/exp_004_standard_edm_n1000_40000kimg_20260415/"
    "network-snapshot-040000.pkl"
)
EDM_50K_CHECKPOINT = Path(
    "/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/"
    "network-snapshot-040000.pkl"
)
EDM_1K_SHA256 = "8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1"
EDM_50K_SHA256 = "a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b"


def parse_args() -> argparse.Namespace:
    """Parse bounded E006 execution arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edm-root", type=Path, default=EDM_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate provenance, run E006, and write the frozen artifacts."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing to run E006 outside Slurm")
    args = parse_args()
    validate_args(args)
    config = load_and_validate_config(args.config)
    provenance = validate_provenance(args, config)
    if args.preflight_only:
        print(json.dumps({"preflight": "pass", **provenance}, indent=2))
        return

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    started = time.time()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("E006 full/smoke evaluation requires CUDA")
    configure_determinism()
    manifest = build_manifest(args, config, provenance, device, output_dir)
    dump_json(output_dir / "experiment_06_manifest.json", manifest)

    subset_indices = load_subset_indices()
    reference = load_reference_images(subset_indices)
    networks = load_networks(args.edm_root, device)
    conditions = selected_conditions(args.smoke)

    if args.smoke:
        generated = run_smoke_batching_check(
            networks=networks,
            conditions=conditions,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        generated = sample_all_conditions(
            networks=networks,
            conditions=conditions,
            seeds=SAMPLE_SEEDS,
            device=device,
            batch_size=args.batch_size,
        )

    rows, neighbor_rows, failures = evaluate_nearest_neighbors(
        generated=generated,
        conditions=conditions,
        seeds=(0, 1) if args.smoke else SAMPLE_SEEDS,
        reference=reference,
        reference_indices=subset_indices,
        device=device,
        nn_batch_size=args.nn_batch_size,
        batch_size=args.batch_size,
    )
    write_csv(
        output_dir / "experiment_06_per_sample.csv",
        rows,
        per_sample_header(),
    )
    write_csv(
        output_dir / "experiment_06_nearest_neighbors.csv",
        neighbor_rows,
        nearest_neighbor_header(),
    )
    write_csv(
        output_dir / "experiment_06_failures.csv",
        failures,
        failure_header(),
    )
    np.savez_compressed(
        output_dir / "experiment_06_generated_samples.npz",
        **generated,
    )

    summaries = build_condition_summaries(rows, conditions)
    write_csv(
        output_dir / "experiment_06_condition_summary.csv",
        summaries,
        condition_summary_header(),
    )
    paired = build_paired_comparisons(rows, conditions)
    write_csv(
        output_dir / "experiment_06_paired_comparisons.csv",
        paired,
        paired_comparison_header(),
    )
    if args.smoke:
        outcome = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "outcome": "INCONCLUSIVE",
            "smoke_only": True,
            "interpretation": (
                "Smoke mode validates execution and does not classify E006."
            ),
            "failure_summary": {"count": len(failures)},
        }
    else:
        outcome = build_outcome(rows, conditions, failures)
    dump_json(output_dir / "experiment_06_outcome.json", outcome)
    qualitative = build_qualitative_manifest(rows, conditions)
    dump_json(
        output_dir / "experiment_06_qualitative_sample_manifest.json",
        qualitative,
    )
    validation = validate_outputs(
        rows=rows,
        neighbor_rows=neighbor_rows,
        summaries=summaries,
        paired=paired,
        failures=failures,
        conditions=conditions,
        seeds=(0, 1) if args.smoke else SAMPLE_SEEDS,
    )
    dump_json(output_dir / "experiment_06_validation.json", validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"E006 output validation failed: {validation}")

    if not args.smoke:
        generate_figures(
            summaries=summaries,
            paired=paired,
            rows=rows,
            conditions=conditions,
            generated=generated,
            reference=reference,
            subset_indices=subset_indices,
            qualitative=qualitative,
            output_dir=figures_dir,
        )
    manifest["completed_at_unix"] = time.time()
    manifest["runtime_sec"] = time.time() - started
    manifest["outcome"] = outcome["outcome"]
    manifest["outputs"] = {
        path.name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.glob("experiment_06_*"))
        if path.is_file() and path.name != "experiment_06_manifest.json"
    }
    manifest["figures"] = {
        path.name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(figures_dir.glob("*.png"))
    }
    dump_json(output_dir / "experiment_06_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "outcome": outcome["outcome"],
                "conditions": len(conditions),
                "samples": len(rows),
                "runtime_sec": manifest["runtime_sec"],
            },
            indent=2,
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    """Reject execution settings outside the frozen protocol."""
    if args.batch_size <= 0 or args.nn_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.preflight_only and args.smoke:
        raise ValueError("--preflight-only and --smoke are mutually exclusive")
    if not args.smoke and args.output_dir.resolve() != DEFAULT_OUTPUT:
        raise ValueError(f"Full E006 output is frozen at {DEFAULT_OUTPUT}")


def load_and_validate_config(path: Path) -> dict[str, Any]:
    """Load the frozen JSON config and verify its scientific fields."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError("E006 experiment ID mismatch")
    if config["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("E006 protocol commit mismatch")
    sampler = config["sampler"]
    expected_sampler = {
        "algorithm": "pure_euler",
        "num_denoiser_calls": 18,
        "sigma_max": 80.0,
        "sigma_min": 0.002,
        "rho": 7.0,
        "terminal_sigma": 0.0,
        "churn": 0.0,
        "heun_correction": False,
        "class_labels": None,
        "canonical_per_seed_inference": True,
    }
    if sampler != expected_sampler:
        raise RuntimeError(f"Frozen sampler config mismatch: {sampler}")
    if config["statistics"]["effect_threshold"] != EFFECT_THRESHOLD:
        raise RuntimeError("E006 effect-size threshold mismatch")
    if config["sample_seeds"][
        "start"
    ] != 0 or config["sample_seeds"]["stop_inclusive"] != 255:
        raise RuntimeError("E006 sample-seed range mismatch")
    return config


def validate_provenance(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate every frozen identity before output creation."""
    checks = {
        "config_sha256": sha256_file(args.config),
        "protocol_sha256": sha256_file(
            REPO_ROOT / "docs/experiment_06_transition_swap_protocol.md"
        ),
        "transition_windows_sha256": sha256_file(
            REPO_ROOT
            / "results/experiment_05/experiment_05_transition_windows.json"
        ),
        "subset_text_sha256": sha256_file(
            REPO_ROOT / "data/e005_cifar10_subset_1k_indices.txt"
        ),
        "subset_int64_sha256": hash_subset_int64(load_subset_indices()),
        "archive_sha256": sha256_file(TRAIN_ARCHIVE),
        "edm_1k_checkpoint_sha256": sha256_file(EDM_1K_CHECKPOINT),
        "edm_50k_checkpoint_sha256": sha256_file(EDM_50K_CHECKPOINT),
        "git_commit": git_commit(),
        "edm_commit": repository_commit(args.edm_root),
    }
    expected = {
        "protocol_sha256": (
            "8906a2a09efb6fbcf8c301ca472624905656992534a1cafbe88c1ba576ef23ae"
        ),
        "transition_windows_sha256": (
            "aa588d071716e81694ea467f282947cc9949834ffdc8011abe847d0344cbd6bf"
        ),
        "subset_text_sha256": SUBSET_TEXT_SHA256,
        "subset_int64_sha256": SUBSET_INT64_SHA256,
        "archive_sha256": TRAIN_ARCHIVE_SHA256,
        "edm_1k_checkpoint_sha256": EDM_1K_SHA256,
        "edm_50k_checkpoint_sha256": EDM_50K_SHA256,
    }
    for key, expected_value in expected.items():
        if checks[key] != expected_value:
            raise RuntimeError(f"{key} mismatch: {checks[key]} != {expected_value}")
    expected_repo_commit = os.environ.get("E006_REPO_COMMIT")
    if expected_repo_commit and checks["git_commit"] != expected_repo_commit:
        raise RuntimeError(
            f"Repository commit mismatch: {checks['git_commit']} != "
            f"{expected_repo_commit}"
        )
    if config["models"]["edm_1k"]["sha256"] != EDM_1K_SHA256:
        raise RuntimeError("Config EDM-1K checkpoint identity mismatch")
    if config["models"]["edm_50k"]["sha256"] != EDM_50K_SHA256:
        raise RuntimeError("Config EDM-50K checkpoint identity mismatch")
    return checks


def configure_determinism() -> None:
    """Enable deterministic Torch behavior for canonical per-seed sampling."""
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def selected_conditions(smoke: bool) -> tuple[ConditionSpec, ...]:
    """Return all frozen conditions or the frozen four-condition smoke subset."""
    conditions = frozen_conditions()
    if not smoke:
        return conditions
    wanted = {
        "edm_1k_no_swap",
        "edm_50k_no_swap",
        "edm_1k_base__edm_50k_donor__low_transition",
        "edm_50k_base__edm_1k_donor__high_transition",
    }
    return tuple(condition for condition in conditions if condition.name in wanted)


def load_networks(
    edm_root: Path, device: torch.device
) -> dict[str, torch.nn.Module]:
    """Load both validated EMA denoisers."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    networks: dict[str, torch.nn.Module] = {}
    for name, checkpoint in (
        ("edm_1k", EDM_1K_CHECKPOINT),
        ("edm_50k", EDM_50K_CHECKPOINT),
    ):
        with checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        network = payload["ema"].to(device).eval()
        if getattr(network, "label_dim", 0) != 0:
            raise RuntimeError(f"{name} is not unconditional")
        networks[name] = network
    return networks


def seeded_latent(seed: int, device: torch.device) -> torch.Tensor:
    """Generate one Gaussian latent from one device-local Torch generator."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (1, 3, 32, 32),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def sample_one(
    networks: dict[str, torch.nn.Module],
    condition: ConditionSpec,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Sample one seed with exactly 18 pure-Euler denoiser calls."""
    state = seeded_latent(seed, device).to(torch.float64) * SIGMA_GRID[0]
    schedule = SIGMA_GRID + (TERMINAL_SIGMA,)
    with torch.no_grad():
        for step_index, (sigma, sigma_next) in enumerate(
            zip(schedule[:-1], schedule[1:])
        ):
            model_name = condition.model_for_step(step_index)
            sigma_tensor = torch.full(
                (1,), sigma, device=device, dtype=torch.float64
            )
            denoised = networks[model_name](
                state, sigma_tensor, class_labels=None
            ).to(torch.float64)
            if not torch.isfinite(denoised).all().item():
                raise RuntimeError(
                    f"Nonfinite denoiser output at seed={seed}, step={step_index}"
                )
            derivative = (state - denoised) / sigma
            state = state + (sigma_next - sigma) * derivative
    if not torch.isfinite(state).all().item():
        raise RuntimeError(f"Nonfinite final sample for seed={seed}")
    return state[0].detach().cpu().numpy().astype(np.float64, copy=False)


def sample_all_conditions(
    *,
    networks: dict[str, torch.nn.Module],
    conditions: tuple[ConditionSpec, ...],
    seeds: Iterable[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Generate every condition in stable condition/seed order."""
    seed_values = tuple(int(seed) for seed in seeds)
    generated: dict[str, np.ndarray] = {}
    for condition in conditions:
        samples = np.empty((len(seed_values), 3, 32, 32), dtype=np.float64)
        progress = tqdm(
            range(0, len(seed_values), batch_size),
            desc=condition.name,
            leave=False,
        )
        for start in progress:
            for offset, seed in enumerate(
                seed_values[start : start + batch_size], start=start
            ):
                samples[offset] = sample_one(networks, condition, seed, device)
        generated[condition.name] = samples
    return generated


def run_smoke_batching_check(
    *,
    networks: dict[str, torch.nn.Module],
    conditions: tuple[ConditionSpec, ...],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run seeds 0/1 twice and require byte-identical partition-independent output."""
    first = sample_all_conditions(
        networks=networks,
        conditions=conditions,
        seeds=(0, 1),
        device=device,
        batch_size=1,
    )
    second = sample_all_conditions(
        networks=networks,
        conditions=conditions,
        seeds=(0, 1),
        device=device,
        batch_size=max(2, batch_size),
    )
    for condition in conditions:
        if not np.array_equal(first[condition.name], second[condition.name]):
            maximum = float(
                np.max(np.abs(first[condition.name] - second[condition.name]))
            )
            raise RuntimeError(
                f"Batching independence failed for {condition.name}: {maximum}"
            )
        for sample_a, sample_b in zip(
            first[condition.name], second[condition.name]
        ):
            if generated_sample_hash(sample_a) != generated_sample_hash(sample_b):
                raise RuntimeError("Generated-sample hash changed across batches")
    return first


def load_subset_indices() -> np.ndarray:
    """Load the frozen ordered clean-room 1K subset."""
    values = np.loadtxt(
        REPO_ROOT / "data/e005_cifar10_subset_1k_indices.txt",
        dtype=np.int64,
    )
    if values.shape != (1000,) or len(np.unique(values)) != 1000:
        raise RuntimeError(f"Invalid E006 reference subset: {values.shape}")
    return values


def hash_subset_int64(values: np.ndarray) -> str:
    """Hash ordered indices as canonical little-endian int64 bytes."""
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_reference_images(subset_indices: np.ndarray) -> np.ndarray:
    """Load frozen reference images as unquantized NCHW float64 in [-1, 1]."""
    images = np.empty((len(subset_indices), 3, 32, 32), dtype=np.float64)
    with zipfile.ZipFile(TRAIN_ARCHIVE) as archive:
        png_names = sorted(
            name for name in archive.namelist() if name.endswith(".png")
        )
        for position, dataset_index in enumerate(subset_indices):
            with archive.open(png_names[int(dataset_index)]) as handle:
                image = PIL.Image.open(handle).convert("RGB")
                array = np.asarray(image, dtype=np.float64)
            images[position] = np.transpose(
                2.0 * (array / 255.0) - 1.0, (2, 0, 1)
            )
    return images


def nearest_two_torch(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact float64 pixel-space nearest-two neighbors in chunks."""
    reference_tensor = torch.from_numpy(
        reference.reshape(reference.shape[0], -1)
    ).to(device=device, dtype=torch.float64)
    reference_norm = torch.sum(reference_tensor * reference_tensor, dim=1)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    for start in range(0, generated.shape[0], batch_size):
        query = torch.from_numpy(
            generated[start : start + batch_size].reshape(
                min(batch_size, generated.shape[0] - start), -1
            )
        ).to(device=device, dtype=torch.float64)
        squared = (
            torch.sum(query * query, dim=1, keepdim=True)
            + reference_norm[None, :]
            - 2.0 * (query @ reference_tensor.T)
        ).clamp_min_(0.0)
        values, indices = torch.topk(
            squared, k=2, dim=1, largest=False, sorted=True
        )
        all_indices.append(indices.cpu().numpy().astype(np.int64))
        all_distances.append(torch.sqrt(values).cpu().numpy().astype(np.float64))
    return np.concatenate(all_indices), np.concatenate(all_distances)


def evaluate_nearest_neighbors(
    *,
    generated: dict[str, np.ndarray],
    conditions: tuple[ConditionSpec, ...],
    seeds: Iterable[int],
    reference: np.ndarray,
    reference_indices: np.ndarray,
    device: torch.device,
    nn_batch_size: int,
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate all generated samples and produce stable per-sample records."""
    seed_values = tuple(int(seed) for seed in seeds)
    rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for condition in conditions:
        samples = generated[condition.name]
        indices, distances = nearest_two_torch(
            samples, reference, device=device, batch_size=nn_batch_size
        )
        for position, seed in enumerate(seed_values):
            status = "ok"
            error = ""
            if not np.all(np.isfinite(samples[position])):
                status = "nonfinite_sample"
                error = "generated sample contains nonfinite values"
            elif not np.all(np.isfinite(distances[position])):
                status = "nearest_neighbor_failure"
                error = "nearest-neighbor distances are nonfinite"
            d1 = float(distances[position, 0])
            d2 = float(distances[position, 1])
            ratio = float(d1 / d2) if d2 > 0.0 else float("inf")
            memorized = bool(d1 < d2 / 3.0) if status == "ok" else False
            window = condition.window
            row = {
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "condition": condition.name,
                "base_model": condition.base_model,
                "base_checkpoint_sha256": checkpoint_hash(condition.base_model),
                "donor_model": condition.donor_model or "",
                "donor_checkpoint_sha256": (
                    checkpoint_hash(condition.donor_model)
                    if condition.donor_model
                    else ""
                ),
                "window_name": window.name if window else "none",
                "window_start_index": window.start_index if window else "",
                "window_end_index": window.end_index if window else "",
                "window_start_sigma": (
                    f"{window.start_sigma:.17g}" if window else ""
                ),
                "window_end_sigma": f"{window.end_sigma:.17g}" if window else "",
                "sampler": "pure_euler_18_step_rho7",
                "sample_seed": seed,
                "batch_index": position // batch_size,
                "generated_sample_hash": generated_sample_hash(samples[position]),
                "d1nn": f"{d1:.17g}",
                "d2nn": f"{d2:.17g}",
                "d1nn_reference_index": int(
                    reference_indices[indices[position, 0]]
                ),
                "d2nn_reference_index": int(
                    reference_indices[indices[position, 1]]
                ),
                "d1nn_over_d2nn": f"{ratio:.17g}",
                "memorized": int(memorized),
                "status": status,
                "error": error,
            }
            rows.append(row)
            if status != "ok":
                failures.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "run_id": RUN_ID,
                        "condition": condition.name,
                        "sample_seed": seed,
                        "status": status,
                        "error": error,
                    }
                )
            for rank in (0, 1):
                neighbor_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "run_id": RUN_ID,
                        "condition": condition.name,
                        "sample_seed": seed,
                        "rank": rank + 1,
                        "reference_index": int(
                            reference_indices[indices[position, rank]]
                        ),
                        "reference_subset_position": int(
                            indices[position, rank]
                        ),
                        "distance": f"{distances[position, rank]:.17g}",
                    }
                )
    return rows, neighbor_rows, failures


def checkpoint_hash(model_name: str) -> str:
    """Return a frozen checkpoint digest by model name."""
    if model_name == "edm_1k":
        return EDM_1K_SHA256
    if model_name == "edm_50k":
        return EDM_50K_SHA256
    raise ValueError(f"Unknown model: {model_name}")


def rows_by_condition(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group stable rows by condition."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    return grouped


def labels_for(rows: list[dict[str, object]]) -> np.ndarray:
    """Return binary memorization labels from stable rows."""
    return np.asarray([int(row["memorized"]) for row in rows], dtype=np.int8)


def build_condition_summaries(
    rows: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
) -> list[dict[str, object]]:
    """Aggregate exact binomial summaries for every condition."""
    grouped = rows_by_condition(rows)
    summaries: list[dict[str, object]] = []
    for condition in conditions:
        condition_rows = grouped[condition.name]
        labels = labels_for(condition_rows)
        count = int(labels.sum())
        ci_low, ci_high = clopper_pearson_interval(count, len(labels))
        ratios = np.asarray(
            [float(row["d1nn_over_d2nn"]) for row in condition_rows],
            dtype=np.float64,
        )
        window = condition.window
        summaries.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "condition": condition.name,
                "base_model": condition.base_model,
                "donor_model": condition.donor_model or "",
                "window_name": window.name if window else "none",
                "window_start_index": window.start_index if window else "",
                "window_end_index": window.end_index if window else "",
                "window_start_sigma": (
                    f"{window.start_sigma:.17g}" if window else ""
                ),
                "window_end_sigma": f"{window.end_sigma:.17g}" if window else "",
                "n_samples": len(labels),
                "memorized_count": count,
                "memorization_rate": f"{labels.mean():.17g}",
                "ci95_low": f"{ci_low:.17g}",
                "ci95_high": f"{ci_high:.17g}",
                "mean_d1nn_over_d2nn": f"{ratios.mean():.17g}",
                "median_d1nn_over_d2nn": f"{np.median(ratios):.17g}",
                "status": (
                    "ok"
                    if all(row["status"] == "ok" for row in condition_rows)
                    else "failed"
                ),
            }
        )
    return summaries


def build_paired_comparisons(
    rows: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
) -> list[dict[str, object]]:
    """Compare every swap against its corresponding no-swap baseline."""
    grouped = rows_by_condition(rows)
    output: list[dict[str, object]] = []
    for comparison_index, condition in enumerate(conditions):
        if condition.window is None:
            continue
        baseline_name = f"{condition.base_model}_no_swap"
        summary = paired_effect_summary(
            labels_for(grouped[baseline_name]),
            labels_for(grouped[condition.name]),
            seed=PAIR_BOOTSTRAP_SEED + comparison_index * 10,
            resamples=PAIR_BOOTSTRAP_RESAMPLES,
        )
        control_type = (
            "primary_transition"
            if condition.window.name in {"low_transition", "high_transition"}
            else (
                "matched_control"
                if condition.window.name.endswith("_control")
                else "secondary"
            )
        )
        output.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "comparison": f"{condition.name}_vs_{baseline_name}",
                "base_model": condition.base_model,
                "donor_model": condition.donor_model,
                "swap_condition": condition.name,
                "baseline_condition": baseline_name,
                "window_name": condition.window.name,
                "control_type": control_type,
                **{
                    key: (
                        f"{value:.17g}" if isinstance(value, float) else value
                    )
                    for key, value in summary.items()
                    if key != "direction_supported"
                },
                "passes_practical_threshold": "",
                "status": "ok",
            }
        )
    output_by_condition = {
        str(row["swap_condition"]): row for row in output
    }
    available = set(grouped)
    for direction_index, (base, donor) in enumerate(
        (("edm_1k", "edm_50k"), ("edm_50k", "edm_1k"))
    ):
        baseline_name = f"{base}_no_swap"
        for window_index, (window, pre_control, post_control) in enumerate(
            (
                ("low_transition", "low_pre_control", "low_post_control"),
                ("high_transition", "high_pre_control", "high_post_control"),
            )
        ):
            names = {
                "transition": primary_condition_name(base, donor, window),
                "pre": primary_condition_name(base, donor, pre_control),
                "post": primary_condition_name(base, donor, post_control),
            }
            if not {baseline_name, *names.values()}.issubset(available):
                continue
            result = transition_influence(
                labels_for(grouped[baseline_name]),
                labels_for(grouped[names["transition"]]),
                labels_for(grouped[names["pre"]]),
                labels_for(grouped[names["post"]]),
                seed=PAIR_BOOTSTRAP_SEED
                + direction_index * 100
                + window_index * 10,
                resamples=PAIR_BOOTSTRAP_RESAMPLES,
            )
            output_by_condition[names["transition"]][
                "passes_practical_threshold"
            ] = int(bool(result["influential"]))
    for row in output:
        if row["passes_practical_threshold"] == "":
            row["passes_practical_threshold"] = "not_applicable"
    return output


def primary_condition_name(
    base: str, donor: str, window: str
) -> str:
    """Build one stable swap-condition identifier."""
    return f"{base}_base__{donor}_donor__{window}"


def build_outcome(
    rows: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the frozen transition-versus-control rule mechanically."""
    grouped = rows_by_condition(rows)
    transition_results: dict[str, object] = {}
    influential: dict[tuple[str, str], bool] = {}
    for direction_index, (base, donor, direction) in enumerate(
        (
            ("edm_1k", "edm_50k", "edm_1k_to_edm_50k"),
            ("edm_50k", "edm_1k", "edm_50k_to_edm_1k"),
        )
    ):
        baseline = labels_for(grouped[f"{base}_no_swap"])
        for window_index, (window, pre_control, post_control) in enumerate(
            (
                ("low_transition", "low_pre_control", "low_post_control"),
                ("high_transition", "high_pre_control", "high_post_control"),
            )
        ):
            result = transition_influence(
                baseline,
                labels_for(
                    grouped[primary_condition_name(base, donor, window)]
                ),
                labels_for(
                    grouped[primary_condition_name(base, donor, pre_control)]
                ),
                labels_for(
                    grouped[primary_condition_name(base, donor, post_control)]
                ),
                seed=PAIR_BOOTSTRAP_SEED
                + direction_index * 100
                + window_index * 10,
                resamples=PAIR_BOOTSTRAP_RESAMPLES,
            )
            key = f"{direction}__{window}"
            transition_results[key] = result
            influential[(direction, window)] = bool(result["influential"])

    baseline_counts = {
        baseline: int(labels_for(grouped[baseline]).sum())
        for baseline in ("edm_1k_no_swap", "edm_50k_no_swap")
    }
    n_samples = len(grouped["edm_1k_no_swap"])
    baseline_degenerate = any(
        count in {0, n_samples} for count in baseline_counts.values()
    )
    outcome = classify_outcome(
        influential,
        invalid=bool(failures),
        baseline_degenerate=baseline_degenerate,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "outcome": outcome,
        "decision_rule_version": "e006_protocol_068c7e3",
        "effect_size_threshold_pp": 10,
        "condition_results": {
            condition.name: {
                "base_model": condition.base_model,
                "donor_model": condition.donor_model,
                "window_name": condition.window.name if condition.window else None,
            }
            for condition in conditions
        },
        "paired_results": "experiment_06_paired_comparisons.csv",
        "transition_vs_control_results": transition_results,
        "baseline_memorized_counts": baseline_counts,
        "baseline_degenerate_definition": (
            "memorized_count is exactly 0 or n under the frozen sample set"
        ),
        "baseline_degenerate": baseline_degenerate,
        "unsupported_danger_zone_language_check": (
            outcome not in {"YES", "PARTIAL"}
        ),
        "failure_summary": {
            "count": len(failures),
            "path": "experiment_06_failures.csv",
        },
        "interpretation": outcome_interpretation(outcome),
    }


def outcome_interpretation(outcome: str) -> str:
    """Return bounded language for one frozen outcome."""
    messages = {
        "YES": (
            "Both E005 transition windows dominate matched controls in both "
            "swap directions under the frozen clean-room intervention."
        ),
        "PARTIAL": (
            "A transition-window effect is supported only for a subset of "
            "windows or swap directions."
        ),
        "MIXED": (
            "Window and direction effects do not support a single transition "
            "interpretation."
        ),
        "NO": (
            "E005 transition-window swaps do not dominate matched controls "
            "under the frozen threshold."
        ),
        "INCONCLUSIVE": (
            "The frozen protocol's failure or baseline-degeneracy safeguard "
            "prevents a directional conclusion."
        ),
    }
    return messages[outcome]


def build_qualitative_manifest(
    rows: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
) -> dict[str, object]:
    """Select qualitative seeds by the frozen deterministic category rule."""
    grouped = rows_by_condition(rows)
    selections: dict[str, object] = {}
    for condition in conditions:
        if condition.window is None or condition.window.name not in {
            "low_transition",
            "high_transition",
        }:
            continue
        condition_rows = grouped[condition.name]
        baseline_rows = grouped[f"{condition.base_model}_no_swap"]
        seeds = [int(row["sample_seed"]) for row in condition_rows]
        selected = deterministic_qualitative_selection(
            labels_for(baseline_rows),
            labels_for(condition_rows),
            seeds,
        )
        selections[condition.name] = {
            "baseline_condition": f"{condition.base_model}_no_swap",
            "categories": selected,
            "missing_categories": [
                name for name, category_seeds in selected.items() if not category_seeds
            ],
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "rule": "first two ascending seeds per frozen category; no replacements",
        "selections": selections,
    }


def validate_outputs(
    *,
    rows: list[dict[str, object]],
    neighbor_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    paired: list[dict[str, object]],
    failures: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
    seeds: Iterable[int],
) -> dict[str, object]:
    """Validate row counts, uniqueness, order, and finite primary records."""
    seed_values = tuple(seeds)
    expected_rows = len(conditions) * len(seed_values)
    keys = [(row["condition"], int(row["sample_seed"])) for row in rows]
    expected_keys = [
        (condition.name, seed) for condition in conditions for seed in seed_values
    ]
    finite = all(
        np.isfinite(float(row[field]))
        for row in rows
        for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
    )
    checks = {
        "expected_per_sample_rows": expected_rows,
        "observed_per_sample_rows": len(rows),
        "expected_nearest_neighbor_rows": expected_rows * 2,
        "observed_nearest_neighbor_rows": len(neighbor_rows),
        "condition_summary_rows": len(summaries),
        "paired_comparison_rows": len(paired),
        "unique_per_sample_keys": len(set(keys)) == len(keys),
        "stable_row_order": keys == expected_keys,
        "finite_nearest_neighbor_records": finite,
        "failure_rows": len(failures),
    }
    status = "pass" if all(
        (
            checks["observed_per_sample_rows"] == expected_rows,
            checks["observed_nearest_neighbor_rows"] == expected_rows * 2,
            checks["condition_summary_rows"] == len(conditions),
            checks["paired_comparison_rows"]
            == len([condition for condition in conditions if condition.window]),
            checks["unique_per_sample_keys"],
            checks["stable_row_order"],
            checks["finite_nearest_neighbor_records"],
        )
    ) else "fail"
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "status": status,
        **checks,
    }


def generate_figures(
    *,
    summaries: list[dict[str, object]],
    paired: list[dict[str, object]],
    rows: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
    generated: dict[str, np.ndarray],
    reference: np.ndarray,
    subset_indices: np.ndarray,
    qualitative: dict[str, object],
    output_dir: Path,
) -> None:
    """Generate the six frozen E006 figure types."""
    plot_memorization_rates(
        summaries, output_dir / "experiment_06_memorization_rates.png"
    )
    plot_paired_changes(
        paired, output_dir / "experiment_06_paired_changes.png"
    )
    plot_transition_controls(
        paired, output_dir / "experiment_06_transition_vs_controls.png"
    )
    plot_ratio_distributions(
        rows, conditions, output_dir / "experiment_06_ratio_distributions.png"
    )
    plot_qualitative_pairs(
        rows,
        generated,
        reference,
        subset_indices,
        qualitative,
        output_dir / "experiment_06_generated_nn_pairs.png",
    )
    plot_paper_medium(
        paired, output_dir / "experiment_06_paper_medium_reference.png"
    )


def plot_memorization_rates(
    summaries: list[dict[str, object]], path: Path
) -> None:
    """Plot exact condition-level memorization rates and intervals."""
    names = [short_condition(str(row["condition"])) for row in summaries]
    rates = np.asarray(
        [float(row["memorization_rate"]) for row in summaries]
    )
    lows = np.asarray([float(row["ci95_low"]) for row in summaries])
    highs = np.asarray([float(row["ci95_high"]) for row in summaries])
    colors = [
        "#3b6c8e" if row["base_model"] == "edm_1k" else "#b7623c"
        for row in summaries
    ]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(names))
    ax.bar(x, rates, color=colors, alpha=0.9)
    ax.errorbar(
        x,
        rates,
        yerr=np.vstack((rates - lows, highs - rates)),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    ax.set_xticks(x, names, rotation=55, ha="right")
    ax.set_ylabel("Memorization rate")
    ax.set_title("Whole-denoiser swap conditions (exact 95% binomial intervals)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paired_changes(paired: list[dict[str, object]], path: Path) -> None:
    """Plot paired memorization-rate changes from each base-model baseline."""
    names = [short_condition(str(row["swap_condition"])) for row in paired]
    deltas = np.asarray([float(row["rate_difference"]) for row in paired])
    lows = np.asarray([float(row["paired_ci95_low"]) for row in paired])
    highs = np.asarray([float(row["paired_ci95_high"]) for row in paired])
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(names))
    ax.bar(
        x,
        deltas,
        color=["#3b6c8e" if value <= 0 else "#b7623c" for value in deltas],
    )
    ax.errorbar(
        x,
        deltas,
        yerr=np.vstack((deltas - lows, highs - deltas)),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(EFFECT_THRESHOLD, color="gray", linestyle=":")
    ax.axhline(-EFFECT_THRESHOLD, color="gray", linestyle=":")
    ax.set_xticks(x, names, rotation=55, ha="right")
    ax.set_ylabel("Paired memorization-rate change")
    ax.set_title("Swap effect relative to the same-base no-swap condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transition_controls(
    paired: list[dict[str, object]], path: Path
) -> None:
    """Compare each primary window to both width-matched controls."""
    index = {str(row["swap_condition"]): row for row in paired}
    groups: list[tuple[str, list[str]]] = []
    for base, donor, label in (
        ("edm_1k", "edm_50k", "1K base / 50K donor"),
        ("edm_50k", "edm_1k", "50K base / 1K donor"),
    ):
        prefix = f"{base}_base__{donor}_donor"
        groups.extend(
            [
                (
                    f"{label}\nlow",
                    [
                        f"{prefix}__low_pre_control",
                        f"{prefix}__low_transition",
                        f"{prefix}__low_post_control",
                    ],
                ),
                (
                    f"{label}\nhigh",
                    [
                        f"{prefix}__high_pre_control",
                        f"{prefix}__high_transition",
                        f"{prefix}__high_post_control",
                    ],
                ),
            ]
        )
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(groups))
    width = 0.24
    styles = (
        ("pre control", "#9aa7ad"),
        ("transition", "#c64835"),
        ("post control", "#557f69"),
    )
    for offset, (label, color) in enumerate(styles):
        values = [
            float(index[condition_names[offset]]["rate_difference"])
            for _, condition_names in groups
        ]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(EFFECT_THRESHOLD, color="gray", linestyle=":")
    ax.axhline(-EFFECT_THRESHOLD, color="gray", linestyle=":")
    ax.set_xticks(x, [name for name, _ in groups])
    ax.set_ylabel("Paired memorization-rate change")
    ax.set_title("E005 transition windows versus width-matched controls")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ratio_distributions(
    rows: list[dict[str, object]],
    conditions: tuple[ConditionSpec, ...],
    path: Path,
) -> None:
    """Plot d1NN/d2NN distributions for every condition."""
    grouped = rows_by_condition(rows)
    values = [
        np.asarray(
            [float(row["d1nn_over_d2nn"]) for row in grouped[condition.name]]
        )
        for condition in conditions
    ]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.boxplot(values, showfliers=False, whis=(5, 95))
    ax.axhline(
        1.0 / 3.0,
        color="#c64835",
        linestyle="--",
        label="memorization threshold",
    )
    ax.set_xticks(
        np.arange(1, len(conditions) + 1),
        [short_condition(condition.name) for condition in conditions],
        rotation=55,
        ha="right",
    )
    ax.set_ylabel("d1NN / d2NN")
    ax.set_title("Pixel-space nearest-neighbor ratio distributions")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_qualitative_pairs(
    rows: list[dict[str, object]],
    generated: dict[str, np.ndarray],
    reference: np.ndarray,
    subset_indices: np.ndarray,
    qualitative: dict[str, object],
    path: Path,
) -> None:
    """Plot deterministic generated/nearest-neighbor pairs."""
    grouped = rows_by_condition(rows)
    selections = qualitative["selections"]
    panels: list[tuple[str, str, int]] = []
    for condition, record in selections.items():
        for category, seeds in record["categories"].items():
            panels.extend((condition, category, int(seed)) for seed in seeds)
    if not panels:
        panels = []
    fig, axes = plt.subplots(
        max(1, len(panels)),
        2,
        figsize=(7.5, max(3.0, 2.4 * len(panels))),
        squeeze=False,
    )
    for row_index, (condition, category, seed) in enumerate(panels):
        condition_rows = grouped[condition]
        record = next(
            row for row in condition_rows if int(row["sample_seed"]) == seed
        )
        reference_index = int(record["d1nn_reference_index"])
        subset_position = int(np.flatnonzero(subset_indices == reference_index)[0])
        axes[row_index, 0].imshow(to_display(generated[condition][seed]))
        axes[row_index, 1].imshow(to_display(reference[subset_position]))
        axes[row_index, 0].set_title(
            f"{short_condition(condition)} | seed {seed}\n{category}",
            fontsize=8,
        )
        ratio = float(record["d1nn_over_d2nn"])
        axes[row_index, 1].set_title(
            f"nearest subset index {reference_index}\nratio={ratio:.3f}",
            fontsize=8,
        )
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    if not panels:
        axes[0, 0].text(0.5, 0.5, "No qualitative seeds selected", ha="center")
        axes[0, 1].axis("off")
    fig.suptitle("Deterministic generated-sample / nearest-neighbor pairs", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paper_medium(paired: list[dict[str, object]], path: Path) -> None:
    """Compare the paper-style medium reference with E005 windows."""
    selected = [
        row
        for row in paired
        if row["window_name"]
        in {
            "low_transition",
            "high_transition",
            "combined_transition",
            "paper_medium_reference",
        }
    ]
    names = [short_condition(str(row["swap_condition"])) for row in selected]
    values = [float(row["rate_difference"]) for row in selected]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(
        np.arange(len(selected)),
        values,
        color=[
            "#1f6f8b" if row["window_name"] == "paper_medium_reference" else "#d07c3e"
            for row in selected
        ],
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(selected)), names, rotation=50, ha="right")
    ax.set_ylabel("Paired memorization-rate change")
    ax.set_title("E005 transitions and paper-style medium clean-room reference")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def to_display(image_chw: np.ndarray) -> np.ndarray:
    """Convert an unquantized scientific tensor for display only."""
    return np.clip(
        np.transpose(np.asarray(image_chw), (1, 2, 0)) * 0.5 + 0.5,
        0.0,
        1.0,
    )


def short_condition(name: str) -> str:
    """Return compact but unambiguous figure text."""
    return (
        name.replace("edm_1k_base__edm_50k_donor__", "1K→50K ")
        .replace("edm_50k_base__edm_1k_donor__", "50K→1K ")
        .replace("edm_1k_no_swap", "1K baseline")
        .replace("edm_50k_no_swap", "50K baseline")
        .replace("_transition", " transition")
        .replace("_control", " control")
        .replace("_reference", " reference")
        .replace("_", " ")
    )


def build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    provenance: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the complete pre-execution run manifest."""
    conditions = selected_conditions(args.smoke)
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "git_commit": git_commit(),
        "protocol_commit": PROTOCOL_COMMIT,
        "reproduction_claim": (
            "paper-derived clean-room reimplementation; not code-identical "
            "or an exact numerical reproduction"
        ),
        "paper_title": (
            "Two Calm Ends and the Wild Middle: A Geometric Picture of "
            "Memorization in Diffusion Models"
        ),
        "paper_sha256": PAPER_SHA256,
        "corrected_plan_sha256": CORRECTED_PLAN_SHA256,
        "model_conditions": [condition.name for condition in conditions],
        "checkpoint_paths": {
            "edm_1k": str(EDM_1K_CHECKPOINT),
            "edm_50k": str(EDM_50K_CHECKPOINT),
        },
        "checkpoint_sha256": {
            "edm_1k": EDM_1K_SHA256,
            "edm_50k": EDM_50K_SHA256,
        },
        "model_source_repositories": {"edm": str(args.edm_root)},
        "model_source_commits": {"edm": provenance["edm_commit"]},
        "sampler": config["sampler"],
        "sigma_schedule": list(SIGMA_GRID),
        "terminal_sigma": TERMINAL_SIGMA,
        "swap_step_semantics": "inclusive denoiser-call indices 0..17",
        "condition_table": [
            {
                "condition": condition.name,
                "base_model": condition.base_model,
                "donor_model": condition.donor_model,
                "window": (
                    None
                    if condition.window is None
                    else {
                        "name": condition.window.name,
                        "start_index": condition.window.start_index,
                        "end_index": condition.window.end_index,
                        "start_sigma": condition.window.start_sigma,
                        "end_sigma": condition.window.end_sigma,
                    }
                ),
            }
            for condition in conditions
        ],
        "latent_seed_policy": config["sample_seeds"],
        "sample_seeds": [0, 1] if args.smoke else list(SAMPLE_SEEDS),
        "dataset_archive_identity": {
            "path": str(TRAIN_ARCHIVE),
            "sha256": TRAIN_ARCHIVE_SHA256,
        },
        "reference_subset_manifest": {
            "path": "data/e005_cifar10_subset_1k_indices.txt",
            "text_sha256": SUBSET_TEXT_SHA256,
            "little_endian_int64_sha256": SUBSET_INT64_SHA256,
        },
        "nearest_neighbor_metric": (
            "Euclidean L2 on unquantized RGB float64 NCHW-flattened vectors"
        ),
        "tensor_domain": "RGB scientific [-1,1] domain; no display transform",
        "output_clamping": False,
        "output_quantization": False,
        "batch_size": args.batch_size,
        "sampler_inference_batch_size": 1,
        "nearest_neighbor_batch_size": args.nn_batch_size,
        "device": str(device),
        "dependency_versions": dependency_versions(),
        "execution_host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_dir": str(output_dir),
        "config_path": str(args.config.resolve()),
        "config_sha256": provenance["config_sha256"],
        "provenance_checks": provenance,
        "started_at_unix": time.time(),
        "smoke": args.smoke,
    }


def dependency_versions() -> dict[str, str]:
    """Return execution dependency versions."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": getattr(PIL, "__version__", "unknown"),
    }


def git_commit() -> str:
    """Return the current repository commit."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def repository_commit(path: Path) -> str:
    """Return one source repository commit."""
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def dump_json(path: Path, payload: object) -> None:
    """Write stable indented JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]
) -> None:
    """Write rows with a frozen schema."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def per_sample_header() -> list[str]:
    """Return the frozen per-sample schema."""
    return [
        "experiment_id",
        "run_id",
        "condition",
        "base_model",
        "base_checkpoint_sha256",
        "donor_model",
        "donor_checkpoint_sha256",
        "window_name",
        "window_start_index",
        "window_end_index",
        "window_start_sigma",
        "window_end_sigma",
        "sampler",
        "sample_seed",
        "batch_index",
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


def nearest_neighbor_header() -> list[str]:
    """Return the frozen nearest-neighbor schema."""
    return [
        "experiment_id",
        "run_id",
        "condition",
        "sample_seed",
        "rank",
        "reference_index",
        "reference_subset_position",
        "distance",
    ]


def condition_summary_header() -> list[str]:
    """Return the frozen condition-summary schema."""
    return [
        "experiment_id",
        "run_id",
        "condition",
        "base_model",
        "donor_model",
        "window_name",
        "window_start_index",
        "window_end_index",
        "window_start_sigma",
        "window_end_sigma",
        "n_samples",
        "memorized_count",
        "memorization_rate",
        "ci95_low",
        "ci95_high",
        "mean_d1nn_over_d2nn",
        "median_d1nn_over_d2nn",
        "status",
    ]


def paired_comparison_header() -> list[str]:
    """Return the frozen paired-comparison schema."""
    return [
        "experiment_id",
        "run_id",
        "comparison",
        "base_model",
        "donor_model",
        "swap_condition",
        "baseline_condition",
        "window_name",
        "control_type",
        "n_pairs",
        "baseline_rate",
        "swap_rate",
        "rate_difference",
        "paired_mean_delta",
        "paired_ci95_low",
        "paired_ci95_high",
        "discordant_negative",
        "discordant_positive",
        "discordant_zero",
        "sign_test_p_value",
        "passes_practical_threshold",
        "status",
    ]


def failure_header() -> list[str]:
    """Return the frozen failure-report schema."""
    return [
        "experiment_id",
        "run_id",
        "condition",
        "sample_seed",
        "status",
        "error",
    ]


if __name__ == "__main__":
    main()
