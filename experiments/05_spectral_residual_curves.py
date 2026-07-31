#!/usr/bin/env python3
"""Experiment 5: decompose fixed-sigma denoising residual energy by frequency.

This is a paper-derived clean-room reimplementation. It evaluates the frozen
EDM-1K and matched EDM-50K checkpoints on known clean CIFAR-10 targets,
computes unquantized residuals, and applies complementary Fourier projections
directly to those residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import socket
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from _bootstrap import REPO_ROOT, SRC_ROOT
from spectral_diffusion_playground.e005_spectral_residuals import (
    BANDS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CUTOFFS,
    EXPERIMENT_ID,
    MASTER_NOISE_SEED,
    PRIMARY_CUTOFFS,
    REFERENCE_CUTOFF,
    RELATIVE_ENERGY_ATOL,
    RECONSTRUCTION_ATOL,
    REDUCTION_ELEMENTS,
    RUN_ID,
    SIGMA_GRID,
    aggregate_curves,
    aggregated_csv_header,
    attach_adjacent_cutoff_stability,
    compute_projection_energy,
    deterministic_noise,
    dump_json,
    extract_transition_window,
    raw_csv_header,
    seed_sequence_material,
    sha256_file,
    validate_projection_energy,
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
TRAIN_ARCHIVE = Path("/home/xggh8/datasets/edm/cifar10-32x32-train50k.zip")
FULL_ARCHIVE = Path("/home/xggh8/datasets/edm/cifar10-32x32.zip")
TRAIN_ARCHIVE_SHA256 = (
    "795cdc1444465ae4e19e25a0615d05ba0a0e83caa5db6b1b811deaf4c7910dfa"
)
FULL_ARCHIVE_SHA256 = "d47def5da86196bfa5825dba162b3724a9b02aea2067386fdba4ed47ce122ca5"
EDM_ROOT = Path("/home/xggh8/edm")


@dataclass(frozen=True)
class ModelCondition:
    """One clean-room denoiser condition."""

    name: str
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class SplitSpec:
    """One model/split image manifest."""

    name: str
    split_code: int
    image_indices: np.ndarray
    archive_indices: np.ndarray
    archive_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/xggh8/data/zw-lab/e005_spectral_residual_curves"),
    )
    parser.add_argument("--edm-root", type=Path, default=EDM_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--images-per-split", type=int, default=1000)
    parser.add_argument("--noise-repeats", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("edm_1k", "edm_50k"),
        default=["edm_1k", "edm_50k"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["train", "test"],
    )
    parser.add_argument(
        "--sigma-indices",
        nargs="+",
        type=int,
        default=list(range(len(SIGMA_GRID))),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the evaluator."""
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing to run E005 evaluation outside Slurm")
    validate_args(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.edm_root))
    manifest = build_initial_manifest(args, output_dir, device)
    dump_json(output_dir / "experiment_05_manifest.json", manifest)

    subset_indices = load_subset_indices()
    models = available_models()
    raw_path = output_dir / "experiment_05_bandwise_residuals.csv"
    identity = initialize_identity_report(args)
    aggregate_rows: list[dict[str, object]] = []
    transition_inputs: dict[tuple[str, str, int, str], np.ndarray] = {}

    with raw_path.open("w", newline="", encoding="utf-8") as raw_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=raw_csv_header())
        writer.writeheader()
        for model_name in args.models:
            model = models[model_name]
            validate_checkpoint(model)
            net = load_network(model, args.edm_root, device)
            for split_name in args.splits:
                split = build_split_spec(
                    model_name=model_name,
                    split_name=split_name,
                    subset_indices=subset_indices,
                    images_per_split=args.images_per_split,
                )
                images = load_images(split)
                energies = np.empty(
                    (
                        args.images_per_split,
                        args.noise_repeats,
                        len(SIGMA_GRID),
                        len(CUTOFFS),
                        len(BANDS),
                    ),
                    dtype=np.float64,
                )
                energies.fill(np.nan)
                evaluate_split(
                    net=net,
                    model=model,
                    split=split,
                    images=images,
                    args=args,
                    device=device,
                    writer=writer,
                    energies=energies,
                    identity=identity,
                )
                del images
                for cutoff_index, cutoff in enumerate(CUTOFFS):
                    if not np.all(np.isfinite(energies[:, :, :, cutoff_index, :])):
                        raise RuntimeError(
                            f"Missing finite energies for {model_name}/{split_name}/r={cutoff}"
                        )
                    rows = aggregate_curves(
                        energies[:, :, :, cutoff_index, :],
                        model=model.name,
                        split=split.name,
                        checkpoint_sha256=model.checkpoint_sha256,
                        cutoff=cutoff,
                        bootstrap_resamples=args.bootstrap_resamples,
                    )
                    aggregate_rows.extend(rows)
                    image_mean = energies[:, :, :, cutoff_index, :].mean(axis=(0, 1))
                    for band_index, band in enumerate(BANDS):
                        transition_inputs[(model.name, split.name, cutoff, band)] = (
                            image_mean[:, band_index]
                        )
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_path = output_dir / "experiment_05_aggregated_curves.csv"
    write_aggregate_csv(aggregate_path, aggregate_rows)
    identity["status"] = "pass"
    identity["runtime_sec"] = time.time() - start
    dump_json(output_dir / "experiment_05_identity_validation.json", identity)

    transitions = build_transition_summary(transition_inputs)
    dump_json(output_dir / "experiment_05_transition_windows.json", transitions)
    generate_figures(aggregate_rows, transitions, output_dir / "figures")
    manifest["completed_at_unix"] = time.time()
    manifest["outputs"] = {
        "raw_residuals": str(raw_path),
        "aggregated_curves": str(aggregate_path),
        "identity_validation": str(
            output_dir / "experiment_05_identity_validation.json"
        ),
        "transition_windows": str(output_dir / "experiment_05_transition_windows.json"),
        "figures": str(output_dir / "figures"),
    }
    manifest["identity_validation_sha256"] = sha256_file(
        output_dir / "experiment_05_identity_validation.json"
    )
    manifest["transition_windows_sha256"] = sha256_file(
        output_dir / "experiment_05_transition_windows.json"
    )
    dump_json(output_dir / "experiment_05_manifest.json", manifest)


def validate_args(args: argparse.Namespace) -> None:
    """Validate bounded smoke/full arguments."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.noise_repeats <= 0 or args.noise_repeats > 8:
        raise ValueError("--noise-repeats must be in [1, 8]")
    if args.images_per_split <= 0 or args.images_per_split > 1000:
        raise ValueError("--images-per-split must be in [1, 1000]")
    if any(index < 0 or index >= len(SIGMA_GRID) for index in args.sigma_indices):
        raise ValueError("--sigma-indices must reference the frozen 18-point grid")
    if not args.smoke:
        if args.images_per_split != 1000:
            raise ValueError("Full E005 requires exactly 1000 images per split")
        if args.noise_repeats != 8:
            raise ValueError("Full E005 requires exactly 8 noise repeats")
        if args.sigma_indices != list(range(len(SIGMA_GRID))):
            raise ValueError("Full E005 requires the complete 18-point sigma grid")
        if set(args.models) != {"edm_1k", "edm_50k"}:
            raise ValueError("Full E005 requires both matched model conditions")
        if set(args.splits) != {"train", "test"}:
            raise ValueError("Full E005 requires train and test splits")


def available_models() -> dict[str, ModelCondition]:
    """Return the frozen model conditions."""
    return {
        "edm_1k": ModelCondition("edm_1k", EDM_1K_CHECKPOINT, EDM_1K_SHA256),
        "edm_50k": ModelCondition("edm_50k", EDM_50K_CHECKPOINT, EDM_50K_SHA256),
    }


def validate_checkpoint(model: ModelCondition) -> None:
    """Require the checkpoint file to match the frozen SHA-256."""
    observed = sha256_file(model.checkpoint_path)
    if observed != model.checkpoint_sha256:
        raise RuntimeError(
            f"{model.name} checkpoint SHA mismatch: {observed} != "
            f"{model.checkpoint_sha256}"
        )


def load_network(
    model: ModelCondition,
    edm_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load the EMA denoiser from a frozen EDM checkpoint."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    with model.checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    net = checkpoint["ema"].to(device).eval()
    return net


def load_subset_indices() -> np.ndarray:
    """Load the frozen clean-room EDM-1K subset indices."""
    path = REPO_ROOT / "data/e005_cifar10_subset_1k_indices.txt"
    values = np.loadtxt(path, dtype=np.int64)
    if values.shape != (1000,):
        raise RuntimeError(f"Unexpected subset shape: {values.shape}")
    return values


def build_split_spec(
    *,
    model_name: str,
    split_name: str,
    subset_indices: np.ndarray,
    images_per_split: int,
) -> SplitSpec:
    """Build the frozen image manifest for one model/split."""
    positions = np.arange(images_per_split, dtype=np.int64)
    if split_name == "test":
        return SplitSpec(
            name="test",
            split_code=1,
            image_indices=positions.copy(),
            archive_indices=50_000 + positions,
            archive_path=FULL_ARCHIVE,
        )
    if model_name == "edm_1k":
        image_indices = subset_indices[:images_per_split]
    else:
        image_indices = positions.copy()
    return SplitSpec(
        name="train",
        split_code=0,
        image_indices=image_indices.copy(),
        archive_indices=image_indices.copy(),
        archive_path=TRAIN_ARCHIVE,
    )


def load_images(split: SplitSpec) -> np.ndarray:
    """Load one split as NCHW float64 in the scientific [-1, 1] domain."""
    expected_hash = (
        FULL_ARCHIVE_SHA256
        if split.archive_path == FULL_ARCHIVE
        else TRAIN_ARCHIVE_SHA256
    )
    observed_hash = sha256_file(split.archive_path)
    if observed_hash != expected_hash:
        raise RuntimeError(f"Archive SHA mismatch for {split.archive_path}")
    wanted = {
        int(archive_index): pos
        for pos, archive_index in enumerate(split.archive_indices)
    }
    images = np.empty((len(wanted), 3, 32, 32), dtype=np.float64)
    with zipfile.ZipFile(split.archive_path) as archive:
        png_names = sorted(name for name in archive.namelist() if name.endswith(".png"))
        for archive_index, position in wanted.items():
            name = png_names[archive_index]
            with archive.open(name) as handle:
                image = PIL.Image.open(handle).convert("RGB")
                array = np.asarray(image, dtype=np.float64)
            images[position] = np.transpose(2.0 * (array / 255.0) - 1.0, (2, 0, 1))
    return images


def evaluate_split(
    *,
    net: torch.nn.Module,
    model: ModelCondition,
    split: SplitSpec,
    images: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    writer: csv.DictWriter[str],
    energies: np.ndarray,
    identity: dict[str, Any],
) -> None:
    """Evaluate one model/split and append raw rows."""
    tasks = [
        (image_pos, repeat, sigma_index)
        for image_pos in range(args.images_per_split)
        for repeat in range(args.noise_repeats)
        for sigma_index in args.sigma_indices
    ]
    progress = tqdm(
        range(0, len(tasks), args.batch_size),
        desc=f"{model.name}/{split.name}",
        leave=False,
    )
    for start in progress:
        batch_tasks = tasks[start : start + args.batch_size]
        clean_batch = np.stack([images[image_pos] for image_pos, _, _ in batch_tasks])
        noise_batch = np.empty_like(clean_batch, dtype=np.float64)
        sigma_values = np.empty((len(batch_tasks),), dtype=np.float32)
        noise_seeds: list[int] = []
        for local_index, (image_pos, repeat, sigma_index) in enumerate(batch_tasks):
            noise, derived_seed = deterministic_noise(
                clean_batch[local_index].shape,
                split_code=split.split_code,
                dataset_index=int(split.image_indices[image_pos]),
                noise_repeat=repeat,
                sigma_index=sigma_index,
            )
            noise_batch[local_index] = noise
            noise_seeds.append(derived_seed)
            sigma_values[local_index] = np.float32(SIGMA_GRID[sigma_index])
        x_sigma = (
            clean_batch
            + sigma_values[:, None, None, None].astype(np.float64) * noise_batch
        )
        with torch.no_grad():
            model_input = torch.from_numpy(x_sigma.astype(np.float32)).to(device)
            sigmas = torch.from_numpy(sigma_values).to(device)
            output = net(model_input, sigmas, class_labels=None)
            if not torch.isfinite(output).all().item():
                raise RuntimeError(
                    f"Nonfinite model output for {model.name}/{split.name}"
                )
            output_np = output.detach().cpu().numpy().astype(np.float64)
        residuals = output_np - clean_batch
        for local_index, (image_pos, repeat, sigma_index) in enumerate(batch_tasks):
            residual_hwc = np.transpose(residuals[local_index], (1, 2, 0))
            for cutoff_index, cutoff in enumerate(CUTOFFS):
                energy = compute_projection_energy(residual_hwc, cutoff)
                validate_projection_energy(energy)
                energies[image_pos, repeat, sigma_index, cutoff_index, :] = [
                    energy.full,
                    energy.low,
                    energy.high,
                ]
                identity["observed_rows"] += 1
                identity["max_reconstruction_error"] = max(
                    identity["max_reconstruction_error"],
                    energy.reconstruction_max_abs_error,
                )
                identity["max_additivity_absolute_error"] = max(
                    identity["max_additivity_absolute_error"],
                    energy.additivity_absolute_error,
                )
                identity["max_additivity_relative_error"] = max(
                    identity["max_additivity_relative_error"],
                    energy.additivity_relative_error,
                )
                identity["max_orthogonality_relative_error"] = max(
                    identity["max_orthogonality_relative_error"],
                    energy.orthogonality_relative_error,
                )
                writer.writerow(
                    raw_row(
                        model=model,
                        split=split,
                        image_pos=image_pos,
                        repeat=repeat,
                        noise_seed=noise_seeds[local_index],
                        sigma_index=sigma_index,
                        cutoff=cutoff,
                        energy=energy,
                    )
                )


def raw_row(
    *,
    model: ModelCondition,
    split: SplitSpec,
    image_pos: int,
    repeat: int,
    noise_seed: int,
    sigma_index: int,
    cutoff: int,
    energy: Any,
) -> dict[str, object]:
    """Build one per-sample CSV row."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "model": model.name,
        "checkpoint_sha256": model.checkpoint_sha256,
        "split": split.name,
        "image_index": int(split.image_indices[image_pos]),
        "image_manifest_position": image_pos,
        "noise_repeat": repeat,
        "noise_seed": noise_seed,
        "sigma_index": sigma_index,
        "sigma": f"{SIGMA_GRID[sigma_index]:.17g}",
        "sigma_grid": "edm_18_rho7",
        "cutoff": cutoff,
        "cutoff_normalized": f"{cutoff / 16.0:.17g}",
        "reduction_elements": REDUCTION_ELEMENTS,
        "full_squared_error": f"{energy.full:.17g}",
        "low_squared_error": f"{energy.low:.17g}",
        "high_squared_error": f"{energy.high:.17g}",
        "full_mean_squared_error": f"{energy.full_mse:.17g}",
        "low_mean_squared_error": f"{energy.low_mse:.17g}",
        "high_mean_squared_error": f"{energy.high_mse:.17g}",
        "reconstruction_max_abs_error": f"{energy.reconstruction_max_abs_error:.17g}",
        "energy_additivity_absolute_error": f"{energy.additivity_absolute_error:.17g}",
        "energy_additivity_relative_error": f"{energy.additivity_relative_error:.17g}",
        "orthogonality_relative_error": f"{energy.orthogonality_relative_error:.17g}",
        "status": "ok",
        "error": "",
    }


def write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write aggregate curve rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregated_csv_header())
        writer.writeheader()
        writer.writerows(rows)


def build_transition_summary(
    transition_inputs: dict[tuple[str, str, int, str], np.ndarray],
) -> dict[str, object]:
    """Extract frozen transition windows from EDM-1K held-out test curves."""
    transitions: dict[str, dict[str, dict[str, object]]] = {
        "low_frequency_residual": {},
        "high_frequency_residual": {},
    }
    for band in transitions:
        for cutoff in CUTOFFS:
            curve = transition_inputs[("edm_1k", "test", cutoff, band)]
            transitions[band][str(cutoff)] = extract_transition_window(curve)
    attach_adjacent_cutoff_stability(transitions)
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_curve": "edm_1k/test",
        "reference_cutoff": REFERENCE_CUTOFF,
        "primary_cutoffs": list(PRIMARY_CUTOFFS),
        "optional_extended_cutoff": 6,
        "transitions": transitions,
    }


def build_initial_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Build the run manifest before inference."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "git_commit": git_commit(),
        "protocol_commit": "533ec4605cef58186363f93c4ec88edf5ef1d387",
        "reproduction_claim": (
            "paper-derived clean-room reimplementation; not an exact paper "
            "reproduction"
        ),
        "paper_title": (
            "Two Calm Ends and the Wild Middle: A Geometric Picture of "
            "Memorization in Diffusion Models"
        ),
        "model_conditions": args.models,
        "checkpoint_paths": {
            "edm_1k": str(EDM_1K_CHECKPOINT),
            "edm_50k": str(EDM_50K_CHECKPOINT),
        },
        "checkpoint_sha256": {
            "edm_1k": EDM_1K_SHA256,
            "edm_50k": EDM_50K_SHA256,
        },
        "model_source_repositories": {"edm": str(args.edm_root)},
        "model_source_commits": {"edm": edm_commit(args.edm_root)},
        "model_call_semantics": {
            "call": "checkpoint['ema'](x_sigma_nchw_float32, sigma_float32, class_labels=None)",
            "output": "unquantized unclamped NCHW float32 clean-image prediction",
        },
        "dataset_archive_integrity": {
            str(TRAIN_ARCHIVE): TRAIN_ARCHIVE_SHA256,
            str(FULL_ARCHIVE): FULL_ARCHIVE_SHA256,
        },
        "train_index_manifests": {
            "edm_1k": "data/e005_cifar10_subset_1k_indices.txt",
            "edm_50k": "canonical CIFAR-10 train indices 0..999",
        },
        "test_indices": "canonical CIFAR-10 test indices 0..999 from archive indices 50000..50999",
        "image_domain": "HWC/NCHW RGB float64 in [-1, 1]",
        "model_domain": "NCHW RGB float32 in [-1, 1] plus Gaussian noise",
        "layout_conversion": "scientific HWC residuals projected after NCHW model output conversion",
        "model_dtype": "float32",
        "autocast": False,
        "output_clamping": False,
        "output_quantization": False,
        "primary_sigma_formula": "EDM 18-point rho=7 schedule from 80 to 0.002",
        "primary_sigma_values": list(SIGMA_GRID),
        "dense_grid_status": "not_run",
        "cutoffs": list(CUTOFFS),
        "master_noise_seed": MASTER_NOISE_SEED,
        "noise_generator": "NumPy PCG64DXSM with SeedSequence material",
        "noise_repeats": args.noise_repeats,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": args.bootstrap_resamples,
        "host": socket.gethostname(),
        "device": str(device),
        "dependency_versions": dependency_versions(),
        "output_dir": str(output_dir),
        "started_at_unix": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "smoke": args.smoke,
    }


def initialize_identity_report(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize the identity-validation report."""
    expected_rows = (
        len(args.models)
        * len(args.splits)
        * args.images_per_split
        * args.noise_repeats
        * len(args.sigma_indices)
        * len(CUTOFFS)
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "expected_rows": expected_rows,
        "observed_rows": 0,
        "duplicate_keys": [],
        "missing_keys": [],
        "max_reconstruction_error": 0.0,
        "max_additivity_absolute_error": 0.0,
        "max_additivity_relative_error": 0.0,
        "max_orthogonality_relative_error": 0.0,
        "reconstruction_tolerance": RECONSTRUCTION_ATOL,
        "relative_energy_tolerance": RELATIVE_ENERGY_ATOL,
        "failed_rows": [],
        "nonfinite_rows": [],
        "status": "running",
    }


def git_commit() -> str:
    """Return the current repository commit."""
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def edm_commit(edm_root: Path) -> str:
    """Return the EDM source commit."""
    import subprocess

    return subprocess.run(
        ["git", "-C", str(edm_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def dependency_versions() -> dict[str, str]:
    """Return versions of runtime dependencies relevant to the run."""
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
    }


def generate_figures(
    aggregate_rows: list[dict[str, object]],
    transitions: dict[str, object],
    figures_dir: Path,
) -> None:
    """Generate required E005 figures from validated aggregates."""
    data = normalize_aggregate_rows(aggregate_rows)
    plot_model_band_curves(
        data,
        figures_dir / "experiment_05_edm1k_low_high_residual_curves.png",
        model="edm_1k",
        title="EDM-1K spectral residual curves",
    )
    plot_model_band_curves(
        data,
        figures_dir / "experiment_05_edm50k_low_high_residual_curves.png",
        model="edm_50k",
        title="EDM-50K spectral residual curves",
    )
    plot_train_test_comparison(
        data,
        figures_dir / "experiment_05_train_test_comparison.png",
    )
    plot_cutoff_sensitivity(
        data,
        figures_dir / "experiment_05_cutoff_sensitivity.png",
    )
    plot_transition_windows(
        data,
        transitions,
        figures_dir / "experiment_05_transition_windows.png",
    )
    plot_additivity_diagnostics(
        figures_dir / "experiment_05_additivity_diagnostics.png",
        Path(figures_dir).parent / "experiment_05_identity_validation.json",
    )


def normalize_aggregate_rows(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, int, str], np.ndarray]:
    """Convert aggregate rows to keyed arrays of mean summed energy."""
    data: dict[tuple[str, str, int, str], np.ndarray] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["split"]),
            int(row["cutoff"]),
            str(row["band"]),
        )
        data.setdefault(key, np.empty(len(SIGMA_GRID), dtype=np.float64))
        data[key][int(row["sigma_index"])] = float(row["mean_summed_squared_error"])
    return data


def plot_model_band_curves(
    data: dict[tuple[str, str, int, str], np.ndarray],
    path: Path,
    *,
    model: str,
    title: str,
) -> None:
    """Plot low/high residual curves for one model at r=4."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for split, linestyle in (("train", "-"), ("test", "--")):
        ax.plot(
            SIGMA_GRID,
            data[(model, split, REFERENCE_CUTOFF, "low_frequency_residual")],
            linestyle=linestyle,
            marker="o",
            label=f"{split}: Low-frequency residual energy — general-structure proxy",
        )
        ax.plot(
            SIGMA_GRID,
            data[(model, split, REFERENCE_CUTOFF, "high_frequency_residual")],
            linestyle=linestyle,
            marker="s",
            label=f"{split}: High-frequency residual energy — fine-detail proxy",
        )
    format_sigma_axis(ax, title)
    ax.set_ylabel("Mean summed squared residual energy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_train_test_comparison(
    data: dict[tuple[str, str, int, str], np.ndarray],
    path: Path,
) -> None:
    """Plot train/test comparisons by model and band."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, band, label in (
        (
            axes[0],
            "low_frequency_residual",
            "Low-frequency residual energy — general-structure proxy",
        ),
        (
            axes[1],
            "high_frequency_residual",
            "High-frequency residual energy — fine-detail proxy",
        ),
    ):
        for model in ("edm_1k", "edm_50k"):
            for split, linestyle in (("train", "-"), ("test", "--")):
                ax.plot(
                    SIGMA_GRID,
                    data[(model, split, REFERENCE_CUTOFF, band)],
                    linestyle=linestyle,
                    marker="o",
                    label=f"{model} {split}",
                )
        format_sigma_axis(ax, label)
        ax.set_ylabel("Mean summed squared residual energy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cutoff_sensitivity(
    data: dict[tuple[str, str, int, str], np.ndarray],
    path: Path,
) -> None:
    """Plot r=3,4,5 cutoff sensitivity with r=6 as extended context."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, band, label in (
        (
            axes[0],
            "low_frequency_residual",
            "Low-frequency residual energy — general-structure proxy",
        ),
        (
            axes[1],
            "high_frequency_residual",
            "High-frequency residual energy — fine-detail proxy",
        ),
    ):
        for cutoff in CUTOFFS:
            alpha = 1.0 if cutoff in PRIMARY_CUTOFFS else 0.45
            ax.plot(
                SIGMA_GRID,
                data[("edm_1k", "test", cutoff, band)],
                marker="o",
                alpha=alpha,
                label=f"r={cutoff}" if cutoff != 6 else "r=6 extended",
            )
        format_sigma_axis(ax, f"Cutoff sensitivity: {label}")
        ax.set_ylabel("Mean summed squared residual energy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transition_windows(
    data: dict[tuple[str, str, int, str], np.ndarray],
    transitions: dict[str, object],
    path: Path,
) -> None:
    """Plot normalized transition curves with frozen annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    transition_data = transitions["transitions"]
    for ax, band, label in (
        (
            axes[0],
            "low_frequency_residual",
            "Low-frequency residual energy — general-structure proxy",
        ),
        (
            axes[1],
            "high_frequency_residual",
            "High-frequency residual energy — fine-detail proxy",
        ),
    ):
        for cutoff in CUTOFFS:
            record = transition_data[band][str(cutoff)]
            if "normalized_recovery" not in record:
                continue
            y = np.asarray(record["normalized_recovery"], dtype=np.float64)
            ax.plot(SIGMA_GRID, y, marker="o", label=f"r={cutoff}")
            if record["status"] == "ok":
                ax.axvspan(
                    record["entry_sigma"],
                    record["exit_sigma"],
                    alpha=0.08,
                    color="black",
                )
        ax.axhline(0.2, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=1)
        format_sigma_axis(ax, f"Transition summary: {label}")
        ax.set_ylabel("Normalized residual-energy recovery")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_additivity_diagnostics(path: Path, identity_path: Path) -> None:
    """Plot max identity diagnostics."""
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    labels = [
        "reconstruction",
        "additivity abs",
        "additivity rel",
        "orthogonality rel",
    ]
    values = [
        identity["max_reconstruction_error"],
        identity["max_additivity_absolute_error"],
        identity["max_additivity_relative_error"],
        identity["max_orthogonality_relative_error"],
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_title("Full-residual additivity diagnostics")
    ax.set_ylabel("Maximum observed error")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_sigma_axis(ax: plt.Axes, title: str) -> None:
    """Format a sigma-axis plot in high-to-low order."""
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("sigma, descending high-to-low noise")
    ax.set_title(title, fontsize=11)


if __name__ == "__main__":
    main()
