#!/usr/bin/env python3
"""Run E010 directional whole-denoiser memorization-transfer swaps."""

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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from spectral_diffusion_playground.e010_directional_transfer import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_RECORDS,
    EXPERIMENT_ID,
    SAMPLE_SEEDS,
    DirectionalCondition,
    formal_outcomes,
    frozen_conditions,
    nearest_two_cpu,
    target_control_summary,
    transition_category,
    validate_condition_registry,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/e010_directional_memorization_transfer.json"
PER_SAMPLE = "experiment_10_per_sample.csv"
FAILURES = "experiment_10_failures.csv"


def parse_args() -> argparse.Namespace:
    """Parse the guarded E010 interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("preflight", "smoke", "full", "summarize", "plot-only"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--edm-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return one file SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def dump_json(path: Path, payload: object) -> None:
    """Write stable strict JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_commit() -> str:
    """Return the current repository commit."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_execution_commit() -> str:
    """Require an exact clean checkout pinned by the launcher."""
    expected = os.environ.get("E010_REPO_COMMIT", "")
    if len(expected) != 40:
        raise RuntimeError("E010_REPO_COMMIT must contain the pinned commit")
    observed = git_commit()
    if observed != expected:
        raise RuntimeError(f"Execution commit mismatch: {observed} != {expected}")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise RuntimeError("E010 requires a clean repository checkout")
    return observed


def validate_static_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every committed E010 manifest and frozen registry."""
    config = load_json(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Unexpected experiment configuration")
    observed: dict[str, Any] = {"config_sha256": sha256_file(config_path)}
    manifests: dict[str, dict[str, Any]] = {}
    for name, path_key, hash_key in (
        ("model_pair", "model_pair_manifest", "model_pair_manifest_sha256"),
        ("conditions", "condition_manifest", "condition_manifest_sha256"),
        ("seeds", "seed_manifest", "seed_manifest_sha256"),
        ("geometry", "geometry_target", "geometry_target_sha256"),
    ):
        path = REPO_ROOT / config["inputs"][path_key]
        digest = sha256_file(path)
        if digest != config["inputs"][hash_key]:
            raise RuntimeError(f"Frozen {name} hash mismatch")
        manifests[name] = load_json(path)
        observed[f"{name}_sha256"] = digest

    condition_rows = manifests["conditions"]["conditions"]
    registered = [DirectionalCondition(**row) for row in condition_rows]
    registered = [
        DirectionalCondition(
            item.condition_id,
            item.direction,
            item.band,
            item.role,
            item.recipient,
            item.donor,
            tuple(item.swap_indices),
        )
        for item in registered
    ]
    validate_condition_registry(registered)
    seed_manifest = manifests["seeds"]
    if (
        seed_manifest["seed_start"] != SAMPLE_SEEDS[0]
        or seed_manifest["seed_end"] != SAMPLE_SEEDS[-1]
        or seed_manifest["expected_record_count"] != EXPECTED_RECORDS
    ):
        raise RuntimeError("Frozen seed manifest mismatch")
    geometry = manifests["geometry"]
    if geometry["low_band_geometry_target_indices"] != [8]:
        raise RuntimeError("E004B low target changed")
    if geometry["high_band_geometry_target_indices"] != [9, 10]:
        raise RuntimeError("E004B high target changed")
    return config, {"hashes": observed, "manifests": manifests}


def validate_external_inputs(
    config: Mapping[str, Any], frozen: Mapping[str, Any], edm_root: Path
) -> dict[str, Any]:
    """Hash checkpoints, dataset, subset, and inspect model compatibility."""
    pair = frozen["manifests"]["model_pair"]
    checkpoints = {}
    identities = {}
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    for role in ("memorizing_model", "generalizing_model"):
        record = pair[role]
        path = Path(record["canonical_path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Checkpoint identity mismatch: {path}")
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Checkpoint size mismatch: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        network = payload.get("ema")
        if network is None:
            raise RuntimeError(f"EMA missing from checkpoint: {path}")
        identity = {
            "img_resolution": int(getattr(network, "img_resolution", -1)),
            "img_channels": int(getattr(network, "img_channels", -1)),
            "label_dim": int(getattr(network, "label_dim", -1)),
        }
        if identity != {"img_resolution": 32, "img_channels": 3, "label_dim": 0}:
            raise RuntimeError(f"Incompatible model identity: {identity}")
        checkpoints[record["model_id"]] = str(path)
        identities[record["model_id"]] = identity
        del network, payload
    if len({json.dumps(value, sort_keys=True) for value in identities.values()}) != 1:
        raise RuntimeError("Model architectures differ")

    reference = config["reference"]
    archive = Path(reference["archive"])
    subset = REPO_ROOT / reference["subset_indices"]
    if sha256_file(archive) != reference["archive_sha256"]:
        raise RuntimeError("CIFAR-10 archive hash mismatch")
    if sha256_file(subset) != reference["subset_text_sha256"]:
        raise RuntimeError("Reference subset text hash mismatch")
    indices = np.loadtxt(subset, dtype=np.int64)
    int_hash = hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    if indices.shape != (1000,) or int_hash != reference["subset_int64_le_sha256"]:
        raise RuntimeError("Reference subset content mismatch")
    return {
        "checkpoints": checkpoints,
        "model_identities": identities,
        "archive_sha256": reference["archive_sha256"],
        "subset_text_sha256": reference["subset_text_sha256"],
        "subset_int64_le_sha256": int_hash,
        "edm_commit": subprocess.check_output(
            ["git", "-C", str(edm_root), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def configure_determinism() -> None:
    """Enable deterministic generation operations."""
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_networks(
    frozen: Mapping[str, Any], device: torch.device, edm_root: Path
) -> dict[str, torch.nn.Module]:
    """Load the two registered unconditional EMA networks."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    result = {}
    pair = frozen["manifests"]["model_pair"]
    for role in ("memorizing_model", "generalizing_model"):
        record = pair[role]
        with Path(record["canonical_path"]).open("rb") as handle:
            network = pickle.load(handle)["ema"].to(device).eval()
        result[record["model_id"]] = network
    return result


def seeded_latent(seed: int, device: torch.device) -> torch.Tensor:
    """Create one canonical latent with an isolated device generator."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (1, 3, 32, 32), generator=generator, device=device, dtype=torch.float32
    )


def sample_condition(
    networks: Mapping[str, torch.nn.Module],
    condition: DirectionalCondition,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Run one frozen 18-call pure-Euler directional trajectory."""
    state = seeded_latent(seed, device).to(torch.float64) * SIGMA_GRID[0]
    schedule = SIGMA_GRID + (TERMINAL_SIGMA,)
    with torch.no_grad():
        for step, (sigma, sigma_next) in enumerate(zip(schedule[:-1], schedule[1:])):
            model_id = condition.model_for_step(step)
            sigma_tensor = torch.full((1,), sigma, dtype=torch.float64, device=device)
            denoised = networks[model_id](state, sigma_tensor, class_labels=None)
            denoised = denoised.to(torch.float64)
            if not torch.isfinite(denoised).all().item():
                raise RuntimeError(
                    f"Nonfinite denoiser output at seed={seed}, step={step}"
                )
            state = state + (sigma_next - sigma) * (state - denoised) / sigma
    if not torch.isfinite(state).all().item():
        raise RuntimeError(f"Nonfinite final sample at seed={seed}")
    return state[0].detach().cpu().numpy().astype(np.float64, copy=False)


def load_reference(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen CIFAR-10 1K reference in NCHW [-1,1]."""
    indices = np.loadtxt(
        REPO_ROOT / config["reference"]["subset_indices"], dtype=np.int64
    )
    images = np.empty((1000, 3, 32, 32), dtype=np.float64)
    with zipfile.ZipFile(config["reference"]["archive"]) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".png"))
        for position, index in enumerate(indices):
            with archive.open(names[int(index)]) as handle:
                image = np.asarray(
                    PIL.Image.open(handle).convert("RGB"), dtype=np.float64
                )
            images[position] = np.transpose(image / 127.5 - 1.0, (2, 0, 1))
    return images, indices


def per_sample_header() -> list[str]:
    """Return the frozen row schema."""
    return [
        "experiment_id",
        "condition_id",
        "direction",
        "band",
        "role",
        "recipient_checkpoint_sha256",
        "donor_checkpoint_sha256",
        "swap_indices",
        "sample_seed",
        "generated_sample_hash",
        "d1nn",
        "d2nn",
        "d1nn_over_d2nn",
        "d1nn_reference_index",
        "d2nn_reference_index",
        "memorized",
        "sampler",
        "status",
        "error",
    ]


def evaluate(
    networks: Mapping[str, torch.nn.Module],
    condition: DirectionalCondition,
    seeds: Sequence[int],
    reference: np.ndarray,
    reference_indices: np.ndarray,
    frozen: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """Generate and deterministically score one registered condition."""
    pair = frozen["manifests"]["model_pair"]
    hashes = {
        pair["memorizing_model"]["model_id"]: pair["memorizing_model"]["sha256"],
        pair["generalizing_model"]["model_id"]: pair["generalizing_model"]["sha256"],
    }
    rows = []
    samples = []
    for seed in seeds:
        try:
            sample = sample_condition(networks, condition, seed, device)
            d1, d2, first, second = nearest_two_cpu(sample, reference)
            ratio = d1 / d2 if d2 > 0.0 else float("inf")
            row = {
                "experiment_id": EXPERIMENT_ID,
                "condition_id": condition.condition_id,
                "direction": condition.direction,
                "band": condition.band,
                "role": condition.role,
                "recipient_checkpoint_sha256": hashes[condition.recipient],
                "donor_checkpoint_sha256": (
                    hashes[condition.donor] if condition.donor else ""
                ),
                "swap_indices": json.dumps(
                    list(condition.swap_indices), separators=(",", ":")
                ),
                "sample_seed": seed,
                "generated_sample_hash": generated_sample_hash(sample),
                "d1nn": f"{d1:.17g}",
                "d2nn": f"{d2:.17g}",
                "d1nn_over_d2nn": f"{ratio:.17g}",
                "d1nn_reference_index": int(reference_indices[first]),
                "d2nn_reference_index": int(reference_indices[second]),
                "memorized": int(d1 < d2 / 3.0),
                "sampler": "pure_euler_18_call_no_churn",
                "status": "ok",
                "error": "",
            }
            samples.append(sample)
        except Exception as error:
            row = {key: "" for key in per_sample_header()}
            row.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "condition_id": condition.condition_id,
                    "direction": condition.direction,
                    "band": condition.band,
                    "role": condition.role,
                    "sample_seed": seed,
                    "memorized": 0,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        rows.append(row)
    return rows, samples


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], header: Sequence[str]
) -> None:
    """Write a CSV using a fixed schema."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_manifest(
    mode: str,
    output_dir: Path,
    commit: str,
    config_path: Path,
    frozen: Mapping[str, Any],
    external: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build pre-execution provenance."""
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "repository_commit": commit,
        "config_path": str(config_path.resolve()),
        "config_sha256": frozen["hashes"]["config_sha256"],
        "frozen_input_hashes": frozen["hashes"],
        "external_provenance": external,
        "condition_count": 14,
        "sample_seeds": list(SAMPLE_SEEDS if mode == "full" else SAMPLE_SEEDS[:2]),
        "expected_records": EXPECTED_RECORDS if mode == "full" else 28,
        "whole_denoiser_swap": True,
        "frequency_component_swap": False,
        "training_executed": False,
        "e008_status": "BLOCKED_UNEXECUTED",
        "device": str(device),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "output_dir": str(output_dir),
        "started_at_unix": time.time(),
    }


def execute_run(
    mode: str,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    external: Mapping[str, Any],
    commit: str,
    edm_root: Path,
    resume: bool,
) -> None:
    """Run the isolated smoke or complete frozen E010 evaluation."""
    if output_dir.exists() and not resume:
        raise RuntimeError(f"Refusing existing output directory: {output_dir}")
    if resume and mode != "full":
        raise RuntimeError("Resume is permitted only for the frozen full run")
    output_dir.mkdir(parents=True, exist_ok=resume)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("E010 inference requires CUDA")
    configure_determinism()
    manifest_path = output_dir / "experiment_10_manifest.json"
    if resume:
        manifest = load_json(manifest_path)
        if (
            manifest["repository_commit"] != commit
            or manifest["config_sha256"] != frozen["hashes"]["config_sha256"]
            or manifest["mode"] != "full"
        ):
            raise RuntimeError("Existing E010 output is incompatible with resume")
    else:
        manifest = build_manifest(
            mode, output_dir, commit, config_path, frozen, external, device
        )
        dump_json(manifest_path, manifest)
    reference, reference_indices = load_reference(config)
    networks = load_networks(frozen, device, edm_root)
    seeds = SAMPLE_SEEDS[:2] if mode == "smoke" else SAMPLE_SEEDS
    rows: list[dict[str, object]] = (
        [dict(row) for row in read_rows(output_dir / PER_SAMPLE)]
        if resume and (output_dir / PER_SAMPLE).is_file()
        else []
    )
    sample_root = output_dir / "samples"
    sample_root.mkdir(exist_ok=True)
    samples_by_condition: dict[str, np.ndarray] = {}
    for registered in frozen_conditions():
        files = [
            sample_root / registered.condition_id / f"seed_{seed}.npy" for seed in seeds
        ]
        if all(path.is_file() for path in files):
            samples_by_condition[registered.condition_id] = np.stack(
                [np.load(path) for path in files]
            )
    successful_keys = {
        (str(row["condition_id"]), int(row["sample_seed"]))
        for row in rows
        if row["status"] == "ok"
    }
    for condition in frozen_conditions():
        missing = [
            seed
            for seed in seeds
            if (condition.condition_id, seed) not in successful_keys
        ]
        if not missing:
            continue
        condition_rows, samples = evaluate(
            networks,
            condition,
            missing,
            reference,
            reference_indices,
            frozen,
            device,
        )
        missing_set = set(missing)
        rows = [
            row
            for row in rows
            if not (
                row["condition_id"] == condition.condition_id
                and int(row["sample_seed"]) in missing_set
            )
        ]
        rows.extend(condition_rows)
        if samples:
            successful_rows = [row for row in condition_rows if row["status"] == "ok"]
            condition_dir = sample_root / condition.condition_id
            condition_dir.mkdir(exist_ok=True)
            for row, sample in zip(successful_rows, samples):
                np.save(condition_dir / f"seed_{int(row['sample_seed'])}.npy", sample)
        condition_files = [
            sample_root / condition.condition_id / f"seed_{seed}.npy" for seed in seeds
        ]
        if all(path.is_file() for path in condition_files):
            samples_by_condition[condition.condition_id] = np.stack(
                [np.load(path) for path in condition_files]
            )
        rows.sort(key=lambda row: (str(row["condition_id"]), int(row["sample_seed"])))
        write_csv(output_dir / PER_SAMPLE, rows, per_sample_header())
        failures = [row for row in rows if row["status"] != "ok"]
        write_csv(output_dir / FAILURES, failures, per_sample_header())
        np.savez_compressed(
            output_dir / "experiment_10_generated_samples.npz",
            **samples_by_condition,
        )
    failures = [row for row in rows if row["status"] != "ok"]
    if mode == "smoke":
        repeat_rows, _ = evaluate(
            networks,
            frozen_conditions()[0],
            seeds[:1],
            reference,
            reference_indices,
            frozen,
            device,
        )
        original = next(
            row
            for row in rows
            if row["condition_id"] == "A0" and row["sample_seed"] == seeds[0]
        )
        exact = original == repeat_rows[0]
        dump_json(
            output_dir / "experiment_10_smoke_validation.json",
            {
                "status": (
                    "pass" if not failures and exact and len(rows) == 28 else "fail"
                ),
                "record_count": len(rows),
                "failure_count": len(failures),
                "exact_repeat_A0_seed_40000": exact,
                "sample_hash_equal": original["generated_sample_hash"]
                == repeat_rows[0]["generated_sample_hash"],
            },
        )
        if failures or not exact or len(rows) != 28:
            raise RuntimeError("E010 smoke validation failed")
    else:
        summarize(output_dir)
    manifest["completed_at_unix"] = time.time()
    manifest["observed_records"] = len(rows)
    manifest["failure_count"] = len(failures)
    manifest["outputs"] = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("experiment_10_*"))
        if path.name != "experiment_10_manifest.json"
    }
    dump_json(output_dir / "experiment_10_manifest.json", manifest)


def grouped_vectors(rows: Sequence[Mapping[str, str]]) -> dict[str, np.ndarray]:
    """Return condition vectors aligned to the frozen seed order."""
    by_condition = {}
    for condition in frozen_conditions():
        selected = sorted(
            (row for row in rows if row["condition_id"] == condition.condition_id),
            key=lambda row: int(row["sample_seed"]),
        )
        by_condition[condition.condition_id] = np.asarray(
            [int(row["memorized"]) for row in selected], dtype=np.int8
        )
    return by_condition


def summarize(output_dir: Path) -> None:
    """Validate and summarize a completed full run without model inference."""
    rows = read_rows(output_dir / PER_SAMPLE)
    failures = [row for row in rows if row["status"] != "ok"]
    keys = [(row["condition_id"], int(row["sample_seed"])) for row in rows]
    expected = {
        (condition.condition_id, seed)
        for condition in frozen_conditions()
        for seed in SAMPLE_SEEDS
    }
    finite = all(
        np.isfinite(float(row["d1nn"]))
        and np.isfinite(float(row["d2nn"]))
        and np.isfinite(float(row["d1nn_over_d2nn"]))
        for row in rows
        if row["status"] == "ok"
    )
    if (
        len(rows) != EXPECTED_RECORDS
        or set(keys) != expected
        or len(keys) != len(set(keys))
    ):
        raise RuntimeError("E010 record keys are incomplete or duplicated")
    if failures or not finite:
        raise RuntimeError("E010 contains failed or nonfinite records")

    vectors = grouped_vectors(rows)
    summaries = []
    transitions = []
    condition_map = {
        condition.condition_id: condition for condition in frozen_conditions()
    }
    for condition in frozen_conditions():
        values = vectors[condition.condition_id]
        summaries.append(
            {
                "condition_id": condition.condition_id,
                "direction": condition.direction,
                "band": condition.band,
                "role": condition.role,
                "swap_indices": json.dumps(
                    list(condition.swap_indices), separators=(",", ":")
                ),
                "memorized_count": int(values.sum()),
                "sample_count": int(values.size),
                "memorization_rate": f"{values.mean():.17g}",
            }
        )
        if condition.role != "baseline":
            baseline_id = "A0" if condition.direction == "suppression" else "B0"
            counts = Counter(
                transition_category(bool(base), bool(swap))
                for base, swap in zip(vectors[baseline_id], values)
            )
            transitions.append(
                {
                    "condition_id": condition.condition_id,
                    "direction": condition.direction,
                    "band": condition.band,
                    "role": condition.role,
                    **{
                        name: counts[name]
                        for name in (
                            "memorized_to_non_memorized",
                            "non_memorized_to_memorized",
                            "memorized_to_memorized",
                            "non_memorized_to_non_memorized",
                        )
                    },
                }
            )

    analysis = []
    passes = {}
    for direction, prefix in (("suppression", "A"), ("induction", "B")):
        baseline = vectors[f"{prefix}0"]
        for band, identifiers in (("low", ("1", "2", "3")), ("high", ("4", "5", "6"))):
            result = target_control_summary(
                baseline,
                vectors[prefix + identifiers[0]],
                vectors[prefix + identifiers[1]],
                vectors[prefix + identifiers[2]],
                direction=direction,
                seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            )
            passes[(direction, band)] = bool(result["criterion_pass"])
            analysis.append({"direction": direction, "band": band, **result})
    outcome = {
        "experiment_id": EXPERIMENT_ID,
        "formal_outcomes": formal_outcomes(passes),
        "primary_tests": analysis,
        "asymmetric_baselines": True,
        "baseline_matched": False,
        "e008_status": "BLOCKED_UNEXECUTED",
        "causal_scope": "whole-denoiser swaps over frequency-derived intervals",
    }
    write_csv(
        output_dir / "experiment_10_condition_summary.csv",
        summaries,
        list(summaries[0]),
    )
    write_csv(
        output_dir / "experiment_10_paired_transitions.csv",
        transitions,
        list(transitions[0]),
    )
    write_csv(
        output_dir / "experiment_10_directional_analysis.csv",
        analysis,
        list(analysis[0]),
    )
    dump_json(output_dir / "experiment_10_outcome.json", outcome)
    validation = {
        "status": "pass",
        "expected_records": EXPECTED_RECORDS,
        "observed_records": len(rows),
        "unique_keys": len(set(keys)),
        "failure_count": len(failures),
        "finite_nearest_neighbor_records": finite,
        "condition_counts": {key: len(value) for key, value in vectors.items()},
        "seed_range": [SAMPLE_SEEDS[0], SAMPLE_SEEDS[-1]],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "e008_status": "BLOCKED_UNEXECUTED",
    }
    dump_json(output_dir / "experiment_10_validation.json", validation)
    generate_figures(output_dir, summaries, transitions, analysis, rows, condition_map)


def generate_figures(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]],
    analysis: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, str]],
    condition_map: Mapping[str, DirectionalCondition],
) -> None:
    """Generate the seven frozen review figures."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    geometry_rows = [
        row
        for row in read_rows(
            REPO_ROOT / "results/experiment_04b/frequency_restricted_geometry.csv"
        )
        if int(row["cutoff"]) == 4
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, band, target in zip(axes, ("low", "high"), ((8,), (9, 10))):
        selected = [row for row in geometry_rows if row["band"] == band]
        indices = [int(row["sigma_index"]) for row in selected]
        axis.plot(
            indices,
            [float(row["coverage_estimate"]) for row in selected],
            label=f"C_{band}",
        )
        axis.plot(
            indices,
            [float(row["posterior_weight_estimate"]) for row in selected],
            label=f"W_{band}",
        )
        for index in target:
            axis.axvspan(index - 0.45, index + 0.45, color="#e59f3a", alpha=0.22)
        axis.axhline(0.8, color="0.4", linestyle="--", linewidth=1)
        axis.set(
            title=f"{band.title()}-frequency geometry",
            xlabel="Denoiser-call index",
            ylabel="Estimate",
        )
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "geometry_targets.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    for y, condition in enumerate(frozen_conditions()):
        axis.scatter(condition.swap_indices, [y] * len(condition.swap_indices), s=90)
    axis.set(
        yticks=range(14),
        yticklabels=[c.condition_id for c in frozen_conditions()],
        xlabel="Denoiser-call index",
        title="Frozen whole-denoiser condition map",
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "condition_map.png", dpi=180)
    plt.close(fig)

    for direction, filename in (
        ("suppression", "suppression_rates.png"),
        ("induction", "induction_rates.png"),
    ):
        selected = [row for row in summaries if row["direction"] == direction]
        fig, axis = plt.subplots(figsize=(10, 4.8))
        axis.bar(
            [row["condition_id"] for row in selected],
            [float(row["memorization_rate"]) for row in selected],
            color="#2b6f77" if direction == "suppression" else "#b85c38",
        )
        axis.set(
            ylim=(0, 1),
            ylabel="Memorization rate",
            title=f"{direction.title()} direction",
        )
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=180)
        plt.close(fig)

    categories = (
        "memorized_to_non_memorized",
        "non_memorized_to_memorized",
        "memorized_to_memorized",
        "non_memorized_to_non_memorized",
    )
    fig, axis = plt.subplots(figsize=(12, 5.5))
    bottom = np.zeros(len(transitions))
    for category in categories:
        values = np.asarray([int(row[category]) for row in transitions])
        axis.bar(
            [row["condition_id"] for row in transitions],
            values,
            bottom=bottom,
            label=category.replace("_", " "),
        )
        bottom += values
    axis.set(ylabel="Paired seeds", title="Seed-level memorization transitions")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_transitions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    labels = [f"{row['direction']}\n{row['band']}" for row in analysis]
    values = np.asarray([float(row["contrast"]) for row in analysis])
    low = np.asarray([float(row["contrast_ci95_low"]) for row in analysis])
    high = np.asarray([float(row["contrast_ci95_high"]) for row in analysis])
    axis.errorbar(
        labels,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="o",
        capsize=5,
        color="#1f4e5f",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        ylabel="Target minus mean controls",
        title="Frozen directional contrasts (paired 95% bootstrap CI)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "target_control_contrasts.png", dpi=180)
    plt.close(fig)

    samples = np.load(output_dir / "experiment_10_generated_samples.npz")
    reference, reference_indices = load_reference(load_json(DEFAULT_CONFIG))
    baseline_rows = sorted(
        (row for row in rows if row["condition_id"] in ("A0", "B0")),
        key=lambda row: (row["condition_id"], int(row["sample_seed"])),
    )
    selected_rows = baseline_rows[:2] + baseline_rows[256:258]
    fig, axes = plt.subplots(4, 2, figsize=(6.5, 11))
    for row_axes, row in zip(axes, selected_rows):
        condition_id = row["condition_id"]
        position = int(row["sample_seed"]) - SAMPLE_SEEDS[0]
        image = np.transpose(samples[condition_id][position], (1, 2, 0))
        reference_index = int(row["d1nn_reference_index"])
        reference_position = int(
            np.flatnonzero(reference_indices == reference_index)[0]
        )
        nearest = np.transpose(reference[reference_position], (1, 2, 0))
        row_axes[0].imshow(np.clip((image + 1) / 2, 0, 1))
        row_axes[0].set_title(
            f"Generated {condition_id}, seed {row['sample_seed']}", fontsize=9
        )
        row_axes[1].imshow(np.clip((nearest + 1) / 2, 0, 1))
        row_axes[1].set_title(
            f"Nearest reference {reference_index}\n"
            f"ratio={float(row['d1nn_over_d2nn']):.3f}",
            fontsize=9,
        )
        for axis in row_axes:
            axis.axis("off")
    fig.suptitle("Representative samples and nearest references (display clipped)")
    fig.tight_layout()
    fig.savefig(figure_dir / "representative_samples.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    rates = {row["condition_id"]: float(row["memorization_rate"]) for row in summaries}
    for condition_id, rate in rates.items():
        condition = condition_map[condition_id]
        x = min(condition.swap_indices) if condition.swap_indices else -1
        axis.scatter(x, rate, label=condition_id, s=65)
    axis.set(
        xlabel="First swapped call (-1 = no swap)",
        ylabel="Memorization rate",
        ylim=(0, 1),
        title="Executed conditions on the sampler timeline",
    )
    axis.legend(frameon=False, ncol=7, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "timeline_rate_map.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Validate frozen inputs and execute only the requested mode."""
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ and args.mode not in (
        "summarize",
        "plot-only",
    ):
        raise RuntimeError("E010 preflight and inference must run through Slurm")
    config_path = args.config.resolve()
    config, frozen = validate_static_inputs(config_path)
    edm_root = (args.edm_root or Path(config["execution"]["edm_root"])).resolve()
    if args.mode in ("summarize", "plot-only"):
        if args.output_dir is None:
            raise RuntimeError("Summarization requires --output-dir")
        summarize(args.output_dir.resolve())
        return
    commit = validate_execution_commit()
    external = validate_external_inputs(config, frozen, edm_root)
    if args.mode == "preflight":
        print(
            json.dumps(
                {"status": "pass", "commit": commit, **frozen["hashes"], **external},
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        raise RuntimeError("Inference requires --output-dir")
    execute_run(
        args.mode,
        args.output_dir.resolve(),
        config_path,
        config,
        frozen,
        external,
        commit,
        edm_root,
        args.resume,
    )


if __name__ == "__main__":
    main()
