#!/usr/bin/env python3
"""Pre-staged baseline-only evaluator for the frozen E009 Stage A pool."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from _bootstrap import REPO_ROOT
import spectral_diffusion_playground.e008_checkpoint_preflight as core

CONFIG_PATH = REPO_ROOT / "configs/e009_stage_a_evaluation.json"
OUTPUT_DIR = Path("/home/xggh8/data/zw-lab/e009_stage_a_baseline")
EXPERIMENT_ID = "experiment_09_stage_a_baseline"
PILOT_SEEDS = tuple(range(20_000, 20_128))
CONFIRMATORY_SEEDS = tuple(range(256))
ROLES = ("edm_2k", "edm_5k", "edm_10k")
TRAINING_COMMIT = "d19c470bc4b547e2bad5488b30892be2814c7b12"
SMOKE_SEEDS = PILOT_SEEDS[:2]
SMOKE_KIMG = 12_000
IDENTITY_FILENAME = "checkpoint_pool_identity.json"
SMOKE_VALIDATION_FILENAME = "smoke_validation.json"
RATE_FIGURE_FILENAME = "baseline_memorization_rate_by_kimg.png"
TRAINING_CONFIG_BY_ROLE = {
    "edm_2k": "configs/e009_edm2k_12000kimg.yaml",
    "edm_5k": "configs/e009_edm5k_12000kimg.yaml",
    "edm_10k": "configs/e009_edm10k_12000kimg.yaml",
}
PERSISTENT_DATA_ROOT = Path("/home/xggh8/data/zw-lab")


def is_under_persistent_root(
    candidate: Path,
    persistent_root: Path = PERSISTENT_DATA_ROOT,
) -> bool:
    """Return whether a candidate resolves beneath the approved data root."""
    return candidate.resolve().is_relative_to(persistent_root.resolve())


def load_e008_entrypoint():
    """Load the audited E008 implementation as the numerical execution engine."""
    path = REPO_ROOT / "experiments/08_checkpoint_baseline_preflight.py"
    spec = importlib.util.spec_from_file_location("e009_e008_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the E008 baseline engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and strictly validate the frozen E009 evaluation contract."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("E009 evaluation experiment ID changed")
    if config["pilot_seeds"] != {
        "start": 20000,
        "stop_inclusive": 20127,
        "count": 128,
    }:
        raise ValueError("E009 pilot seeds changed")
    if config["expected_training_kimg"] != list(range(0, 12001, 1000)):
        raise ValueError("E009 checkpoint cadence changed")
    if config["eligibility"]["count_interval_inclusive"] != [13, 115]:
        raise ValueError("E009 eligibility changed")
    if set(config["candidate_roots"]) != {"edm_2k", "edm_5k", "edm_10k"}:
        raise ValueError("E009 candidate roles changed")
    if set(PILOT_SEEDS).intersection(CONFIRMATORY_SEEDS):
        raise ValueError("Pilot and confirmatory seeds overlap")
    return config


def configure_engine(engine, config: Mapping[str, Any]) -> None:
    """Bind E009 constants without altering the frozen E008 source or artifacts."""
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.PILOT_SEEDS = PILOT_SEEDS
    engine.CONFIRMATORY_SEEDS = CONFIRMATORY_SEEDS
    engine.DEFAULT_CONFIG = CONFIG_PATH
    engine.DEFAULT_OUTPUT = OUTPUT_DIR
    core.PILOT_SEEDS = PILOT_SEEDS


def build_args(config: Mapping[str, Any], output_dir: Path) -> argparse.Namespace:
    """Build the bounded argument object consumed by the audited engine."""
    return argparse.Namespace(
        config=CONFIG_PATH,
        output_dir=output_dir,
        dataset_root=Path(config["dataset"]["archive"]),
        reference_subset=REPO_ROOT / config["memorization"]["reference_subset"],
        edm_root=Path(config["execution"]["edm_root"]),
        device="auto",
        batch_size=int(config["execution"]["batch_size"]),
        nn_batch_size=int(config["execution"]["nn_batch_size"]),
        smoke=False,
        resume=True,
    )


def freeze_inventory(engine, config: Mapping[str, Any], output_dir: Path) -> None:
    """Hash, load, and freeze exactly 39 completed Stage A checkpoints."""
    if output_dir.exists():
        raise RuntimeError(f"E009 evaluation output already exists: {output_dir}")
    args = build_args(config, output_dir)
    provenance = engine.validate_static_provenance(args, config)
    protocol = json.loads(
        (REPO_ROOT / "configs/e009_stage_a_protocol.json").read_text(encoding="utf-8")
    )
    staging_dir = output_dir.with_name(
        f"{output_dir.name}.inventory_{os.environ['SLURM_JOB_ID']}"
    )
    if staging_dir.exists():
        raise RuntimeError(f"E009 inventory staging directory exists: {staging_dir}")
    rows: list[dict[str, object]] = []
    expected_kimg = config["expected_training_kimg"]
    for role, root_string in config["candidate_roots"].items():
        root = Path(root_string).resolve()
        if not is_under_persistent_root(root):
            raise RuntimeError(f"Candidate root is not persistent storage: {root}")
        accepted, malformed = core.discover_checkpoint_paths(root)
        observed_kimg = [core.parse_training_kimg(path.name) for path in accepted]
        if observed_kimg != expected_kimg or malformed:
            raise RuntimeError(
                f"Incomplete checkpoint inventory for {role}: "
                f"observed={observed_kimg}, malformed={malformed}"
            )
        config_source = engine.find_training_config(root)
        config_hash = core.sha256_file(config_source) if config_source else ""
        expected_config_hash = protocol["artifact_hashes"]["config_sha256"][
            TRAINING_CONFIG_BY_ROLE[role]
        ]
        if config_hash != expected_config_hash:
            raise RuntimeError(
                f"Frozen training config mismatch for {role}: "
                f"{config_hash} != {expected_config_hash}"
            )
        for path in accepted:
            row = engine.base_inventory_record(
                role,
                path,
                int(config["candidate_subset_sizes"][role]),
                config_source,
                config_hash,
            )
            try:
                row.update(engine.inspect_checkpoint(path, args.edm_root))
                row["inventory_status"] = "accepted"
                row["rejection_reason"] = ""
            except Exception as error:
                row.update(
                    {
                        "checkpoint_sha256": core.sha256_file(path),
                        "architecture_identity": "",
                        "ema_status": "unknown",
                        "label_conditioning_status": "unknown",
                        "loadability_status": "failed",
                        "inventory_status": "rejected",
                        "rejection_reason": f"{type(error).__name__}: {error}",
                    }
                )
            rows.append(row)
    rows.sort(key=lambda row: (str(row["model_role"]), int(row["training_kimg"])))
    staging_dir.mkdir(parents=True)
    inventory_path = staging_dir / engine.INVENTORY_FILENAME
    engine.write_csv(inventory_path, rows, engine.inventory_header())
    accepted = [row for row in rows if row["inventory_status"] == "accepted"]
    rejected = [row for row in rows if row["inventory_status"] == "rejected"]
    architecture_identities = {str(row["architecture_identity"]) for row in accepted}
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "pool_frozen_before_pilot": True,
        "pilot_started": False,
        "inventory_job_id": os.environ["SLURM_JOB_ID"],
        "training_commit": TRAINING_COMMIT,
        "scientific_scope": config["scientific_scope"],
        "config_sha256": core.sha256_file(CONFIG_PATH),
        "repository_commit": engine.git_commit(),
        "edm_source_commit": engine.repository_commit(args.edm_root),
        "provenance": provenance,
        "expected_checkpoint_count": 39,
        "expected_record_count": 4_992,
        "pilot_seeds": list(PILOT_SEEDS),
        "future_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "excluded_seed_ranges": [[0, 255], [10_000, 10_127]],
        "eligibility": config["eligibility"],
        "memorization": config["memorization"],
        "sampler": config["sampler"],
        "pair_selection": config["pair_selection"],
        "nearest_neighbor_evaluator": {
            "device": "cpu",
            "dtype": "float64",
            "distance": "direct_squared_differences_then_euclidean_sqrt",
            "tie_break": "reference_subset_position",
        },
        "no_swap_boundary": {
            "baseline_only": True,
            "donor_models_allowed": False,
            "swap_windows_allowed": False,
        },
        "inventory": {
            "path": str(output_dir / engine.INVENTORY_FILENAME),
            "sha256": core.sha256_file(inventory_path),
            "row_count": len(rows),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "records": rows,
        },
    }
    manifest_path = staging_dir / engine.POOL_MANIFEST_FILENAME
    engine.dump_json(manifest_path, manifest)
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "inventory_sha256": core.sha256_file(inventory_path),
        "manifest_sha256": core.sha256_file(manifest_path),
        "inventory_job_id": os.environ["SLURM_JOB_ID"],
        "repository_commit": engine.git_commit(),
        "training_commit": TRAINING_COMMIT,
    }
    engine.dump_json(staging_dir / IDENTITY_FILENAME, identity)
    inventory_valid = (
        len(rows) == 39
        and len(accepted) == 39
        and not rejected
        and len(architecture_identities) == 1
    )
    if inventory_valid:
        freeze_role_shards(engine, manifest, rows, staging_dir, output_dir)
    staging_dir.replace(output_dir)
    if not inventory_valid:
        raise RuntimeError(
            "E009 inventory failed acceptance: "
            f"rows={len(rows)}, accepted={len(accepted)}, "
            f"rejected={len(rejected)}, architectures={len(architecture_identities)}"
        )
    print(json.dumps({"status": "pool_frozen", **identity}, indent=2))


def freeze_role_shards(
    engine,
    master_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
    staging_dir: Path,
    final_output_dir: Path,
) -> None:
    """Create three immutable role-specific pools for isolated array writes."""
    for role in ROLES:
        shard_dir = staging_dir / "shards" / role
        shard_dir.mkdir(parents=True)
        shard_rows = [dict(row) for row in rows if row["model_role"] == role]
        if len(shard_rows) != 13:
            raise RuntimeError(f"Expected 13 frozen checkpoints for {role}")
        inventory_path = shard_dir / engine.INVENTORY_FILENAME
        engine.write_csv(inventory_path, shard_rows, engine.inventory_header())
        shard_manifest = dict(master_manifest)
        shard_manifest["parent_inventory_sha256"] = master_manifest["inventory"][
            "sha256"
        ]
        shard_manifest["inventory"] = {
            "path": str(final_output_dir / "shards" / role / engine.INVENTORY_FILENAME),
            "sha256": core.sha256_file(inventory_path),
            "row_count": len(shard_rows),
            "accepted_count": len(shard_rows),
            "rejected_count": 0,
            "records": shard_rows,
        }
        engine.dump_json(shard_dir / engine.POOL_MANIFEST_FILENAME, shard_manifest)


def verify_pool_identity(engine, output_dir: Path) -> dict[str, Any]:
    """Verify the non-self-referential inventory and manifest hash sidecar."""
    identity = json.loads((output_dir / IDENTITY_FILENAME).read_text(encoding="utf-8"))
    expected = {
        "inventory_sha256": core.sha256_file(output_dir / engine.INVENTORY_FILENAME),
        "manifest_sha256": core.sha256_file(output_dir / engine.POOL_MANIFEST_FILENAME),
    }
    observed = {key: identity[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"E009 pool identity mismatch: {observed} != {expected}")
    return identity


def prepare_smoke_pool(
    engine,
    source_dir: Path,
    destination: Path,
) -> None:
    """Freeze one final checkpoint per role into an isolated smoke pool."""
    if destination.exists():
        raise RuntimeError(f"E009 smoke output exists: {destination}")
    source_manifest = json.loads(
        (source_dir / engine.POOL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    source_rows = engine.read_csv_rows(source_dir / engine.INVENTORY_FILENAME)
    selected = [
        next(
            row
            for row in source_rows
            if row["model_role"] == role
            and int(row["training_kimg"]) == SMOKE_KIMG
            and row["inventory_status"] == "accepted"
        )
        for role in ROLES
    ]
    destination.mkdir(parents=True)
    inventory_path = destination / engine.INVENTORY_FILENAME
    engine.write_csv(inventory_path, selected, engine.inventory_header())
    manifest = dict(source_manifest)
    manifest["mode"] = "smoke"
    manifest["pilot_seeds"] = list(SMOKE_SEEDS)
    manifest["inventory"] = {
        "path": str(inventory_path),
        "sha256": core.sha256_file(inventory_path),
        "row_count": len(selected),
        "accepted_count": len(selected),
        "rejected_count": 0,
        "records": selected,
    }
    engine.dump_json(destination / engine.POOL_MANIFEST_FILENAME, manifest)


def run_smoke(engine, config: Mapping[str, Any], output_dir: Path) -> None:
    """Run two exact baseline-only regenerations over three final checkpoints."""
    source_dir = OUTPUT_DIR.resolve()
    verify_pool_identity(engine, source_dir)
    if output_dir.exists():
        raise RuntimeError(f"E009 smoke root exists: {output_dir}")
    output_dir.mkdir(parents=True)
    engine.PILOT_SEEDS = SMOKE_SEEDS
    core.PILOT_SEEDS = SMOKE_SEEDS
    engine.summarize = lambda *_: None
    run_dirs = [output_dir / "run_a", output_dir / "run_b"]
    for run_dir in run_dirs:
        prepare_smoke_pool(engine, source_dir, run_dir)
        engine.run_pilot(build_args(config, run_dir), config)
        rewrite_e009_run_manifest(
            engine,
            run_dir,
            verify_pool_identity(engine, source_dir),
            mode="smoke",
        )
    first = engine.read_csv_rows(run_dirs[0] / engine.PER_SAMPLE_FILENAME)
    second = engine.read_csv_rows(run_dirs[1] / engine.PER_SAMPLE_FILENAME)
    expected_keys = {(role, seed) for role in ROLES for seed in SMOKE_SEEDS}
    observed_keys = {(str(row["model_role"]), int(row["sample_seed"])) for row in first}
    finite = all(
        row["status"] == "ok"
        and all(
            math.isfinite(float(row[field]))
            for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
        )
        and bool(row["generated_sample_hash"])
        for row in first
    )
    manifests = [
        json.loads((run_dir / "pilot_run_manifest.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    checks = {
        "row_count_is_6": len(first) == 6 and len(second) == 6,
        "checkpoint_seed_keys_exact": observed_keys == expected_keys,
        "rerun_rows_exact": first == second,
        "successful_values_finite": finite,
        "cpu_float64_nearest_neighbor": all(
            manifest["nearest_neighbor_evaluator"]["device"] == "cpu"
            and manifest["nearest_neighbor_evaluator"]["dtype"] == "float64"
            for manifest in manifests
        ),
        "swap_conditions_absent": all(
            field not in first[0]
            for field in ("donor_model", "swap_window", "window_name")
        ),
        "confirmatory_seeds_untouched": not set(SMOKE_SEEDS).intersection(
            CONFIRMATORY_SEEDS
        ),
        "e008_unexecuted": not config["scientific_scope"]["e008_executed"],
    }
    validation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "repository_commit": engine.git_commit(),
        "source_pool_identity": verify_pool_identity(engine, source_dir),
        "checkpoint_kimg": SMOKE_KIMG,
        "seeds": list(SMOKE_SEEDS),
        "checks": checks,
    }
    engine.dump_json(output_dir / SMOKE_VALIDATION_FILENAME, validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"E009 smoke validation failed: {validation}")
    print(json.dumps(validation, indent=2, sort_keys=True))


def rewrite_e009_run_manifest(
    engine,
    run_dir: Path,
    pool_identity: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    """Replace inherited E008 diagnostic history with E009 execution provenance."""
    path = run_dir / "pilot_run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("pre_execution_and_diagnostic_jobs", None)
    manifest["mode"] = mode
    manifest["numerical_engine"] = "audited E008 baseline-only evaluator"
    manifest["training_commit"] = TRAINING_COMMIT
    manifest["master_pool_identity"] = dict(pool_identity)
    engine.dump_json(path, manifest)


def read_bool(value: object) -> bool:
    """Parse one CSV boolean deterministically."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def select_pair(
    new_rows: Sequence[Mapping[str, object]],
    small_rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Apply the frozen rate, dataset-size, then SHA ordering."""
    new_eligible = [
        row
        for row in new_rows
        if read_bool(row["eligible"])
        and int(str(row["model_role"]).split("_")[1][:-1]) >= 5
    ]
    small_eligible = [row for row in small_rows if read_bool(row["eligible"])]
    ranked = []
    for new in new_eligible:
        size = int(str(new["model_role"]).split("_")[1][:-1]) * 1000
        for small in small_eligible:
            difference = abs(
                float(new["memorization_rate"]) - float(small["memorization_rate"])
            )
            ranked.append(
                (
                    difference,
                    -size,
                    str(new["checkpoint_sha256"]),
                    str(small["checkpoint_sha256"]),
                    new,
                    small,
                )
            )
    if not ranked:
        return None
    difference, _, _, _, new, small = min(ranked, key=lambda item: item[:4])
    return {
        "absolute_pilot_rate_difference": difference,
        "larger_data_checkpoint": dict(new),
        "edm_1k_checkpoint": dict(small),
    }


def summarize(engine, config: Mapping[str, Any], output_dir: Path) -> None:
    """Aggregate all candidates and apply the frozen Stage A decision tree."""
    pool_identity = verify_pool_identity(engine, output_dir)
    merge_role_shards(engine, output_dir)
    inventory = engine.read_csv_rows(output_dir / engine.INVENTORY_FILENAME)
    samples = engine.read_csv_rows(output_dir / engine.PER_SAMPLE_FILENAME)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in samples:
        grouped.setdefault(str(row["checkpoint_sha256"]), []).append(row)
    summaries = [
        core.summarize_candidate(
            model_role=str(candidate["model_role"]),
            checkpoint_path=str(candidate["checkpoint_path"]),
            checkpoint_sha256=str(candidate["checkpoint_sha256"]),
            training_kimg=int(candidate["training_kimg"]),
            rows=grouped.get(str(candidate["checkpoint_sha256"]), []),
        )
        for candidate in inventory
    ]
    summaries.sort(key=lambda row: (str(row["model_role"]), int(row["training_kimg"])))
    engine.write_csv(
        output_dir / engine.SUMMARY_FILENAME, summaries, engine.summary_header()
    )
    small_path = REPO_ROOT / config["pair_selection"]["edm_1k_summary"]
    if (
        core.sha256_file(small_path)
        != config["pair_selection"]["edm_1k_summary_sha256"]
    ):
        raise RuntimeError("Frozen E008 EDM-1K summary hash mismatch")
    with small_path.open(newline="", encoding="utf-8") as handle:
        small_rows = list(csv.DictReader(handle))
    pair = select_pair(summaries, small_rows)
    eligible_2k = any(
        row["model_role"] == "edm_2k" and bool(row["eligible"]) for row in summaries
    )
    if pair is not None:
        outcome = "ELIGIBLE_LARGER_DATA_PAIR_FROZEN"
    elif eligible_2k:
        outcome = "PROVISIONAL_2K_ONLY_STAGE_B_REQUIRED"
    else:
        outcome = "BLOCKED_NO_ELIGIBLE_STAGE_A_CHECKPOINT"
    pilot_manifests = {
        role: json.loads(
            (output_dir / "shards" / role / "pilot_run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for role in ROLES
    }
    pair_payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outcome": outcome,
        "selected_pair": pair,
        "selection_rule": config["pair_selection"],
        "pool_identity": pool_identity,
        "evaluation_seeds": list(PILOT_SEEDS),
        "reserved_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "repository_commit": engine.git_commit(),
        "summarization_job_id": os.environ["SLURM_JOB_ID"],
        "pilot_jobs": {
            role: manifest["slurm_job_id"] for role, manifest in pilot_manifests.items()
        },
        "e008_executed": False,
    }
    engine.dump_json(
        output_dir / engine.PAIR_FILENAME,
        pair_payload,
    )
    engine.dump_json(
        output_dir / engine.OUTCOME_FILENAME,
        {
            "experiment_id": EXPERIMENT_ID,
            "outcome": outcome,
            "eligible_counts": {
                role: sum(
                    row["model_role"] == role and bool(row["eligible"])
                    for row in summaries
                )
                for role in config["candidate_roots"]
            },
            "e008_executed": False,
            "swap_conditions_generated": False,
            "stage_b_started": False,
        },
    )
    generate_rate_figure(engine, summaries, output_dir)
    accepted = [row for row in inventory if row["inventory_status"] == "accepted"]
    observed_keys = {
        (str(row["checkpoint_sha256"]), int(row["sample_seed"])) for row in samples
    }
    expected_keys = {
        (str(candidate["checkpoint_sha256"]), seed)
        for candidate in accepted
        for seed in PILOT_SEEDS
    }
    checks = {
        "accepted_checkpoint_count_is_39": len(accepted) == 39,
        "per_sample_row_count_is_4992": len(samples) == 4992,
        "checkpoint_seed_keys_exact": observed_keys == expected_keys,
        "checkpoint_seed_keys_unique": len(observed_keys) == len(samples),
        "pilot_seed_set_exact": {int(row["sample_seed"]) for row in samples}
        == set(PILOT_SEEDS),
        "confirmatory_seed_overlap_absent": not {
            int(row["sample_seed"]) for row in samples
        }.intersection(CONFIRMATORY_SEEDS),
        "all_records_explicit": all(
            row["status"] in {"ok", "failed"}
            and (row["status"] == "ok" or bool(row["error"]))
            for row in samples
        ),
        "successful_distances_finite": all(
            all(
                math.isfinite(float(row[field]))
                for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
            )
            for row in samples
            if row["status"] == "ok"
        ),
        "swap_fields_absent": all(
            field not in samples[0] if samples else False
            for field in ("donor_model", "swap_window", "window_name")
        ),
        "e008_unexecuted": not config["scientific_scope"]["e008_executed"],
    }
    validation = {
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "accepted_checkpoint_count": len(accepted),
        "per_sample_row_count": len(samples),
        "failed_row_count": sum(row["status"] != "ok" for row in samples),
        "pool_identity": pool_identity,
        "outcome": outcome,
    }
    engine.dump_json(output_dir / engine.VALIDATION_FILENAME, validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"E009 baseline validation failed: {validation}")


def generate_rate_figure(
    engine,
    summaries: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Plot separate baseline-rate trajectories with the frozen eligibility band."""
    figure, axes = engine.plt.subplots(1, 3, figsize=(14.0, 4.3), sharey=True)
    labels = {"edm_2k": "2K", "edm_5k": "5K", "edm_10k": "10K"}
    for axis, role in zip(axes, ROLES):
        role_rows = [row for row in summaries if row["model_role"] == role]
        x_values = [int(row["training_kimg"]) for row in role_rows]
        y_values = [float(row["memorization_rate"]) for row in role_rows]
        axis.axhspan(0.1, 0.9, color="#d9ead3", alpha=0.65, label="Eligible")
        axis.plot(x_values, y_values, color="#1f5d85", marker="o", linewidth=2)
        axis.set_title(f"{labels[role]} baseline")
        axis.set_xlabel("Training (kimg)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Memorization rate")
    axes[0].legend(loc="best")
    figure.suptitle("E009 Stage A Baseline Memorization Trajectories", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / RATE_FIGURE_FILENAME, dpi=180, bbox_inches="tight")
    engine.plt.close(figure)
    for role in ROLES:
        role_rows = [row for row in summaries if row["model_role"] == role]
        role_figure, axis = engine.plt.subplots(figsize=(6.2, 4.5))
        axis.axhspan(0.1, 0.9, color="#d9ead3", alpha=0.65, label="Eligible")
        axis.plot(
            [int(row["training_kimg"]) for row in role_rows],
            [float(row["memorization_rate"]) for row in role_rows],
            color="#1f5d85",
            marker="o",
            linewidth=2,
        )
        axis.set(
            title=f"{labels[role]} Baseline Memorization",
            xlabel="Training (kimg)",
            ylabel="Memorization rate",
            ylim=(-0.03, 1.03),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        role_figure.tight_layout()
        role_figure.savefig(
            output_dir / f"baseline_memorization_rate_{role}.png",
            dpi=180,
            bbox_inches="tight",
        )
        engine.plt.close(role_figure)


def merge_role_shards(engine, output_dir: Path) -> None:
    """Merge three complete disjoint role shards into stable master CSV files."""
    merged: list[dict[str, object]] = []
    for role in ROLES:
        shard_dir = output_dir / "shards" / role
        rows = [
            engine.normalize_resume_row(row)
            for row in engine.read_csv_rows(shard_dir / engine.PER_SAMPLE_FILENAME)
        ]
        if len(rows) != 13 * len(PILOT_SEEDS):
            raise RuntimeError(f"Incomplete E009 pilot shard for {role}: {len(rows)}")
        if any(row["model_role"] != role for row in rows):
            raise RuntimeError(f"Cross-role rows found in {role} shard")
        merged.extend(rows)
    merged = core.merge_resume_rows([], merged)
    engine.write_csv(
        output_dir / engine.PER_SAMPLE_FILENAME, merged, engine.per_sample_header()
    )
    failures = [row for row in merged if row["status"] != "ok"]
    engine.write_csv(
        output_dir / engine.FAILURES_FILENAME, failures, engine.per_sample_header()
    )


def parse_args() -> argparse.Namespace:
    """Parse a pre-staged mode; no mode is run automatically."""
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inventory-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run-pilot", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Freeze inventory, execute baseline rows, or summarize a completed pilot."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing to run E009 evaluation outside Slurm")
    args = parse_args()
    config = load_config()
    engine = load_e008_entrypoint()
    configure_engine(engine, config)
    if args.inventory_only:
        freeze_inventory(engine, config, args.output_dir.resolve())
    elif args.smoke:
        if args.role is not None:
            raise ValueError("--role is not valid with --smoke")
        run_smoke(engine, config, args.output_dir.resolve())
    elif args.run_pilot:
        if args.role is None:
            raise ValueError("--run-pilot requires one frozen --role")
        pool_identity = verify_pool_identity(engine, args.output_dir.resolve())
        engine.summarize = lambda *_: None
        shard_dir = args.output_dir.resolve() / "shards" / args.role
        engine.run_pilot(build_args(config, shard_dir), config)
        rewrite_e009_run_manifest(
            engine,
            shard_dir,
            pool_identity,
            mode="full_pilot",
        )
    else:
        if args.role is not None:
            raise ValueError("--role is valid only with --run-pilot")
        summarize(engine, config, args.output_dir.resolve())


if __name__ == "__main__":
    main()
