#!/usr/bin/env python3
"""Evaluate the frozen E009 Stage B 5K and EDM-1K baseline cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from _bootstrap import REPO_ROOT
import spectral_diffusion_playground.e008_checkpoint_preflight as core

CONFIG_PATH = REPO_ROOT / "configs/e009_stage_b_evaluation.json"
OUTPUT_DIR = Path("/home/xggh8/data/zw-lab/e009_stage_b_baseline")
EXPERIMENT_ID = "experiment_09_stage_b_baseline"
PILOT_SEEDS = tuple(range(20_000, 20_128))
CONFIRMATORY_SEEDS = tuple(range(256))
ROLES = ("edm_5k", "edm_1k")
ROLE_COUNTS = {"edm_5k": 18, "edm_1k": 6}
SUBSET_SIZES = {"edm_5k": 5000, "edm_1k": 1000}
SMOKE_KIMG = {"edm_5k": 30000, "edm_1k": 12000}
SMOKE_SEEDS = PILOT_SEEDS[:2]
IDENTITY_FILENAME = "checkpoint_pool_identity.json"
SMOKE_VALIDATION_FILENAME = "smoke_validation.json"
RATE_FIGURE_FILENAME = "stage_b_baseline_memorization_rates.png"


def load_engine():
    """Load the audited E008 baseline implementation as the numerical engine."""
    path = REPO_ROOT / "experiments/08_checkpoint_baseline_preflight.py"
    spec = importlib.util.spec_from_file_location("e009_stage_b_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the E008 baseline engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and strictly validate the frozen Stage B evaluation contract."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Stage B evaluation experiment ID changed")
    if config["pilot_seeds"] != {
        "start": 20000,
        "stop_inclusive": 20127,
        "count": 128,
    }:
        raise ValueError("Stage B evaluation seeds changed")
    if config["eligibility"]["count_interval_inclusive"] != [13, 115]:
        raise ValueError("Stage B eligibility changed")
    if {role: len(rows) for role, rows in config["candidates"].items()} != ROLE_COUNTS:
        raise ValueError("Stage B candidate counts changed")
    if set(PILOT_SEEDS).intersection(CONFIRMATORY_SEEDS):
        raise ValueError("Pilot and confirmatory seeds overlap")
    if config["execution"]["expected_record_count"] != 3072:
        raise ValueError("Stage B expected record count changed")
    return config


def configure_engine(engine) -> None:
    """Bind Stage B constants without changing the audited numerical engine."""
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.PILOT_SEEDS = PILOT_SEEDS
    engine.CONFIRMATORY_SEEDS = CONFIRMATORY_SEEDS
    engine.DEFAULT_CONFIG = CONFIG_PATH
    engine.DEFAULT_OUTPUT = OUTPUT_DIR
    core.PILOT_SEEDS = PILOT_SEEDS


def require_execution_identity(engine, config: Mapping[str, Any]) -> dict[str, str]:
    """Require the executed commit to equal the published remote branch hash."""
    commit = engine.git_commit()
    remote_commit = os.environ.get("E009_REMOTE_BRANCH_COMMIT", "")
    if remote_commit != commit:
        raise RuntimeError(
            f"Published branch hash mismatch: {remote_commit!r} != {commit!r}"
        )
    return {
        "remote_branch": str(config["remote_branch"]),
        "remote_branch_commit": remote_commit,
        "repository_commit": commit,
    }


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


def candidate_rows(config: Mapping[str, Any]) -> list[dict[str, object]]:
    """Return the explicit 24-checkpoint cohort in deterministic order."""
    rows = []
    for role in ROLES:
        for candidate in config["candidates"][role]:
            rows.append({"model_role": role, **candidate})
    return rows


def freeze_inventory(engine, config: Mapping[str, Any], output_dir: Path) -> None:
    """Hash, CPU-load, and freeze exactly 24 preregistered checkpoints."""
    if output_dir.exists():
        raise RuntimeError(f"Stage B evaluation output exists: {output_dir}")
    args = build_args(config, output_dir)
    provenance = engine.validate_static_provenance(args, config)
    execution = require_execution_identity(engine, config)
    staging = output_dir.with_name(
        f"{output_dir.name}.inventory_{os.environ['SLURM_JOB_ID']}"
    )
    if staging.exists():
        raise RuntimeError(f"Stage B inventory staging exists: {staging}")
    rows = []
    for candidate in candidate_rows(config):
        path = Path(str(candidate["checkpoint_path"]))
        if not path.is_file():
            raise RuntimeError(f"Missing frozen checkpoint: {path}")
        if core.sha256_file(path) != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint hash mismatch: {path}")
        record = engine.base_inventory_record(
            str(candidate["model_role"]),
            path,
            SUBSET_SIZES[str(candidate["model_role"])],
            None,
            "",
        )
        record.update(engine.inspect_checkpoint(path, args.edm_root))
        record["inventory_status"] = "accepted"
        record["rejection_reason"] = ""
        rows.append(record)
    rows.sort(key=lambda row: (str(row["model_role"]), int(row["training_kimg"])))
    if len({str(row["architecture_identity"]) for row in rows}) != 1:
        raise RuntimeError("Stage B candidate architectures differ")
    staging.mkdir(parents=True)
    inventory_path = staging / engine.INVENTORY_FILENAME
    engine.write_csv(inventory_path, rows, engine.inventory_header())
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "pool_frozen_before_pilot": True,
        "pilot_started": False,
        "inventory_job_id": os.environ["SLURM_JOB_ID"],
        "scientific_scope": config["scientific_scope"],
        "config_sha256": core.sha256_file(CONFIG_PATH),
        **execution,
        "edm_source_commit": engine.repository_commit(args.edm_root),
        "provenance": provenance,
        "expected_checkpoint_count": 24,
        "expected_record_count": 3072,
        "pilot_seeds": list(PILOT_SEEDS),
        "future_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
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
        "inventory": {
            "path": str(output_dir / engine.INVENTORY_FILENAME),
            "sha256": core.sha256_file(inventory_path),
            "row_count": len(rows),
            "accepted_count": len(rows),
            "rejected_count": 0,
            "records": rows,
        },
    }
    manifest_path = staging / engine.POOL_MANIFEST_FILENAME
    engine.dump_json(manifest_path, manifest)
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "inventory_sha256": core.sha256_file(inventory_path),
        "manifest_sha256": core.sha256_file(manifest_path),
        "inventory_job_id": os.environ["SLURM_JOB_ID"],
        **execution,
    }
    engine.dump_json(staging / IDENTITY_FILENAME, identity)
    freeze_role_shards(engine, manifest, rows, staging, output_dir)
    staging.replace(output_dir)
    print(json.dumps({"status": "pool_frozen", **identity}, indent=2))


def freeze_role_shards(engine, manifest, rows, staging: Path, output_dir: Path) -> None:
    """Create two immutable role-specific pools for isolated writes."""
    for role in ROLES:
        shard = staging / "shards" / role
        shard.mkdir(parents=True)
        selected = [dict(row) for row in rows if row["model_role"] == role]
        if len(selected) != ROLE_COUNTS[role]:
            raise RuntimeError(f"Wrong frozen checkpoint count for {role}")
        inventory_path = shard / engine.INVENTORY_FILENAME
        engine.write_csv(inventory_path, selected, engine.inventory_header())
        shard_manifest = dict(manifest)
        shard_manifest["parent_inventory_sha256"] = manifest["inventory"]["sha256"]
        shard_manifest["inventory"] = {
            "path": str(output_dir / "shards" / role / engine.INVENTORY_FILENAME),
            "sha256": core.sha256_file(inventory_path),
            "row_count": len(selected),
            "accepted_count": len(selected),
            "rejected_count": 0,
            "records": selected,
        }
        engine.dump_json(shard / engine.POOL_MANIFEST_FILENAME, shard_manifest)


def verify_pool_identity(engine, output_dir: Path) -> dict[str, Any]:
    """Verify the immutable master inventory and manifest sidecar."""
    identity = json.loads((output_dir / IDENTITY_FILENAME).read_text())
    observed = {
        "inventory_sha256": core.sha256_file(output_dir / engine.INVENTORY_FILENAME),
        "manifest_sha256": core.sha256_file(output_dir / engine.POOL_MANIFEST_FILENAME),
    }
    if {key: identity[key] for key in observed} != observed:
        raise RuntimeError("Stage B pool identity changed")
    return identity


def prepare_smoke_pool(engine, source: Path, destination: Path) -> None:
    """Freeze one checkpoint per role into one isolated smoke pool."""
    if destination.exists():
        raise RuntimeError(f"Stage B smoke output exists: {destination}")
    manifest = json.loads((source / engine.POOL_MANIFEST_FILENAME).read_text())
    rows = engine.read_csv_rows(source / engine.INVENTORY_FILENAME)
    selected = [
        next(
            row
            for row in rows
            if row["model_role"] == role
            and int(row["training_kimg"]) == SMOKE_KIMG[role]
        )
        for role in ROLES
    ]
    destination.mkdir(parents=True)
    inventory_path = destination / engine.INVENTORY_FILENAME
    engine.write_csv(inventory_path, selected, engine.inventory_header())
    smoke_manifest = dict(manifest)
    smoke_manifest["mode"] = "smoke"
    smoke_manifest["pilot_seeds"] = list(SMOKE_SEEDS)
    smoke_manifest["inventory"] = {
        "path": str(inventory_path),
        "sha256": core.sha256_file(inventory_path),
        "row_count": 2,
        "accepted_count": 2,
        "rejected_count": 0,
        "records": selected,
    }
    engine.dump_json(destination / engine.POOL_MANIFEST_FILENAME, smoke_manifest)


def rewrite_run_manifest(engine, run_dir: Path, identity, mode: str) -> None:
    """Replace inherited diagnostic history with Stage B provenance."""
    path = run_dir / "pilot_run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("pre_execution_and_diagnostic_jobs", None)
    manifest["mode"] = mode
    manifest["numerical_engine"] = "audited E008 baseline-only evaluator"
    manifest["master_pool_identity"] = identity
    manifest["remote_branch"] = identity["remote_branch"]
    manifest["remote_branch_commit"] = identity["remote_branch_commit"]
    engine.dump_json(path, manifest)


def run_smoke(engine, config: Mapping[str, Any], output_dir: Path) -> None:
    """Run two exact regenerations over one checkpoint per role and two seeds."""
    source = OUTPUT_DIR.resolve()
    identity = verify_pool_identity(engine, source)
    require_execution_identity(engine, config)
    if output_dir.exists():
        raise RuntimeError(f"Stage B smoke root exists: {output_dir}")
    output_dir.mkdir(parents=True)
    engine.PILOT_SEEDS = SMOKE_SEEDS
    core.PILOT_SEEDS = SMOKE_SEEDS
    engine.summarize = lambda *_: None
    run_dirs = [output_dir / "run_a", output_dir / "run_b"]
    for run_dir in run_dirs:
        prepare_smoke_pool(engine, source, run_dir)
        engine.run_pilot(build_args(config, run_dir), config)
        rewrite_run_manifest(engine, run_dir, identity, "smoke")
    first = engine.read_csv_rows(run_dirs[0] / engine.PER_SAMPLE_FILENAME)
    second = engine.read_csv_rows(run_dirs[1] / engine.PER_SAMPLE_FILENAME)
    expected = {(role, seed) for role in ROLES for seed in SMOKE_SEEDS}
    observed = {(str(row["model_role"]), int(row["sample_seed"])) for row in first}
    checks = {
        "row_count_is_4": len(first) == len(second) == 4,
        "checkpoint_seed_keys_exact": observed == expected,
        "rerun_rows_exact": first == second,
        "all_rows_successful_and_finite": all(
            row["status"] == "ok"
            and all(
                math.isfinite(float(row[field]))
                for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
            )
            for row in first
        ),
        "confirmatory_seeds_untouched": not set(SMOKE_SEEDS).intersection(
            CONFIRMATORY_SEEDS
        ),
        "swap_fields_absent": all(
            field not in first[0]
            for field in ("donor_model", "swap_window", "window_name")
        ),
    }
    validation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "repository_commit": engine.git_commit(),
        "source_pool_identity": identity,
        "seeds": list(SMOKE_SEEDS),
        "checks": checks,
    }
    engine.dump_json(output_dir / SMOKE_VALIDATION_FILENAME, validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"Stage B evaluation smoke failed: {validation}")


def read_bool(value: object) -> bool:
    """Parse one CSV boolean deterministically."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def select_pair(new_rows, small_rows) -> dict[str, object] | None:
    """Apply frozen rate-difference then checkpoint-SHA ordering."""
    ranked = []
    for new in (row for row in new_rows if read_bool(row["eligible"])):
        for small in (row for row in small_rows if read_bool(row["eligible"])):
            difference = abs(
                float(new["memorization_rate"]) - float(small["memorization_rate"])
            )
            ranked.append(
                (
                    difference,
                    str(new["checkpoint_sha256"]),
                    str(small["checkpoint_sha256"]),
                    new,
                    small,
                )
            )
    if not ranked:
        return None
    difference, _, _, new, small = min(ranked, key=lambda item: item[:3])
    return {
        "absolute_baseline_rate_difference": difference,
        "edm_5k_checkpoint": dict(new),
        "edm_1k_checkpoint": dict(small),
    }


def merge_shards(engine, output_dir: Path) -> list[dict[str, object]]:
    """Merge two complete disjoint role shards into stable master rows."""
    rows = []
    for role in ROLES:
        shard = output_dir / "shards" / role
        role_rows = [
            engine.normalize_resume_row(row)
            for row in engine.read_csv_rows(shard / engine.PER_SAMPLE_FILENAME)
        ]
        expected = ROLE_COUNTS[role] * len(PILOT_SEEDS)
        if len(role_rows) != expected or any(
            row["model_role"] != role for row in role_rows
        ):
            raise RuntimeError(f"Incomplete Stage B pilot shard for {role}")
        rows.extend(role_rows)
    rows = core.merge_resume_rows([], rows)
    engine.write_csv(
        output_dir / engine.PER_SAMPLE_FILENAME, rows, engine.per_sample_header()
    )
    failures = [row for row in rows if row["status"] != "ok"]
    engine.write_csv(
        output_dir / engine.FAILURES_FILENAME, failures, engine.per_sample_header()
    )
    return rows


def summarize(engine, config: Mapping[str, Any], output_dir: Path) -> None:
    """Aggregate all 24 checkpoints and apply the frozen Stage B stop rule."""
    identity = verify_pool_identity(engine, output_dir)
    require_execution_identity(engine, config)
    samples = merge_shards(engine, output_dir)
    inventory = engine.read_csv_rows(output_dir / engine.INVENTORY_FILENAME)
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
    new_rows = [row for row in summaries if row["model_role"] == "edm_5k"]
    small_rows = [row for row in summaries if row["model_role"] == "edm_1k"]
    pair = select_pair(new_rows, small_rows)
    eligible_5k = sum(read_bool(row["eligible"]) for row in new_rows)
    eligible_1k = sum(read_bool(row["eligible"]) for row in small_rows)
    if pair is not None:
        outcome = "ELIGIBLE_5K_PAIR_FROZEN"
    elif eligible_5k == 0:
        outcome = "BLOCKED_NO_ELIGIBLE_5K_THROUGH_30K"
    else:
        outcome = "BLOCKED_NO_SAME_SEED_ELIGIBLE_EDM_1K"
    engine.dump_json(
        output_dir / engine.PAIR_FILENAME,
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "outcome": outcome,
            "selected_pair": pair,
            "selection_rule": config["pair_selection"],
            "pool_identity": identity,
            "evaluation_seeds": list(PILOT_SEEDS),
            "reserved_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
            "repository_commit": engine.git_commit(),
            "e008_executed": False,
        },
    )
    engine.dump_json(
        output_dir / engine.OUTCOME_FILENAME,
        {
            "experiment_id": EXPERIMENT_ID,
            "outcome": outcome,
            "eligible_edm_5k_count": eligible_5k,
            "eligible_edm_1k_count": eligible_1k,
            "automatic_extension_started": False,
            "e008_executed": False,
        },
    )
    generate_figure(engine, summaries, output_dir)
    accepted = [row for row in inventory if row["inventory_status"] == "accepted"]
    expected_keys = {
        (row["checkpoint_sha256"], seed) for row in accepted for seed in PILOT_SEEDS
    }
    observed_keys = {
        (row["checkpoint_sha256"], int(row["sample_seed"])) for row in samples
    }
    checks = {
        "accepted_checkpoint_count_is_24": len(accepted) == 24,
        "per_sample_row_count_is_3072": len(samples) == 3072,
        "checkpoint_seed_keys_exact": observed_keys == expected_keys,
        "checkpoint_seed_keys_unique": len(observed_keys) == len(samples),
        "confirmatory_seed_overlap_absent": not {
            int(row["sample_seed"]) for row in samples
        }.intersection(CONFIRMATORY_SEEDS),
        "all_records_successful": all(row["status"] == "ok" for row in samples),
        "successful_distances_finite": all(
            all(
                math.isfinite(float(row[field]))
                for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
            )
            for row in samples
        ),
        "swap_fields_absent": all(
            field not in samples[0]
            for field in ("donor_model", "swap_window", "window_name")
        ),
        "e008_unexecuted": not config["scientific_scope"]["e008_executed"],
    }
    validation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "outcome": outcome,
        "failed_row_count": sum(row["status"] != "ok" for row in samples),
        "pool_identity": identity,
    }
    engine.dump_json(output_dir / engine.VALIDATION_FILENAME, validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"Stage B baseline validation failed: {validation}")


def generate_figure(
    engine, summaries: Sequence[Mapping[str, object]], output_dir: Path
) -> None:
    """Plot both frozen same-seed baseline trajectories."""
    figure, axes = engine.plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, role, title in zip(
        axes, ROLES, ("5K warm-start extension", "EDM-1K historical")
    ):
        rows = [row for row in summaries if row["model_role"] == role]
        axis.axhspan(0.1, 0.9, color="#DDE8D5", alpha=0.65, label="Eligible")
        axis.plot(
            [int(row["training_kimg"]) for row in rows],
            [float(row["memorization_rate"]) for row in rows],
            color="#1F5D85" if role == "edm_5k" else "#C8563A",
            marker="o",
            linewidth=2,
        )
        axis.set(title=title, xlabel="Training (kimg)", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Baseline memorization rate")
    axes[0].legend(loc="best")
    figure.suptitle("E009 Stage B same-seed baseline eligibility")
    figure.tight_layout()
    figure.savefig(output_dir / RATE_FIGURE_FILENAME, dpi=180, bbox_inches="tight")
    engine.plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse one bounded execution mode."""
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
    """Freeze, smoke, evaluate, or summarize the Stage B baseline cohort."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Refusing to run Stage B evaluation outside Slurm")
    args = parse_args()
    config = load_config()
    engine = load_engine()
    configure_engine(engine)
    if args.inventory_only:
        freeze_inventory(engine, config, args.output_dir.resolve())
    elif args.smoke:
        if args.role is not None:
            raise ValueError("--role is not valid with --smoke")
        run_smoke(engine, config, args.output_dir.resolve())
    elif args.run_pilot:
        if args.role is None:
            raise ValueError("--run-pilot requires --role")
        identity = verify_pool_identity(engine, args.output_dir.resolve())
        require_execution_identity(engine, config)
        engine.summarize = lambda *_: None
        shard = args.output_dir.resolve() / "shards" / args.role
        engine.run_pilot(build_args(config, shard), config)
        rewrite_run_manifest(engine, shard, identity, "full_pilot")
    else:
        if args.role is not None:
            raise ValueError("--role is valid only with --run-pilot")
        summarize(engine, config, args.output_dir.resolve())


if __name__ == "__main__":
    main()
