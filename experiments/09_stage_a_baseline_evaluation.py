#!/usr/bin/env python3
"""Pre-staged baseline-only evaluator for the frozen E009 Stage A pool."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
    rows: list[dict[str, object]] = []
    expected_kimg = config["expected_training_kimg"]
    for role, root_string in config["candidate_roots"].items():
        root = Path(root_string).resolve()
        accepted, malformed = core.discover_checkpoint_paths(root)
        observed_kimg = [core.parse_training_kimg(path.name) for path in accepted]
        if observed_kimg != expected_kimg or malformed:
            raise RuntimeError(
                f"Incomplete checkpoint inventory for {role}: "
                f"observed={observed_kimg}, malformed={malformed}"
            )
        config_source = engine.find_training_config(root)
        config_hash = core.sha256_file(config_source) if config_source else ""
        for path in accepted:
            row = engine.base_inventory_record(
                role,
                path,
                int(config["candidate_subset_sizes"][role]),
                config_source,
                config_hash,
            )
            row.update(engine.inspect_checkpoint(path, args.edm_root))
            row["inventory_status"] = "accepted"
            row["rejection_reason"] = ""
            rows.append(row)
    rows.sort(key=lambda row: (str(row["model_role"]), int(row["training_kimg"])))
    output_dir.mkdir(parents=True)
    inventory_path = output_dir / engine.INVENTORY_FILENAME
    engine.write_csv(inventory_path, rows, engine.inventory_header())
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "pool_frozen_before_pilot": True,
        "pilot_started": False,
        "scientific_scope": config["scientific_scope"],
        "config_sha256": core.sha256_file(CONFIG_PATH),
        "repository_commit": engine.git_commit(),
        "provenance": provenance,
        "pilot_seeds": list(PILOT_SEEDS),
        "future_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "inventory": {
            "path": str(inventory_path),
            "sha256": core.sha256_file(inventory_path),
            "row_count": len(rows),
            "accepted_count": len(rows),
            "rejected_count": 0,
            "records": rows,
        },
    }
    engine.dump_json(output_dir / engine.POOL_MANIFEST_FILENAME, manifest)


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
        outcome = "ELIGIBLE_PAIR_FROZEN"
    elif eligible_2k:
        outcome = "PROVISIONAL_2K_ONLY_STAGE_B_REQUIRED"
    else:
        outcome = "STAGE_B_REQUIRED_NO_ELIGIBLE_CHECKPOINT"
    engine.dump_json(
        output_dir / engine.PAIR_FILENAME,
        {"experiment_id": EXPERIMENT_ID, "outcome": outcome, "selected_pair": pair},
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
        },
    )
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
        "all_rows_successful": all(row["status"] == "ok" for row in samples),
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
    }
    engine.dump_json(output_dir / engine.VALIDATION_FILENAME, validation)
    if validation["status"] != "pass":
        raise RuntimeError(f"E009 baseline validation failed: {validation}")


def parse_args() -> argparse.Namespace:
    """Parse a pre-staged mode; no mode is run automatically."""
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inventory-only", action="store_true")
    modes.add_argument("--run-pilot", action="store_true")
    modes.add_argument("--summarize", action="store_true")
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
    elif args.run_pilot:
        engine.summarize = lambda *_: None
        engine.run_pilot(build_args(config, args.output_dir.resolve()), config)
    else:
        summarize(engine, config, args.output_dir.resolve())


if __name__ == "__main__":
    main()
