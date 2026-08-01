#!/usr/bin/env python3
"""Compute, validate, or plot E004B frequency-restricted paper geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spectral-diffusion-e004b-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import scipy

import _bootstrap  # noqa: F401

from spectral_diffusion_playground.frequency_restricted_geometry import (
    IMAGE_DIMENSION,
    band_projector_rank,
    evaluate_frequency_restricted_geometry,
    summarize_targets,
)
from spectral_diffusion_playground.paper_geometry_evaluation import (
    index_text_sha256,
    load_cifar10_subsets,
    load_config as load_e004a_config,
    sha256_file,
    subset_manifest_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "e004b_frequency_restricted_geometry.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "experiment_04b"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "experiment_04b"
CSV_NAME = "frequency_restricted_geometry.csv"
MANIFEST_NAME = "frequency_restricted_geometry_manifest.json"
VALIDATION_NAME = "frequency_restricted_geometry_validation.json"
TARGET_NAME = "band_target_summary.json"
SENSITIVITY_NAME = "cutoff_sensitivity.json"

CURVE_FIELDS = (
    "cutoff",
    "band",
    "sigma_index",
    "sigma",
    "band_dimension",
    "coverage_estimate",
    "coverage_ci95_low",
    "coverage_ci95_high",
    "coverage_monte_carlo_se",
    "posterior_weight_estimate",
    "posterior_weight_ci95_low",
    "posterior_weight_ci95_high",
    "posterior_weight_monte_carlo_se",
    "coverage_ge_q_c",
    "posterior_ge_q_w",
    "high_high_point_estimate",
    "high_high_lower_bound",
)


def parse_args() -> argparse.Namespace:
    """Parse mutually exclusive compute, plot-only, and validation modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--compute", action="store_true")
    modes.add_argument("--plot-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic human-readable JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_head() -> str | None:
    """Return the current repository commit when available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_clean() -> bool | None:
    """Return whether the repository worktree is clean when Git is available."""
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return not bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _peak_rss_megabytes() -> float:
    """Return process peak resident memory in platform-correct MiB."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 * 1024.0)


def load_e004b_config(path: Path) -> dict[str, Any]:
    """Load E004B and verify its frozen relationship to the E004A dataset."""
    config = json.loads(path.read_text(encoding="utf-8"))
    base_path = REPO_ROOT / config["base_e004a_config"]
    if sha256_file(base_path) != config["base_e004a_config_sha256"]:
        raise ValueError("Frozen E004A base-config hash mismatch")
    base = load_e004a_config(base_path)
    dataset = config["dataset"]
    base_dataset = base["dataset"]
    for key in (
        "normalization",
        "flattened_dimension",
        "training_index_text_sha256",
        "test_index_text_sha256",
        "dataset_file_sha256",
    ):
        if dataset[key] != base_dataset[key]:
            raise ValueError(f"E004B dataset field differs from E004A: {key}")
    if base_dataset["training_subset_indices"] != list(range(1000)):
        raise ValueError("E004A training selection is no longer first 1000")
    if base_dataset["test_subset_indices"] != list(range(1000)):
        raise ValueError("E004A test selection is no longer first 1000")
    if config["fourier"]["primary_cutoff"] != 4:
        raise ValueError("E004B primary cutoff must remain r=4")
    if config["fourier"]["sensitivity_cutoffs"] != [3, 5]:
        raise ValueError("E004B sensitivity cutoffs must remain r=3,5")
    if float(config["shell_c"]) != 5.0:
        raise ValueError("E004B shell constant must remain c=5")
    config["dataset"]["training_subset_indices"] = base_dataset[
        "training_subset_indices"
    ]
    config["dataset"]["test_subset_indices"] = base_dataset["test_subset_indices"]
    return config


def write_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the stable long-format E004B curve schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_curve_csv(path: Path) -> list[dict[str, Any]]:
    """Read and type the stable long-format E004B curve schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CURVE_FIELDS:
            raise ValueError("Unexpected E004B CSV schema")
        rows = []
        for raw in reader:
            row: dict[str, Any] = {
                "cutoff": int(raw["cutoff"]),
                "band": raw["band"],
                "sigma_index": int(raw["sigma_index"]),
                "sigma": float(raw["sigma"]),
                "band_dimension": int(raw["band_dimension"]),
            }
            for key in (
                "coverage_estimate",
                "coverage_ci95_low",
                "coverage_ci95_high",
                "coverage_monte_carlo_se",
                "posterior_weight_estimate",
                "posterior_weight_ci95_low",
                "posterior_weight_ci95_high",
                "posterior_weight_monte_carlo_se",
            ):
                row[key] = float(raw[key])
            for key in (
                "coverage_ge_q_c",
                "posterior_ge_q_w",
                "high_high_point_estimate",
                "high_high_lower_bound",
            ):
                if raw[key] not in {"True", "False"}:
                    raise ValueError(f"Invalid boolean value for {key}")
                row[key] = raw[key] == "True"
            rows.append(row)
    return rows


def _rows_for(
    rows: list[dict[str, Any]], cutoff: int, band: str
) -> list[dict[str, Any]]:
    """Select one curve in canonical sigma-index order."""
    selected = [row for row in rows if row["cutoff"] == cutoff and row["band"] == band]
    return sorted(selected, key=lambda row: int(row["sigma_index"]))


def _plot_band_geometry(
    rows: list[dict[str, Any]], cutoff: int, band: str, output_path: Path
) -> None:
    """Plot coverage and posterior concentration for one projected band."""
    selected = _rows_for(rows, cutoff, band)
    sigma = np.asarray([row["sigma"] for row in selected], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    for prefix, label, color, marker in (
        ("coverage", f"{band.capitalize()}-frequency coverage", "#d97706", "s"),
        (
            "posterior_weight",
            f"{band.capitalize()}-frequency maximum posterior weight",
            "#0f766e",
            "o",
        ),
    ):
        estimate = np.asarray([row[f"{prefix}_estimate"] for row in selected])
        lower = np.asarray([row[f"{prefix}_ci95_low"] for row in selected])
        upper = np.asarray([row[f"{prefix}_ci95_high"] for row in selected])
        axis.plot(sigma, estimate, color=color, marker=marker, linewidth=2, label=label)
        axis.fill_between(sigma, lower, upper, color=color, alpha=0.16)
    target = np.asarray([row["high_high_lower_bound"] for row in selected], dtype=bool)
    axis.scatter(
        sigma[target],
        np.full(np.count_nonzero(target), 0.8),
        marker="*",
        s=190,
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Lower-bound high-high grid point",
    )
    axis.axhline(0.8, color="#111827", linestyle="--", linewidth=1.2, alpha=0.7)
    axis.set_xscale("log")
    axis.set_xlim(float(sigma.min()), float(sigma.max()))
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel(r"Noise scale $\sigma$ (small to large)")
    axis.set_ylabel("Probability / concentration")
    axis.set_title(
        f"{band.capitalize()}-Frequency Gaussian-Shell Geometry "
        f"(r={cutoff}, d={selected[0]['band_dimension']})"
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(loc="best", fontsize=9)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_four_curve_comparison(
    rows: list[dict[str, Any]], cutoff: int, output_path: Path
) -> None:
    """Plot all four frequency-restricted geometric quantities."""
    figure, axis = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    styles = {
        ("low", "coverage"): ("#d97706", "s", "-"),
        ("low", "posterior_weight"): ("#0f766e", "o", "-"),
        ("high", "coverage"): ("#c2410c", "^", "--"),
        ("high", "posterior_weight"): ("#2563eb", "D", "--"),
    }
    for band in ("low", "high"):
        selected = _rows_for(rows, cutoff, band)
        sigma = np.asarray([row["sigma"] for row in selected])
        for prefix, noun in (
            ("coverage", "coverage"),
            ("posterior_weight", "maximum posterior weight"),
        ):
            color, marker, linestyle = styles[(band, prefix)]
            estimate = np.asarray([row[f"{prefix}_estimate"] for row in selected])
            axis.plot(
                sigma,
                estimate,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                label=f"{band.capitalize()}-frequency {noun}",
            )
    axis.axhline(0.8, color="#111827", linestyle=":", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel(r"Noise scale $\sigma$ (small to large)")
    axis.set_ylabel("Probability / concentration")
    axis.set_title(f"Frequency-Restricted Geometry Comparison (r={cutoff})")
    axis.grid(alpha=0.2, which="both")
    axis.legend(loc="best", fontsize=9)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _shade_index_interval(
    axis: Any, sigma: np.ndarray, start: int, end: int, **kwargs: Any
) -> None:
    """Shade one inclusive interval on the descending exact sigma schedule."""
    axis.axvspan(float(sigma[end]), float(sigma[start]), **kwargs)


def _plot_alignment(rows: list[dict[str, Any]], cutoff: int, output_path: Path) -> None:
    """Align band geometry targets with full-space and residual intervals."""
    low = _rows_for(rows, cutoff, "low")
    high = _rows_for(rows, cutoff, "high")
    sigma = np.asarray([row["sigma"] for row in low])
    figure, axis = plt.subplots(figsize=(11.2, 6.5), constrained_layout=True)
    _shade_index_interval(
        axis,
        sigma,
        5,
        11,
        color="#2563eb",
        alpha=0.10,
        label="E005 low-frequency residual transition (5..11)",
    )
    _shade_index_interval(
        axis,
        sigma,
        11,
        14,
        color="#be123c",
        alpha=0.10,
        label="E005 high-frequency residual transition (11..14)",
    )
    axis.scatter(
        sigma[[8, 9]],
        np.full(2, 0.25),
        marker="X",
        s=115,
        color="#64748b",
        label="E004A full-space geometry target (8..9)",
    )
    for selected, band, y_value, color, marker in (
        (low, "Low", 0.72, "#0f766e", "o"),
        (high, "High", 0.52, "#d97706", "s"),
    ):
        mask = np.asarray(
            [row["high_high_lower_bound"] for row in selected], dtype=bool
        )
        axis.scatter(
            sigma[mask],
            np.full(np.count_nonzero(mask), y_value),
            marker=marker,
            s=105,
            color=color,
            label=f"E004B {band.lower()}-frequency geometry target",
        )
    axis.set_xscale("log")
    axis.set_xlim(float(sigma.min()), float(sigma.max()))
    axis.set_ylim(0.08, 0.9)
    axis.set_yticks([0.25, 0.52, 0.72])
    axis.set_yticklabels(
        ["Full-space geometry", "High-band geometry", "Low-band geometry"]
    )
    axis.set_xlabel(r"Noise scale $\sigma$ (small to large)")
    axis.set_title("Frequency-Restricted Geometry and E005 Residual Transitions")
    axis.grid(alpha=0.18, which="both", axis="x")
    axis.legend(loc="best", fontsize=8.6)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_cutoff_sensitivity(
    rows: list[dict[str, Any]], cutoffs: list[int], output_path: Path
) -> None:
    """Plot lower-bound high-high classifications over adjacent cutoffs."""
    sigma = np.asarray([row["sigma"] for row in _rows_for(rows, cutoffs[0], "low")])
    matrix = []
    labels = []
    for cutoff in cutoffs:
        for band in ("low", "high"):
            selected = _rows_for(rows, cutoff, band)
            matrix.append([int(row["high_high_lower_bound"]) for row in selected])
            labels.append(f"r={cutoff} {band}")
    figure, axis = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    image = axis.imshow(np.asarray(matrix), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xticks(np.arange(len(sigma)))
    axis.set_xticklabels([f"{value:.3g}" for value in sigma], rotation=55, ha="right")
    axis.set_xlabel(r"Exact sampler noise scale $\sigma$")
    axis.set_title("E004B Lower-Bound Geometry Targets Across Frozen Cutoffs")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["not selected", "selected"])
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def generate_figures(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    primary_cutoff: int,
    cutoffs: list[int],
) -> list[Path]:
    """Generate the five required E004B review figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        figure_dir / "low_frequency_coverage_and_posterior.png",
        figure_dir / "high_frequency_coverage_and_posterior.png",
        figure_dir / "low_high_geometry_comparison.png",
        figure_dir / "band_geometry_and_residual_alignment.png",
        figure_dir / "frequency_geometry_cutoff_sensitivity.png",
    ]
    _plot_band_geometry(rows, primary_cutoff, "low", outputs[0])
    _plot_band_geometry(rows, primary_cutoff, "high", outputs[1])
    _plot_four_curve_comparison(rows, primary_cutoff, outputs[2])
    _plot_alignment(rows, primary_cutoff, outputs[3])
    _plot_cutoff_sensitivity(rows, cutoffs, outputs[4])
    return outputs


def validate_outputs(output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    """Validate schemas, finite values, target derivation, and output files."""
    rows = read_curve_csv(output_dir / CSV_NAME)
    target = json.loads((output_dir / TARGET_NAME).read_text(encoding="utf-8"))
    numerical = json.loads((output_dir / VALIDATION_NAME).read_text(encoding="utf-8"))
    expected_rows = 3 * 2 * 18
    numeric_keys = [
        key
        for key in CURVE_FIELDS
        if key
        not in {
            "band",
            "coverage_ge_q_c",
            "posterior_ge_q_w",
            "high_high_point_estimate",
            "high_high_lower_bound",
        }
    ]
    finite = bool(
        np.all(np.isfinite([[float(row[key]) for key in numeric_keys] for row in rows]))
    )
    primary_rows = {
        band: _rows_for(rows, int(target["primary_cutoff"]), band)
        for band in ("low", "high")
    }
    target_recomputed = {
        band: [
            int(row["sigma_index"])
            for row in band_rows
            if row["coverage_ci95_low"] >= target["q_C"]
            and row["posterior_weight_ci95_low"] >= target["q_W"]
        ]
        for band, band_rows in primary_rows.items()
    }
    figures = sorted(figure_dir.glob("*.png"))
    checks = {
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "all_numeric_values_finite": finite,
        "numerical_validation_status": numerical["status"],
        "low_target_recomputed": target_recomputed["low"],
        "high_target_recomputed": target_recomputed["high"],
        "target_derivation_matches": target_recomputed["low"]
        == target["low_lower_bound_indices"]
        and target_recomputed["high"] == target["high_lower_bound_indices"],
        "figure_count": len(figures),
        "all_figures_nonempty": all(path.stat().st_size > 0 for path in figures),
        "e005_residual_data_used_for_target_selection": False,
        "committed_curve_estimates_read_in_compute_mode": False,
    }
    checks["status"] = (
        "pass"
        if checks["row_count"] == checks["expected_row_count"]
        and checks["all_numeric_values_finite"]
        and checks["numerical_validation_status"] == "pass"
        and checks["target_derivation_matches"]
        and checks["figure_count"] == 5
        and checks["all_figures_nonempty"]
        else "fail"
    )
    return checks


def compute(args: argparse.Namespace) -> dict[str, Any]:
    """Run E004B from CIFAR-10 and write fresh compact artifacts."""
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required with --compute")
    if args.device.lower() not in {"auto", "cpu"}:
        raise ValueError("The E004B NumPy oracle supports auto/cpu only")
    output_dir = args.output_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_dir}")
    if figure_dir.exists() and any(figure_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty figures: {figure_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    clean_before = _git_clean()
    config = load_e004b_config(args.config)
    cutoffs = [int(value) for value in args.cutoffs]
    primary_cutoff = int(config["fourier"]["primary_cutoff"])
    if primary_cutoff not in cutoffs:
        raise ValueError("--cutoffs must include the frozen primary cutoff r=4")
    dataset = config["dataset"]
    training, test, observed_hashes = load_cifar10_subsets(
        args.dataset_root,
        dataset["training_subset_indices"],
        dataset["test_subset_indices"],
        dataset["dataset_file_sha256"],
    )
    q_c = float(config["thresholds"]["coverage"])
    q_w = float(config["thresholds"]["maximum_posterior_weight"])
    rows, numerical_validation, targets = evaluate_frequency_restricted_geometry(
        training,
        test,
        cutoffs=cutoffs,
        sigmas=config["sigma_grid"],
        shell_c=float(config["shell_c"]),
        posterior_draws=int(config["posterior_corruption_draws"]),
        coverage_draws=int(config["coverage_corruption_draws"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["random_seed"]),
        query_batch_size=int(config["query_batch_size"]),
        reference_batch_size=int(config["reference_batch_size"]),
        q_coverage=q_c,
        q_posterior_weight=q_w,
    )
    target_summary = summarize_targets(
        targets,
        primary_cutoff=primary_cutoff,
        q_coverage=q_c,
        q_weight=q_w,
    )
    sensitivity = {
        "primary_cutoff": primary_cutoff,
        "candidate_cutoffs": cutoffs,
        "bands": {
            band: {
                str(cutoff): targets["cutoffs"][str(cutoff)][band] for cutoff in cutoffs
            }
            for band in ("low", "high")
        },
        "primary_target_not_revised_by_sensitivity": True,
    }
    write_curve_csv(output_dir / CSV_NAME, rows)
    _write_json(output_dir / VALIDATION_NAME, numerical_validation)
    _write_json(output_dir / TARGET_NAME, target_summary)
    _write_json(output_dir / SENSITIVITY_NAME, sensitivity)
    figure_paths = generate_figures(rows, figure_dir, primary_cutoff, cutoffs)
    output_validation = validate_outputs(output_dir, figure_dir)
    if output_validation["status"] != "pass":
        raise RuntimeError(f"E004B output validation failed: {output_validation}")
    manifest = {
        "experiment_id": "E004B",
        "title": "Frequency-Restricted Gaussian-Shell Geometry",
        "execution_mode": "local_cpu_clean_room_compute",
        "repository_commit": _git_head(),
        "repository_worktree_clean_before_run": clean_before,
        "command": [sys.executable, *sys.argv],
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "base_e004a_config": config["base_e004a_config"],
        "base_e004a_config_sha256": config["base_e004a_config_sha256"],
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "dataset_file_sha256": observed_hashes,
        "training_index_text_sha256": index_text_sha256(
            dataset["training_subset_indices"]
        ),
        "test_index_text_sha256": index_text_sha256(dataset["test_subset_indices"]),
        "subset_sha256": subset_manifest_sha256(
            dataset["training_subset_indices"],
            dataset["test_subset_indices"],
        ),
        "normalization": dataset["normalization"],
        "training_examples": len(training),
        "test_examples": len(test),
        "storage_dimension": IMAGE_DIMENSION,
        "sigma_grid": config["sigma_grid"],
        "cutoffs": cutoffs,
        "primary_cutoff": primary_cutoff,
        "band_dimensions": {
            str(cutoff): {
                band: band_projector_rank(cutoff, band) for band in ("low", "high")
            }
            for cutoff in cutoffs
        },
        "shell_c": config["shell_c"],
        "posterior_corruption_draws": config["posterior_corruption_draws"],
        "coverage_corruption_draws": config["coverage_corruption_draws"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "random_seed": config["random_seed"],
        "query_batch_size": config["query_batch_size"],
        "reference_batch_size": config["reference_batch_size"],
        "thresholds": config["thresholds"],
        "noise_policy": config["noise_policy"],
        "distance_policy": config["distance_policy"],
        "sigma_policy": config["sigma_policy"],
        "device_requested": args.device,
        "device_resolved": "cpu",
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_megabytes": _peak_rss_megabytes(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "target_summary": target_summary,
        "validation": output_validation,
        "artifacts": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in [
                output_dir / CSV_NAME,
                output_dir / VALIDATION_NAME,
                output_dir / TARGET_NAME,
                output_dir / SENSITIVITY_NAME,
                *figure_paths,
            ]
        },
        "scientific_scope": {
            "e004a_full_space_baseline_preserved": True,
            "e005_residual_curves_distinct": True,
            "band_targets_selected_without_e005": True,
            "e008_executed": False,
        },
    }
    _write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def main() -> None:
    """Dispatch compute, plot-only, or validate-only mode."""
    args = parse_args()
    if args.compute:
        manifest = compute(args)
        print(json.dumps(manifest["target_summary"], indent=2, sort_keys=True))
        print(f"runtime_seconds={manifest['runtime_seconds']:.3f}")
        print(f"peak_rss_megabytes={manifest['peak_rss_megabytes']:.3f}")
        return
    rows = read_curve_csv(args.output_dir / CSV_NAME)
    config = load_e004b_config(args.config)
    cutoffs = [int(value) for value in args.cutoffs]
    if args.plot_only:
        generate_figures(
            rows,
            args.figure_dir,
            int(config["fourier"]["primary_cutoff"]),
            cutoffs,
        )
        return
    validation = validate_outputs(args.output_dir, args.figure_dir)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
