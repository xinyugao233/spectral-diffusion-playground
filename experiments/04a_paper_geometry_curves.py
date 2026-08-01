#!/usr/bin/env python3
"""Compute, validate, or plot the E004A clean-room paper geometry baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spectral-diffusion-e004a-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from spectral_diffusion_playground.paper_geometry import SIGMA_GRID
from spectral_diffusion_playground.paper_geometry_evaluation import (
    compute_and_write,
    compare_reproduction,
    evaluate_geometry,
    finalize_manifest,
    load_config,
    load_cifar10_subsets,
    read_curve_csv,
    resolve_device,
    sha256_file,
    subset_manifest_sha256,
    write_comparison,
)
from spectral_diffusion_playground.region_definitions import (
    contiguous_components,
    high_high_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_RESULTS_DIR = REPO_ROOT / "results" / "experiment_04a"
DEFAULT_REPRODUCTION_DIR = REPO_ROOT / "results" / "experiment_04a_reproduction"
CONFIG_PATH = REPO_ROOT / "configs" / "e004a_local_geometry.json"
FIGURES_DIR = REPO_ROOT / "figures" / "experiment_04a"
E006_GRID_RESULTS = COMMITTED_RESULTS_DIR / "e006_grid_geometry.csv"
E006_GRID_MANIFEST = COMMITTED_RESULTS_DIR / "e006_grid_geometry_manifest.json"
E006_GRID_VALIDATION = COMMITTED_RESULTS_DIR / "e006_grid_geometry_validation.json"
E006_GRID_FIGURE = FIGURES_DIR / "e006_grid_geometry_alignment.png"

E005_SIGMA_GRID = np.asarray(
    [
        80.0,
        57.58598472124816,
        40.78557379650796,
        28.374584604156844,
        19.35245298032523,
        12.91008238075732,
        8.400935309099817,
        5.315194521796382,
        3.256821519765537,
        1.9233398370400518,
        1.088170636545279,
        0.5853481231945422,
        0.29644228447915727,
        0.13951646873101678,
        0.05994731123547159,
        0.022934518372333384,
        0.0075280199627840785,
        0.002000000000000003,
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse mutually exclusive compute, plot, and validation modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compute", action="store_true")
    modes.add_argument("--compute-e006-grid", action="store_true")
    modes.add_argument("--plot-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--geometry-figure",
        type=Path,
        help="Override the primary figure path in plot-only mode.",
    )
    parser.add_argument(
        "--comparison-figure",
        type=Path,
        help="Override the comparison figure path in plot-only mode.",
    )
    return parser.parse_args()


def _git_head() -> str:
    """Return the repository commit recorded for a generated artifact."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_e006_grid_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the frozen E006-grid geometry schema."""
    fields = [
        "sigma_index",
        "sigma",
        "coverage_estimate",
        "coverage_ci95_low",
        "coverage_ci95_high",
        "posterior_weight_estimate",
        "posterior_weight_ci95_low",
        "posterior_weight_ci95_high",
        "coverage_ge_q_c",
        "posterior_weight_ge_q_w",
        "high_high_point_estimate",
        "high_high_lower_bound",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_e006_grid_alignment(rows: list[dict[str, object]], output_path: Path) -> None:
    """Plot evaluated geometry and historically distinct E005/E006 regions."""
    sigma = np.asarray([float(row["sigma"]) for row in rows])
    coverage = np.asarray([float(row["coverage_estimate"]) for row in rows])
    coverage_low = np.asarray([float(row["coverage_ci95_low"]) for row in rows])
    coverage_high = np.asarray([float(row["coverage_ci95_high"]) for row in rows])
    posterior = np.asarray([float(row["posterior_weight_estimate"]) for row in rows])
    posterior_low = np.asarray(
        [float(row["posterior_weight_ci95_low"]) for row in rows]
    )
    posterior_high = np.asarray(
        [float(row["posterior_weight_ci95_high"]) for row in rows]
    )
    point_mask = np.asarray([bool(row["high_high_point_estimate"]) for row in rows])

    figure, axis = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    axis.axvspan(
        E005_SIGMA_GRID[13],
        E005_SIGMA_GRID[6],
        facecolor="#64748b",
        alpha=0.10,
        hatch="////",
        edgecolor="#475569",
        label="Table 1 / Figure 10 medium-window reference (indices 6..13)",
    )
    axis.axvspan(
        E005_SIGMA_GRID[11],
        E005_SIGMA_GRID[5],
        color="#2563eb",
        alpha=0.08,
        label="E005 low-frequency spectral transition (indices 5..11)",
    )
    axis.axvspan(
        E005_SIGMA_GRID[14],
        E005_SIGMA_GRID[11],
        color="#be123c",
        alpha=0.08,
        label="E005 high-frequency spectral transition (indices 11..14)",
    )
    for values, low, high, label, color, marker in (
        (
            coverage,
            coverage_low,
            coverage_high,
            r"Coverage $C_\sigma(p,D)$",
            "#d97706",
            "s",
        ),
        (
            posterior,
            posterior_low,
            posterior_high,
            r"Max posterior weight $W_\sigma(D)$",
            "#0f766e",
            "o",
        ),
    ):
        axis.plot(sigma, values, color=color, marker=marker, linewidth=2, label=label)
        axis.fill_between(sigma, low, high, color=color, alpha=0.15)
    axis.scatter(
        sigma[point_mask],
        np.full(np.count_nonzero(point_mask), 0.8),
        marker="*",
        s=180,
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label=r"E004A high-high evaluated points ($q_C=q_W=0.8$)",
    )
    axis.axhline(0.8, color="#111827", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xscale("log")
    axis.set_xlim(float(sigma.min()), float(sigma.max()))
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel(r"Noise scale $\sigma$ (small to large)")
    axis.set_ylabel("Probability / concentration")
    axis.set_title("Geometry evaluated on the exact E006 schedule")
    axis.grid(alpha=0.2, which="both")
    axis.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def compute_e006_grid_geometry(dataset_root: Path, device: str) -> dict[str, object]:
    """Evaluate E004A definitions directly at every frozen E006 sigma point."""
    started = time.perf_counter()
    config = load_config(CONFIG_PATH)
    resolve_device(device)
    dataset = config["dataset"]
    training, test, dataset_hashes = load_cifar10_subsets(
        dataset_root,
        dataset["training_subset_indices"],
        dataset["test_subset_indices"],
        dataset["dataset_file_sha256"],
    )
    curves, numerical_validation = evaluate_geometry(
        training,
        test,
        # The geometry oracle validates ascending grids; E006 records sampler
        # calls in descending sigma order. Reverse only the returned sigma axis.
        sigmas=E005_SIGMA_GRID[::-1],
        shell_c=float(config["shell_c"]),
        posterior_draws=int(config["posterior_corruption_draws"]),
        coverage_draws=int(config["coverage_corruption_draws"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["random_seed"]),
        query_batch_size=int(config["query_batch_size"]),
        reference_batch_size=int(config["reference_batch_size"]),
    )
    curves = {name: values[::-1] for name, values in curves.items()}
    q_c = float(config["thresholds"]["coverage"])
    q_w = float(config["thresholds"]["maximum_posterior_weight"])
    rows: list[dict[str, object]] = []
    for index, sigma in enumerate(E005_SIGMA_GRID):
        coverage = float(curves["gaussian_shell_coverage"][index])
        coverage_low = float(curves["gaussian_shell_coverage_ci95_low"][index])
        posterior = float(curves["maximum_posterior_weight"][index])
        posterior_low = float(curves["maximum_posterior_weight_ci95_low"][index])
        rows.append(
            {
                "sigma_index": index,
                "sigma": float(sigma),
                "coverage_estimate": coverage,
                "coverage_ci95_low": coverage_low,
                "coverage_ci95_high": float(
                    curves["gaussian_shell_coverage_ci95_high"][index]
                ),
                "posterior_weight_estimate": posterior,
                "posterior_weight_ci95_low": posterior_low,
                "posterior_weight_ci95_high": float(
                    curves["maximum_posterior_weight_ci95_high"][index]
                ),
                "coverage_ge_q_c": coverage >= q_c,
                "posterior_weight_ge_q_w": posterior >= q_w,
                "high_high_point_estimate": coverage >= q_c and posterior >= q_w,
                "high_high_lower_bound": coverage_low >= q_c and posterior_low >= q_w,
            }
        )

    point_indices = high_high_indices(
        [float(row["coverage_estimate"]) for row in rows],
        [float(row["posterior_weight_estimate"]) for row in rows],
        q_coverage=q_c,
        q_posterior_weight=q_w,
    )
    lower_indices = high_high_indices(
        [float(row["coverage_ci95_low"]) for row in rows],
        [float(row["posterior_weight_ci95_low"]) for row in rows],
        q_coverage=q_c,
        q_posterior_weight=q_w,
    )
    validation = {
        **numerical_validation,
        "status": numerical_validation["status"],
        "rows": len(rows),
        "expected_rows": len(E005_SIGMA_GRID),
        "all_numeric_values_finite": bool(
            np.all(
                np.isfinite(
                    [
                        float(value)
                        for row in rows
                        for key, value in row.items()
                        if key
                        not in {
                            "coverage_ge_q_c",
                            "posterior_weight_ge_q_w",
                            "high_high_point_estimate",
                            "high_high_lower_bound",
                        }
                    ]
                )
            )
        ),
        "classification_uses_evaluated_points_only": True,
        "interpolation_used": False,
        "gap_filling_used": False,
    }
    if (
        validation["rows"] != validation["expected_rows"]
        or not validation["all_numeric_values_finite"]
    ):
        validation["status"] = "fail"

    COMMITTED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_e006_grid_csv(E006_GRID_RESULTS, rows)
    E006_GRID_VALIDATION.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_e006_grid_alignment(rows, E006_GRID_FIGURE)
    manifest: dict[str, object] = {
        "experiment_id": "E004A",
        "artifact": "e006_grid_geometry",
        "status": "completed" if validation["status"] == "pass" else "failed",
        "reproduction_claim": "paper-derived clean-room evaluation",
        "repository_commit_at_execution": _git_head(),
        "runtime_seconds": time.perf_counter() - started,
        "command": list(sys.argv),
        "device_requested": device,
        "device_resolved": "cpu",
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "source_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "source_config_sha256": sha256_file(CONFIG_PATH),
        "implementation_file_sha256": {
            "experiments/04a_paper_geometry_curves.py": sha256_file(Path(__file__)),
            "src/spectral_diffusion_playground/paper_geometry.py": sha256_file(
                REPO_ROOT
                / "src"
                / "spectral_diffusion_playground"
                / "paper_geometry.py"
            ),
            "src/spectral_diffusion_playground/paper_geometry_evaluation.py": sha256_file(
                REPO_ROOT
                / "src"
                / "spectral_diffusion_playground"
                / "paper_geometry_evaluation.py"
            ),
            "src/spectral_diffusion_playground/region_definitions.py": sha256_file(
                REPO_ROOT
                / "src"
                / "spectral_diffusion_playground"
                / "region_definitions.py"
            ),
        },
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "dataset_file_sha256": dataset_hashes,
        "normalization": dataset["normalization"],
        "flattened_dimension": 3072,
        "training_examples": len(training),
        "test_examples": len(test),
        "subset_sha256": subset_manifest_sha256(
            dataset["training_subset_indices"], dataset["test_subset_indices"]
        ),
        "seed": config["random_seed"],
        "shell_c": config["shell_c"],
        "posterior_corruption_draws": config["posterior_corruption_draws"],
        "coverage_corruption_draws": config["coverage_corruption_draws"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "query_batch_size": config["query_batch_size"],
        "reference_batch_size": config["reference_batch_size"],
        "thresholds": {"q_C": q_c, "q_W": q_w},
        "sigma_grid": [float(value) for value in E005_SIGMA_GRID],
        "classification_rule": (
            "qualify evaluated points independently; no interpolation or gap filling"
        ),
        "clean_room_geometry_high_high_indices": point_indices,
        "clean_room_geometry_high_high_components": contiguous_components(
            point_indices
        ),
        "clean_room_geometry_high_high_lower_bound_indices": lower_indices,
        "clean_room_geometry_high_high_lower_bound_components": contiguous_components(
            lower_indices
        ),
        "outputs": {
            "csv": str(E006_GRID_RESULTS.relative_to(REPO_ROOT)),
            "validation": str(E006_GRID_VALIDATION.relative_to(REPO_ROOT)),
            "figure": str(E006_GRID_FIGURE.relative_to(REPO_ROOT)),
        },
    }
    E006_GRID_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_results(results_dir: Path) -> dict[str, object]:
    """Validate curve schemas, values, and the frozen sigma grid."""
    coverage = read_curve_csv(results_dir / "coverage_curve.csv")
    posterior = read_curve_csv(results_dir / "max_posterior_weight_curve.csv")
    if tuple(float(row["sigma"]) for row in coverage) != SIGMA_GRID:
        raise ValueError("Coverage sigma grid does not match the protocol")
    if tuple(float(row["sigma"]) for row in posterior) != SIGMA_GRID:
        raise ValueError("Posterior sigma grid does not match the protocol")
    values = np.asarray(
        [
            [float(row[key]) for key in ("estimate", "ci95_low", "ci95_high")]
            for row in [*coverage, *posterior]
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Curve files contain nonfinite values")
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("Curve files contain values outside [0,1]")
    thresholds = load_config(CONFIG_PATH)["thresholds"]
    high_high = [
        float(coverage_row["sigma"])
        for coverage_row, posterior_row in zip(coverage, posterior)
        if float(coverage_row["estimate"]) >= thresholds["coverage"]
        and float(posterior_row["estimate"]) >= thresholds["maximum_posterior_weight"]
    ]
    return {
        "status": "pass",
        "results_dir": str(results_dir.resolve()),
        "rows_per_metric": len(coverage),
        "sampled_high_high_sigmas": high_high,
    }


def _curve_arrays(
    results_dir: Path,
) -> tuple[np.ndarray, list[dict[str, str]], list[dict[str, str]]]:
    """Read geometry curves from one explicit result directory."""
    coverage = read_curve_csv(results_dir / "coverage_curve.csv")
    posterior = read_curve_csv(results_dir / "max_posterior_weight_curve.csv")
    sigma = np.asarray([float(row["sigma"]) for row in coverage])
    return sigma, coverage, posterior


def plot_geometry(results_dir: Path, output_path: Path) -> None:
    """Plot coverage and posterior concentration from one result directory."""
    sigma, coverage, posterior = _curve_arrays(results_dir)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    for rows, label, color, marker in (
        (coverage, r"Coverage $C_\sigma(p, D)$", "#d97706", "s"),
        (posterior, r"Max posterior weight $W_\sigma(D)$", "#0f766e", "o"),
    ):
        estimate = np.asarray([float(row["estimate"]) for row in rows])
        low = np.asarray([float(row["ci95_low"]) for row in rows])
        high = np.asarray([float(row["ci95_high"]) for row in rows])
        ax.plot(sigma, estimate, color=color, marker=marker, label=label)
        ax.fill_between(sigma, low, high, color=color, alpha=0.16)
    ax.axvspan(
        2.0,
        5.0,
        color="#b91c1c",
        alpha=0.10,
        label=r"clean-room high-high region ($q_C = q_W = 0.8$)",
    )
    for x_position, regime in (
        (0.12, "small noise"),
        (2.9, "medium noise"),
        (25.0, "large noise"),
    ):
        ax.text(
            x_position,
            0.055,
            regime,
            color="#334155",
            fontsize=9,
            ha="center",
        )
    ax.set_xscale("log")
    ax.set_xlim(sigma.min(), sigma.max())
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"Noise scale $\sigma$ (small to large)")
    ax.set_ylabel("Probability / concentration")
    ax.set_title("Clean-room reproduction of the paper's full-space geometry")
    ax.grid(alpha=0.22, which="both")
    ax.legend(loc="center right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_comparison(results_dir: Path, output_path: Path) -> None:
    """Plot paper geometry and the separate E005 transitions on one sigma axis."""
    sigma, coverage, posterior = _curve_arrays(results_dir)
    transitions = json.loads(
        (
            REPO_ROOT
            / "results"
            / "experiment_05"
            / "experiment_05_transition_windows.json"
        ).read_text()
    )
    low = transitions["transitions"]["low_frequency_residual"]["4"]
    high = transitions["transitions"]["high_frequency_residual"]["4"]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 8.2),
        constrained_layout=True,
        sharex=True,
    )
    for rows, label, color, marker in (
        (coverage, r"Coverage $C_\sigma(p, D)$", "#d97706", "s"),
        (posterior, r"Max posterior weight $W_\sigma(D)$", "#0f766e", "o"),
    ):
        estimate = np.asarray([float(row["estimate"]) for row in rows])
        axes[0].plot(sigma, estimate, color=color, marker=marker, label=label)
    axes[0].set_ylabel("Probability / concentration")
    axes[0].set_title("A. Paper geometry (clean-room full-space reproduction)")

    axes[1].plot(
        E005_SIGMA_GRID,
        low["normalized_recovery"],
        color="#2563eb",
        marker="o",
        label="Low-frequency residual recovery",
    )
    axes[1].plot(
        E005_SIGMA_GRID,
        high["normalized_recovery"],
        color="#be123c",
        marker="s",
        label="High-frequency residual recovery",
    )
    axes[1].axvspan(
        low["exit_sigma"],
        low["entry_sigma"],
        color="#2563eb",
        alpha=0.08,
        label="Low spectral transition",
    )
    axes[1].axvspan(
        high["exit_sigma"],
        high["entry_sigma"],
        color="#be123c",
        alpha=0.08,
        label="High spectral transition",
    )
    axes[1].set_ylabel("Normalized residual recovery")
    axes[1].set_xlabel(r"Noise scale $\sigma$ (small to large)")
    axes[1].set_title("B. Separate E005 spectral residual transitions")
    for axis in axes:
        axis.axvspan(
            2.0,
            5.0,
            color="#b91c1c",
            alpha=0.10,
            label=r"clean-room high-high region ($q_C = q_W = 0.8$)",
        )
        axis.set_xscale("log")
        axis.set_xlim(min(E005_SIGMA_GRID), max(sigma))
        axis.grid(alpha=0.22, which="both")
    axes[0].legend(loc="center right")
    axes[1].legend(loc="center right", fontsize=9)
    figure.suptitle(
        "Paper geometry and spectral residual transitions on a shared sigma axis"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run_plot_only(
    results_dir: Path, geometry_figure: Path, comparison_figure: Path
) -> None:
    """Validate one result directory and regenerate both review figures."""
    validate_results(results_dir)
    plot_geometry(results_dir, geometry_figure)
    plot_comparison(results_dir, comparison_figure)
    print(f"Saved: {geometry_figure}")
    print(f"Saved: {comparison_figure}")


def main() -> None:
    """Execute the requested E004A mode without changing scientific choices."""
    args = parse_args()
    if args.compute_e006_grid:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required with --compute-e006-grid")
        manifest = compute_e006_grid_geometry(args.dataset_root, args.device)
        print(json.dumps(manifest, indent=2))
        return
    mode = "compute" if args.compute else "validate" if args.validate_only else "plot"
    if mode == "compute":
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required with --compute")
        output_dir = (args.output_dir or DEFAULT_REPRODUCTION_DIR).resolve()
        manifest = compute_and_write(
            config_path=args.config,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            device=args.device,
            repository_root=REPO_ROOT,
            command=sys.argv,
        )
        config = load_config(args.config)
        comparison_rows, comparison_summary = compare_reproduction(
            fresh_dir=output_dir,
            committed_dir=COMMITTED_RESULTS_DIR,
            config=config,
        )
        write_comparison(output_dir, comparison_rows, comparison_summary)
        figure_dir = output_dir / "figures"
        run_plot_only(
            output_dir,
            figure_dir / "coverage_and_max_posterior_weight.png",
            figure_dir / "geometry_and_spectral_transitions.png",
        )
        manifest = finalize_manifest(output_dir, comparison_summary)
        print(
            json.dumps(
                {"manifest": manifest, "comparison": comparison_summary}, indent=2
            )
        )
        return

    results_dir = (args.output_dir or COMMITTED_RESULTS_DIR).resolve()
    if mode == "validate":
        print(json.dumps(validate_results(results_dir), indent=2))
        return
    run_plot_only(
        results_dir,
        args.geometry_figure or FIGURES_DIR / "coverage_and_max_posterior_weight.png",
        args.comparison_figure
        or REPO_ROOT
        / "figures"
        / "experiment_05"
        / "geometry_and_spectral_transitions.png",
    )


if __name__ == "__main__":
    main()
