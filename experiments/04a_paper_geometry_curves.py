#!/usr/bin/env python3
"""Compute, validate, or plot the E004A clean-room paper geometry baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spectral-diffusion-e004a-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from spectral_diffusion_playground.paper_geometry import SIGMA_GRID
from spectral_diffusion_playground.paper_geometry_evaluation import (
    compute_and_write,
    compare_reproduction,
    finalize_manifest,
    load_config,
    read_curve_csv,
    write_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_RESULTS_DIR = REPO_ROOT / "results" / "experiment_04a"
DEFAULT_REPRODUCTION_DIR = REPO_ROOT / "results" / "experiment_04a_reproduction"
CONFIG_PATH = REPO_ROOT / "configs" / "e004a_local_geometry.json"
FIGURES_DIR = REPO_ROOT / "figures" / "experiment_04a"

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
