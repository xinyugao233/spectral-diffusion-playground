#!/usr/bin/env python3
"""Inspect the committed clean-room paper-geometry baseline.

The validated 1K/1K evaluation is committed as compact CSV/JSON artifacts.
This entry point verifies those artifacts and regenerates the primary figure;
it does not claim access to the paper's unavailable executed code or subset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from spectral_diffusion_playground.paper_geometry import SIGMA_GRID

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "experiment_04a"
FIGURES_DIR = REPO_ROOT / "figures" / "experiment_04a"


def parse_args() -> argparse.Namespace:
    """Parse output-only reproduction arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FIGURES_DIR / "coverage_and_max_posterior_weight.png",
    )
    parser.add_argument(
        "--comparison-output-path",
        type=Path,
        default=REPO_ROOT
        / "figures"
        / "experiment_05"
        / "geometry_and_spectral_transitions.png",
    )
    return parser.parse_args()


def read_curve(path: Path) -> list[dict[str, str]]:
    """Read one committed geometry curve and require the stable schema."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "sigma_index",
        "sigma",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
        "training_examples",
        "query_examples",
        "shell_c",
        "subset_sha256",
        "seed",
        "status",
    }
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"Unexpected curve schema: {path}")
    return rows


def main() -> None:
    """Validate compact results and regenerate the clean-room geometry figure."""
    args = parse_args()
    coverage = read_curve(RESULTS_DIR / "coverage_curve.csv")
    posterior = read_curve(RESULTS_DIR / "max_posterior_weight_curve.csv")
    manifest = json.loads((RESULTS_DIR / "geometry_manifest.json").read_text())
    if manifest["reproduction_claim"] != "paper-derived clean-room reproduction":
        raise ValueError("Unexpected reproduction claim")

    sigma = np.asarray([float(row["sigma"]) for row in coverage])
    if tuple(sigma) != SIGMA_GRID:
        raise ValueError("Committed sigma grid does not match package constant")

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
        2.0, 5.0, color="#b91c1c", alpha=0.10, label="paper-guided high-high region"
    )
    ax.set_xscale("log")
    ax.set_xlim(sigma.min(), sigma.max())
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"Noise scale $\sigma$ (small to large)")
    ax.set_ylabel("Probability / concentration")
    ax.set_title("Clean-room reproduction of the paper's full-space geometry")
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
    ax.grid(alpha=0.22, which="both")
    ax.legend(loc="center right")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {args.output_path}")

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
    e005_sigma = np.asarray(
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
    comparison, axes = plt.subplots(
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
        e005_sigma,
        low["normalized_recovery"],
        color="#2563eb",
        marker="o",
        label="Low-frequency residual recovery",
    )
    axes[1].plot(
        e005_sigma,
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
            label="paper-guided high-high region",
        )
        axis.set_xscale("log")
        axis.set_xlim(min(e005_sigma), max(sigma))
        axis.grid(alpha=0.22, which="both")
    axes[0].legend(loc="center right")
    axes[1].legend(loc="center right", fontsize=9)
    comparison.suptitle(
        "Paper geometry and spectral residual transitions on a shared sigma axis"
    )
    args.comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.savefig(args.comparison_output_path, dpi=220)
    plt.close(comparison)
    print(f"Saved: {args.comparison_output_path}")


if __name__ == "__main__":
    main()
