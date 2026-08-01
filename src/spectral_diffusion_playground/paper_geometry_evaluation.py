"""End-to-end clean-room evaluation for the paper's full-space geometry."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .paper_geometry import (
    SHELL_C,
    gaussian_shell_radii,
    normalized_weights_from_logits,
    squared_distances,
    validate_sigma_grid,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CURVE_FIELDS = (
    "sigma_index",
    "sigma",
    "metric",
    "estimate",
    "ci95_low",
    "ci95_high",
    "monte_carlo_se",
    "training_examples",
    "query_examples",
    "shell_c",
    "subset_sha256",
    "seed",
    "status",
)

COMPARISON_FIELDS = (
    "sigma_index",
    "sigma",
    "metric",
    "committed_estimate",
    "fresh_estimate",
    "absolute_difference",
    "monte_carlo_se",
    "committed_se_proxy",
    "reproduction_tolerance",
    "within_tolerance",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_text_sha256(indices: Sequence[int]) -> str:
    """Hash a canonical newline-delimited integer index list."""
    payload = "".join(f"{int(index)}\n" for index in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def subset_manifest_sha256(
    training_indices: Sequence[int], test_indices: Sequence[int]
) -> str:
    """Hash the public split-prefixed subset identity used by E004A."""
    payload = "".join(
        [
            *(f"train,{int(index)}\n" for index in training_indices),
            *(f"test,{int(index)}\n" for index in test_indices),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path, *, require_frozen_counts: bool = True) -> dict[str, Any]:
    """Load and validate the frozen local-evaluation configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    train_indices = tuple(int(value) for value in dataset["training_subset_indices"])
    test_indices = tuple(int(value) for value in dataset["test_subset_indices"])
    expected_train_count = 1000 if require_frozen_counts else len(train_indices)
    expected_test_count = 1000 if require_frozen_counts else len(test_indices)
    if len(train_indices) != expected_train_count or len(set(train_indices)) != len(
        train_indices
    ):
        raise ValueError("The training subset must contain unique indices")
    if len(test_indices) != expected_test_count or len(set(test_indices)) != len(
        test_indices
    ):
        raise ValueError("The test subset must contain unique indices")
    if index_text_sha256(train_indices) != dataset["training_index_text_sha256"]:
        raise ValueError("Training-index hash mismatch")
    if index_text_sha256(test_indices) != dataset["test_index_text_sha256"]:
        raise ValueError("Test-index hash mismatch")
    validate_sigma_grid(config["sigma_grid"])
    if float(config["shell_c"]) != SHELL_C:
        raise ValueError("Unexpected shell constant")
    for key in (
        "posterior_corruption_draws",
        "coverage_corruption_draws",
        "bootstrap_replicates",
        "query_batch_size",
        "reference_batch_size",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return config


def resolve_device(requested: str) -> str:
    """Resolve the faithful float64 NumPy reference backend."""
    normalized = requested.lower()
    if normalized not in {"auto", "cpu"}:
        raise ValueError(
            "The reference evaluator supports auto/cpu only; CUDA or MPS acceleration "
            "must be independently validated against this float64 oracle"
        )
    return "cpu"


def _resolve_cifar_directory(dataset_root: Path) -> Path:
    """Resolve either a CIFAR batch directory or its parent directory."""
    root = dataset_root.expanduser().resolve()
    direct = root / "data_batch_1"
    nested = root / "cifar-10-batches-py" / "data_batch_1"
    if direct.is_file():
        return root
    if nested.is_file():
        return root / "cifar-10-batches-py"
    raise FileNotFoundError(f"Could not find CIFAR-10 Python batches under {root}")


def _load_cifar_batch(path: Path) -> NDArray[np.uint8]:
    """Load one canonical CIFAR-10 Python batch without preprocessing."""
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != 3072:
        raise ValueError(f"Unexpected CIFAR-10 array shape in {path}: {data.shape}")
    return data


def load_cifar10_subsets(
    dataset_root: Path,
    training_indices: Sequence[int],
    test_indices: Sequence[int],
    expected_hashes: Mapping[str, str],
) -> tuple[FloatArray, FloatArray, dict[str, str]]:
    """Load, hash, select, normalize, and flatten the frozen CIFAR-10 subsets."""
    data_dir = _resolve_cifar_directory(dataset_root)
    paths = [data_dir / f"data_batch_{index}" for index in range(1, 6)]
    paths.append(data_dir / "test_batch")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-10 files: {missing}")
    observed_hashes = {path.name: sha256_file(path) for path in paths}
    for filename, expected in expected_hashes.items():
        if observed_hashes.get(filename) != expected:
            raise ValueError(f"CIFAR-10 hash mismatch: {filename}")

    train_raw = np.concatenate([_load_cifar_batch(path) for path in paths[:5]])
    test_raw = _load_cifar_batch(paths[-1])
    train_ids = np.asarray(training_indices, dtype=np.int64)
    test_ids = np.asarray(test_indices, dtype=np.int64)
    if np.any(train_ids < 0) or np.any(train_ids >= len(train_raw)):
        raise ValueError("Training indices exceed the CIFAR-10 split")
    if np.any(test_ids < 0) or np.any(test_ids >= len(test_raw)):
        raise ValueError("Test indices exceed the CIFAR-10 split")
    train = train_raw[train_ids].astype(np.float64) / 127.5 - 1.0
    test = test_raw[test_ids].astype(np.float64) / 127.5 - 1.0
    return train, test, observed_hashes


def _cross_term(
    noise: FloatArray, queries: FloatArray, references: FloatArray
) -> FloatArray:
    """Return the cross term in the exact noisy-distance expansion."""
    return np.sum(noise * queries, axis=1)[:, None] - noise @ references.T


def _noise_terms(
    rng: np.random.Generator,
    queries: FloatArray,
    references: FloatArray,
    draws: int,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Generate source-compatible float32 noise and cache scalar distance terms."""
    cross_terms: list[FloatArray] = []
    noise_energies: list[FloatArray] = []
    for _ in range(draws):
        noise = rng.standard_normal(queries.shape, dtype=np.float32).astype(np.float64)
        cross_terms.append(_cross_term(noise, queries, references))
        noise_energies.append(np.square(noise).sum(axis=1))
    return cross_terms, noise_energies


def _posterior_draw_values(
    base_distances: FloatArray,
    cross_term: FloatArray,
    noise_energy: FloatArray,
    sigma: float,
    query_batch_size: int,
) -> tuple[FloatArray, float]:
    """Evaluate one posterior corruption draw in query batches."""
    maxima = np.empty(len(base_distances), dtype=np.float64)
    maximum_error = 0.0
    for start in range(0, len(base_distances), query_batch_size):
        stop = min(start + query_batch_size, len(base_distances))
        distance_sq = np.maximum(
            base_distances[start:stop]
            + 2.0 * sigma * cross_term[start:stop]
            + sigma * sigma * noise_energy[start:stop, None],
            0.0,
        )
        weights = normalized_weights_from_logits(-distance_sq / (2.0 * sigma * sigma))
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
        )
        maxima[start:stop] = weights.max(axis=1)
    return maxima, maximum_error


def _coverage_draw_values(
    base_distances: FloatArray,
    cross_term: FloatArray,
    noise_energy: FloatArray,
    sigma: float,
    dimension: int,
    shell_c: float,
    query_batch_size: int,
    reference_batch_size: int,
) -> BoolArray:
    """Evaluate exact shell-union membership without a query-reference-feature tensor."""
    inner, outer = gaussian_shell_radii(dimension, shell_c)
    lower_sq = (sigma * inner) ** 2
    upper_sq = (sigma * outer) ** 2
    covered = np.zeros(len(base_distances), dtype=bool)
    for query_start in range(0, len(base_distances), query_batch_size):
        query_stop = min(query_start + query_batch_size, len(base_distances))
        block_covered = np.zeros(query_stop - query_start, dtype=bool)
        for reference_start in range(0, base_distances.shape[1], reference_batch_size):
            reference_stop = min(
                reference_start + reference_batch_size,
                base_distances.shape[1],
            )
            distance_sq = np.maximum(
                base_distances[query_start:query_stop, reference_start:reference_stop]
                + 2.0
                * sigma
                * cross_term[query_start:query_stop, reference_start:reference_stop]
                + sigma * sigma * noise_energy[query_start:query_stop, None],
                0.0,
            )
            block_covered |= np.any(
                (distance_sq >= lower_sq) & (distance_sq <= upper_sq),
                axis=1,
            )
        covered[query_start:query_stop] = block_covered
    return covered


def hierarchical_bootstrap_interval(
    values: FloatArray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Bootstrap means by resampling corruption draws and examples."""
    sigma_count, draw_count, example_count = values.shape
    estimates = np.empty((replicates, sigma_count), dtype=np.float64)
    for replicate in range(replicates):
        draw_ids = rng.integers(0, draw_count, size=draw_count)
        example_ids = rng.integers(0, example_count, size=example_count)
        estimates[replicate] = values[:, draw_ids][:, :, example_ids].mean(axis=(1, 2))
    return (
        np.quantile(estimates, 0.025, axis=0),
        np.quantile(estimates, 0.975, axis=0),
    )


def monte_carlo_standard_error(values: FloatArray) -> FloatArray:
    """Estimate Monte Carlo SE from independent corruption-draw means."""
    draw_means = values.mean(axis=2)
    if draw_means.shape[1] < 2:
        return np.zeros(draw_means.shape[0], dtype=np.float64)
    return draw_means.std(axis=1, ddof=1) / math.sqrt(draw_means.shape[1])


def evaluate_geometry(
    training: FloatArray,
    test: FloatArray,
    *,
    sigmas: Sequence[float],
    shell_c: float,
    posterior_draws: int,
    coverage_draws: int,
    bootstrap_replicates: int,
    seed: int,
    query_batch_size: int,
    reference_batch_size: int,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Compute fresh coverage and posterior curves from images and random noise."""
    train = np.asarray(training, dtype=np.float64)
    held_out = np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or held_out.ndim != 2 or train.shape[1] != held_out.shape[1]:
        raise ValueError("training and test must be flattened matrices of equal width")
    sigma_grid = validate_sigma_grid(sigmas)
    rng = np.random.default_rng(seed)
    train_distances = squared_distances(train, train)
    test_distances = squared_distances(held_out, train)
    posterior_cross, posterior_energy = _noise_terms(rng, train, train, posterior_draws)
    coverage_cross, coverage_energy = _noise_terms(rng, held_out, train, coverage_draws)

    posterior_raw = np.empty(
        (len(sigma_grid), posterior_draws, len(train)), dtype=np.float64
    )
    coverage_raw = np.empty(
        (len(sigma_grid), coverage_draws, len(held_out)), dtype=np.float64
    )
    normalization_error = 0.0
    for sigma_index, sigma in enumerate(sigma_grid):
        for draw_index, (cross, energy) in enumerate(
            zip(posterior_cross, posterior_energy)
        ):
            maxima, error = _posterior_draw_values(
                train_distances,
                cross,
                energy,
                sigma,
                query_batch_size,
            )
            posterior_raw[sigma_index, draw_index] = maxima
            normalization_error = max(normalization_error, error)
        for draw_index, (cross, energy) in enumerate(
            zip(coverage_cross, coverage_energy)
        ):
            coverage_raw[sigma_index, draw_index] = _coverage_draw_values(
                test_distances,
                cross,
                energy,
                sigma,
                train.shape[1],
                shell_c,
                query_batch_size,
                reference_batch_size,
            )

    bootstrap_rng = np.random.default_rng(seed + 100_000 + 22)
    posterior_low, posterior_high = hierarchical_bootstrap_interval(
        posterior_raw,
        bootstrap_replicates,
        bootstrap_rng,
    )
    coverage_low, coverage_high = hierarchical_bootstrap_interval(
        coverage_raw,
        bootstrap_replicates,
        bootstrap_rng,
    )
    curves = {
        "maximum_posterior_weight": posterior_raw.mean(axis=(1, 2)),
        "maximum_posterior_weight_ci95_low": posterior_low,
        "maximum_posterior_weight_ci95_high": posterior_high,
        "maximum_posterior_weight_monte_carlo_se": monte_carlo_standard_error(
            posterior_raw
        ),
        "gaussian_shell_coverage": coverage_raw.mean(axis=(1, 2)),
        "gaussian_shell_coverage_ci95_low": coverage_low,
        "gaussian_shell_coverage_ci95_high": coverage_high,
        "gaussian_shell_coverage_monte_carlo_se": monte_carlo_standard_error(
            coverage_raw
        ),
    }
    validation = {
        "posterior_normalization_error_max": normalization_error,
        "posterior_values_in_unit_interval": bool(
            np.all((posterior_raw >= 0.0) & (posterior_raw <= 1.0))
        ),
        "coverage_values_in_unit_interval": bool(
            np.all((coverage_raw >= 0.0) & (coverage_raw <= 1.0))
        ),
        "nonfinite_values": int(
            sum(np.count_nonzero(~np.isfinite(value)) for value in curves.values())
        ),
    }
    validation["status"] = (
        "pass"
        if validation["posterior_normalization_error_max"] <= 1e-10
        and validation["posterior_values_in_unit_interval"]
        and validation["coverage_values_in_unit_interval"]
        and validation["nonfinite_values"] == 0
        else "fail"
    )
    return curves, validation


def _curve_rows(
    config: Mapping[str, Any],
    curves: Mapping[str, FloatArray],
    metric: str,
    subset_hash: str,
) -> list[dict[str, Any]]:
    """Convert one computed metric to the stable fresh-curve schema."""
    query_count = (
        len(config["dataset"]["training_subset_indices"])
        if metric == "maximum_posterior_weight"
        else len(config["dataset"]["test_subset_indices"])
    )
    rows = []
    for index, sigma in enumerate(config["sigma_grid"]):
        rows.append(
            {
                "sigma_index": index,
                "sigma": sigma,
                "metric": metric,
                "estimate": float(curves[metric][index]),
                "ci95_low": float(curves[f"{metric}_ci95_low"][index]),
                "ci95_high": float(curves[f"{metric}_ci95_high"][index]),
                "monte_carlo_se": float(curves[f"{metric}_monte_carlo_se"][index]),
                "training_examples": len(config["dataset"]["training_subset_indices"]),
                "query_examples": query_count,
                "shell_c": config["shell_c"],
                "subset_sha256": subset_hash,
                "seed": config["random_seed"],
                "status": "ok",
            }
        )
    return rows


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    """Write deterministic CSV rows with an explicit stable schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_head(repository_root: Path) -> str | None:
    """Return the current Git commit when available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_clean(repository_root: Path) -> bool | None:
    """Report whether the repository worktree is clean when Git is available."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return not bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _peak_rss_megabytes() -> float:
    """Return process peak resident memory in platform-correct MiB."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 * 1024.0)


def compute_and_write(
    *,
    config_path: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str,
    repository_root: Path,
    command: Sequence[str],
    require_frozen_counts: bool = True,
) -> dict[str, Any]:
    """Run the independent numerical evaluation and write fresh artifacts."""
    started = time.perf_counter()
    destination = output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path, require_frozen_counts=require_frozen_counts)
    backend = resolve_device(device)
    dataset = config["dataset"]
    train_indices = dataset["training_subset_indices"]
    test_indices = dataset["test_subset_indices"]
    training, test, dataset_hashes = load_cifar10_subsets(
        dataset_root,
        train_indices,
        test_indices,
        dataset["dataset_file_sha256"],
    )
    curves, validation = evaluate_geometry(
        training,
        test,
        sigmas=config["sigma_grid"],
        shell_c=float(config["shell_c"]),
        posterior_draws=int(config["posterior_corruption_draws"]),
        coverage_draws=int(config["coverage_corruption_draws"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["random_seed"]),
        query_batch_size=int(config["query_batch_size"]),
        reference_batch_size=int(config["reference_batch_size"]),
    )
    subset_hash = subset_manifest_sha256(train_indices, test_indices)
    coverage_rows = _curve_rows(
        config,
        curves,
        "gaussian_shell_coverage",
        subset_hash,
    )
    posterior_rows = _curve_rows(
        config,
        curves,
        "maximum_posterior_weight",
        subset_hash,
    )
    write_csv(destination / "coverage_curve.csv", coverage_rows, CURVE_FIELDS)
    write_csv(
        destination / "max_posterior_weight_curve.csv",
        posterior_rows,
        CURVE_FIELDS,
    )
    validation.update(
        {
            "curve_rows_per_metric": len(config["sigma_grid"]),
            "subset_sha256": subset_hash,
            "config_sha256": sha256_file(config_path),
        }
    )
    (destination / "geometry_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = time.perf_counter() - started
    thresholds = config["thresholds"]
    high_high = [
        float(sigma)
        for sigma, coverage, posterior in zip(
            config["sigma_grid"],
            curves["gaussian_shell_coverage"],
            curves["maximum_posterior_weight"],
        )
        if coverage >= thresholds["coverage"]
        and posterior >= thresholds["maximum_posterior_weight"]
    ]
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "completed" if validation["status"] == "pass" else "failed",
        "reproduction_claim": config["reproduction_claim"],
        "execution_mode": "independent_local_compute",
        "device_requested": device,
        "device_resolved": backend,
        "runtime_seconds": runtime,
        "peak_rss_megabytes": _peak_rss_megabytes(),
        "command": list(command),
        "repository_commit": _git_head(repository_root),
        "repository_worktree_clean": _git_worktree_clean(repository_root),
        "implementation_file_sha256": {
            "experiments/04a_paper_geometry_curves.py": sha256_file(
                repository_root / "experiments" / "04a_paper_geometry_curves.py"
            ),
            "src/spectral_diffusion_playground/paper_geometry.py": sha256_file(
                repository_root
                / "src"
                / "spectral_diffusion_playground"
                / "paper_geometry.py"
            ),
            "src/spectral_diffusion_playground/paper_geometry_evaluation.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(_resolve_cifar_directory(dataset_root)),
        "dataset_file_sha256": dataset_hashes,
        "normalization": dataset["normalization"],
        "flattened_dimension": training.shape[1],
        "training_index_text_sha256": dataset["training_index_text_sha256"],
        "test_index_text_sha256": dataset["test_index_text_sha256"],
        "subset_sha256": subset_hash,
        "training_examples": len(training),
        "test_examples": len(test),
        "sigma_grid": config["sigma_grid"],
        "shell_c": config["shell_c"],
        "posterior_corruption_draws": config["posterior_corruption_draws"],
        "coverage_corruption_draws": config["coverage_corruption_draws"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "bootstrap_scheme": "hierarchical resampling of draws and examples",
        "random_seed": config["random_seed"],
        "query_batch_size": config["query_batch_size"],
        "reference_batch_size": config["reference_batch_size"],
        "thresholds": thresholds,
        "sampled_high_high_sigmas": high_high,
        "reproduction_tolerance": config["reproduction_tolerance"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "result_files": [
            "coverage_curve.csv",
            "max_posterior_weight_curve.csv",
            "geometry_validation.json",
            "geometry_manifest.json",
        ],
    }
    (destination / "geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_curve_csv(path: Path) -> list[dict[str, str]]:
    """Read either committed or freshly computed curve rows."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sigma_index",
        "sigma",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected curve schema: {path}")
    return rows


def compare_reproduction(
    *,
    fresh_dir: Path,
    committed_dir: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare independent estimates under the frozen uncertainty-aware rule."""
    rows: list[dict[str, Any]] = []
    tolerance_config = config["reproduction_tolerance"]
    z_score = float(tolerance_config["z_score"])
    thresholds = config["thresholds"]
    fresh_by_metric: dict[str, list[dict[str, str]]] = {}
    filenames = {
        "gaussian_shell_coverage": "coverage_curve.csv",
        "maximum_posterior_weight": "max_posterior_weight_curve.csv",
    }
    for metric, filename in filenames.items():
        fresh = read_curve_csv(fresh_dir / filename)
        committed = read_curve_csv(committed_dir / filename)
        fresh_by_metric[metric] = fresh
        if len(fresh) != len(committed):
            raise ValueError(f"Curve length mismatch: {metric}")
        for fresh_row, committed_row in zip(fresh, committed):
            if float(fresh_row["sigma"]) != float(committed_row["sigma"]):
                raise ValueError(f"Sigma-grid mismatch: {metric}")
            fresh_estimate = float(fresh_row["estimate"])
            committed_estimate = float(committed_row["estimate"])
            fresh_se = float(fresh_row["monte_carlo_se"])
            committed_se = (
                float(committed_row["ci95_high"]) - float(committed_row["ci95_low"])
            ) / (2.0 * 1.96)
            tolerance = max(
                float(tolerance_config["absolute_floor"][metric]),
                z_score * math.sqrt(fresh_se * fresh_se + committed_se * committed_se),
            )
            difference = abs(fresh_estimate - committed_estimate)
            rows.append(
                {
                    "sigma_index": int(fresh_row["sigma_index"]),
                    "sigma": float(fresh_row["sigma"]),
                    "metric": metric,
                    "committed_estimate": committed_estimate,
                    "fresh_estimate": fresh_estimate,
                    "absolute_difference": difference,
                    "monte_carlo_se": fresh_se,
                    "committed_se_proxy": committed_se,
                    "reproduction_tolerance": tolerance,
                    "within_tolerance": difference <= tolerance,
                }
            )
    fresh_high_high = [
        float(coverage["sigma"])
        for coverage, posterior in zip(
            fresh_by_metric["gaussian_shell_coverage"],
            fresh_by_metric["maximum_posterior_weight"],
        )
        if float(coverage["estimate"]) >= thresholds["coverage"]
        and float(posterior["estimate"]) >= thresholds["maximum_posterior_weight"]
    ]
    per_metric = {
        metric: {
            "all_within_tolerance": all(
                row["within_tolerance"] for row in rows if row["metric"] == metric
            ),
            "maximum_absolute_difference": max(
                row["absolute_difference"] for row in rows if row["metric"] == metric
            ),
        }
        for metric in filenames
    }
    summary = {
        "status": (
            "pass" if all(row["within_tolerance"] for row in rows) else "discrepancy"
        ),
        "tolerance_policy": tolerance_config,
        "metrics": per_metric,
        "fresh_sampled_high_high_sigmas": fresh_high_high,
        "expected_sampled_high_high_sigmas": [2.0, 3.0, 4.0, 5.0],
        "high_high_set_unchanged": fresh_high_high == [2.0, 3.0, 4.0, 5.0],
        "qualitative_three_regime_picture_unchanged": bool(
            float(fresh_by_metric["maximum_posterior_weight"][0]["estimate"]) >= 0.8
            and float(fresh_by_metric["gaussian_shell_coverage"][0]["estimate"]) < 0.2
            and float(fresh_by_metric["maximum_posterior_weight"][-1]["estimate"]) < 0.2
            and float(fresh_by_metric["gaussian_shell_coverage"][-1]["estimate"]) >= 0.8
        ),
    }
    return rows, summary


def write_comparison(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write the per-sigma comparison and compact agreement summary."""
    write_csv(output_dir / "reproduction_comparison.csv", rows, COMPARISON_FIELDS)
    (output_dir / "reproduction_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finalize_manifest(
    output_dir: Path,
    comparison_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Record comparison, figures, and final artifact hashes after plotting."""
    manifest_path = output_dir / "geometry_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reproduction_comparison"] = comparison_summary
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest["artifact_sha256"] = {
        str(path.relative_to(output_dir)): sha256_file(path) for path in artifact_paths
    }
    manifest["result_files"] = list(manifest["artifact_sha256"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
