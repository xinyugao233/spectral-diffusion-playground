"""Tests for independent E004A computation, comparison, and plotting."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.paper_geometry_evaluation import (
    COMPARISON_FIELDS,
    CURVE_FIELDS,
    compare_reproduction,
    compute_and_write,
    evaluate_geometry,
    index_text_sha256,
    load_config,
    sha256_file,
    subset_manifest_sha256,
    write_comparison,
)


def _load_experiment_module() -> object:
    """Load the numeric-prefixed E004A entry point for direct mode tests."""
    path = REPO_ROOT / "experiments" / "04a_paper_geometry_curves.py"
    experiments_root = str(path.parent)
    if experiments_root not in sys.path:
        sys.path.insert(0, experiments_root)
    spec = importlib.util.spec_from_file_location("experiment_04a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load E004A experiment module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_batch(path: Path, values: np.ndarray) -> None:
    """Write a minimal canonical-shape CIFAR Python batch."""
    with path.open("wb") as handle:
        pickle.dump({b"data": values}, handle, protocol=2)


class PaperGeometryEvaluationTest(unittest.TestCase):
    """Validate the independently runnable clean-room evaluator."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(31)

    def test_frozen_config_records_exact_indices_and_hashes(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "e004a_local_geometry.json")
        dataset = config["dataset"]
        self.assertEqual(dataset["training_subset_indices"], list(range(1000)))
        self.assertEqual(dataset["test_subset_indices"], list(range(1000)))
        self.assertEqual(
            index_text_sha256(dataset["training_subset_indices"]),
            dataset["training_index_text_sha256"],
        )
        self.assertEqual(
            subset_manifest_sha256(
                dataset["training_subset_indices"],
                dataset["test_subset_indices"],
            ),
            "39cdd611c46acdedfeb2a9eff47187276afd83750dfdbeedcad04066e127893a",
        )

    def test_evaluation_is_seed_reproducible_and_batching_invariant(self) -> None:
        training = self.rng.normal(size=(7, 12))
        test = self.rng.normal(size=(5, 12))
        arguments = {
            "sigmas": [0.2, 0.8, 2.0],
            "shell_c": 5.0,
            "posterior_draws": 2,
            "coverage_draws": 3,
            "bootstrap_replicates": 20,
            "seed": 9,
        }
        first, first_validation = evaluate_geometry(
            training,
            test,
            query_batch_size=2,
            reference_batch_size=3,
            **arguments,
        )
        second, second_validation = evaluate_geometry(
            training,
            test,
            query_batch_size=7,
            reference_batch_size=7,
            **arguments,
        )
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(first_validation["status"], "pass")

    def _make_synthetic_run(
        self, root: Path
    ) -> tuple[Path, Path, Path, dict[str, object]]:
        data_dir = root / "cifar-10-batches-py"
        data_dir.mkdir()
        dataset_hashes: dict[str, str] = {}
        for batch_index in range(1, 6):
            path = data_dir / f"data_batch_{batch_index}"
            values = self.rng.integers(0, 256, size=(2, 3072), dtype=np.uint8)
            _write_fake_batch(path, values)
            dataset_hashes[path.name] = sha256_file(path)
        test_path = data_dir / "test_batch"
        _write_fake_batch(
            test_path,
            self.rng.integers(0, 256, size=(4, 3072), dtype=np.uint8),
        )
        dataset_hashes[test_path.name] = sha256_file(test_path)
        train_indices = [0, 2, 4, 6]
        test_indices = [0, 1, 2]
        config: dict[str, object] = {
            "experiment_id": "E004A-test",
            "reproduction_claim": "synthetic test",
            "dataset": {
                "name": "synthetic CIFAR batches",
                "normalization": "x / 127.5 - 1.0",
                "flattened_dimension": 3072,
                "training_subset_indices": train_indices,
                "test_subset_indices": test_indices,
                "training_index_text_sha256": index_text_sha256(train_indices),
                "test_index_text_sha256": index_text_sha256(test_indices),
                "dataset_file_sha256": dataset_hashes,
            },
            "sigma_grid": [0.2, 1.0],
            "shell_c": 5.0,
            "posterior_corruption_draws": 2,
            "coverage_corruption_draws": 2,
            "bootstrap_replicates": 10,
            "random_seed": 4,
            "query_batch_size": 2,
            "reference_batch_size": 2,
            "thresholds": {
                "coverage": 0.8,
                "maximum_posterior_weight": 0.8,
            },
            "reproduction_tolerance": {
                "policy": "test",
                "z_score": 3.0,
                "committed_se_proxy": "test",
                "absolute_floor": {
                    "gaussian_shell_coverage": 0.01,
                    "maximum_posterior_weight": 0.01,
                },
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output_dir = root / "output"
        return data_dir, config_path, output_dir, config

    def test_end_to_end_compute_writes_stable_artifacts_without_committed_input(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, config_path, output_dir, _ = self._make_synthetic_run(root)
            with mock.patch(
                "spectral_diffusion_playground.paper_geometry_evaluation.read_curve_csv",
                side_effect=AssertionError("compute must not read committed curves"),
            ):
                manifest = compute_and_write(
                    config_path=config_path,
                    dataset_root=data_dir,
                    output_dir=output_dir,
                    device="auto",
                    repository_root=REPO_ROOT,
                    command=["synthetic-test"],
                    require_frozen_counts=False,
                )
            self.assertEqual(manifest["device_resolved"], "cpu")
            self.assertEqual(manifest["execution_mode"], "independent_local_compute")
            required_manifest = {
                "config_sha256",
                "dataset_file_sha256",
                "normalization",
                "subset_sha256",
                "sigma_grid",
                "random_seed",
                "runtime_seconds",
                "peak_rss_megabytes",
                "sampled_high_high_sigmas",
            }
            self.assertTrue(required_manifest.issubset(manifest))
            for filename in (
                "coverage_curve.csv",
                "max_posterior_weight_curve.csv",
                "geometry_validation.json",
                "geometry_manifest.json",
            ):
                self.assertTrue((output_dir / filename).is_file())
            with (output_dir / "coverage_curve.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), CURVE_FIELDS)
                self.assertEqual(len(list(reader)), 2)

    def test_same_seed_produces_identical_curve_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, config_path, output_dir, _ = self._make_synthetic_run(root)
            second_output = root / "second-output"
            for destination in (output_dir, second_output):
                compute_and_write(
                    config_path=config_path,
                    dataset_root=data_dir,
                    output_dir=destination,
                    device="cpu",
                    repository_root=REPO_ROOT,
                    command=["synthetic-test"],
                    require_frozen_counts=False,
                )
            for filename in ("coverage_curve.csv", "max_posterior_weight_curve.csv"):
                self.assertEqual(
                    (output_dir / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                )

    def test_comparison_logic_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh = root / "fresh"
            committed = root / "committed"
            fresh.mkdir()
            committed.mkdir()
            for metric, filename in (
                ("gaussian_shell_coverage", "coverage_curve.csv"),
                ("maximum_posterior_weight", "max_posterior_weight_curve.csv"),
            ):
                with (fresh / filename).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sigma_index": 0,
                            "sigma": 2.0,
                            "metric": metric,
                            "estimate": 0.82,
                            "ci95_low": 0.8,
                            "ci95_high": 0.84,
                            "monte_carlo_se": 0.01,
                            "training_examples": 4,
                            "query_examples": 4,
                            "shell_c": 5.0,
                            "subset_sha256": "test",
                            "seed": 0,
                            "status": "ok",
                        }
                    )
                committed_fields = [
                    field for field in CURVE_FIELDS if field != "monte_carlo_se"
                ]
                with (committed / filename).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=committed_fields)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sigma_index": 0,
                            "sigma": 2.0,
                            "metric": metric,
                            "estimate": 0.81,
                            "ci95_low": 0.79,
                            "ci95_high": 0.83,
                            "training_examples": 4,
                            "query_examples": 4,
                            "shell_c": 5.0,
                            "subset_sha256": "test",
                            "seed": 0,
                            "status": "ok",
                        }
                    )
            config = {
                "thresholds": {
                    "coverage": 0.8,
                    "maximum_posterior_weight": 0.8,
                },
                "reproduction_tolerance": {
                    "z_score": 3.0,
                    "absolute_floor": {
                        "gaussian_shell_coverage": 0.01,
                        "maximum_posterior_weight": 0.01,
                    },
                },
            }
            rows, summary = compare_reproduction(
                fresh_dir=fresh,
                committed_dir=committed,
                config=config,
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["within_tolerance"] for row in rows))
            self.assertEqual(summary["status"], "pass")
            write_comparison(fresh, rows, summary)
            with (fresh / "reproduction_comparison.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), COMPARISON_FIELDS)

    def test_plot_only_mode_uses_explicit_curve_directory(self) -> None:
        module = _load_experiment_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geometry.png"
            comparison = root / "comparison.png"
            module.run_plot_only(
                REPO_ROOT / "results" / "experiment_04a",
                geometry,
                comparison,
            )
            self.assertGreater(geometry.stat().st_size, 0)
            self.assertGreater(comparison.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
