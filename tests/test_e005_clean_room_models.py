"""Tests for the frozen E005 clean-room model-pair preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e005_edm50k_matched_40000kimg.yaml"
MANIFEST_PATH = REPO_ROOT / "configs/e005_edm50k_matched_40000kimg_manifest.json"
SUBSET_PATH = REPO_ROOT / "data/e005_cifar10_subset_1k_indices.txt"
LAUNCHER_PATH = REPO_ROOT / "scripts/e005_train_edm50k_matched.slurm"
PREFLIGHT_PATH = REPO_ROOT / "scripts/e005_preflight.py"

SPEC = importlib.util.spec_from_file_location("e005_preflight", PREFLIGHT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load E005 preflight module")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)

ANCHOR_CONFIG = {
    **PREFLIGHT.EXPECTED_CONFIG,
    "experiment": {
        **PREFLIGHT.EXPECTED_CONFIG["experiment"],
        "name": "exp_004_standard_edm_n1000_40000kimg_20260415",
    },
    "dataset": {
        **PREFLIGHT.EXPECTED_CONFIG["dataset"],
        "subset_size": 1000,
    },
}


class E005CleanRoomModelTest(unittest.TestCase):
    """Validate the subset, matched config, and guarded launcher."""

    def test_subset_indices_and_hashes_are_frozen(self) -> None:
        values = np.asarray(
            [int(line) for line in SUBSET_PATH.read_text().splitlines()],
            dtype=np.int64,
        )

        self.assertEqual(values.size, 1000)
        self.assertEqual(np.unique(values).size, 1000)
        self.assertTrue(np.all((0 <= values) & (values < 50000)))
        self.assertTrue(np.all(values[:-1] < values[1:]))
        self.assertEqual(
            hashlib.sha256(values.astype("<i8").tobytes()).hexdigest(),
            "f97076ea6db59a96dc81a59d1b573bc8aaecdb8efa1e93c0d79928bfbf8a43f8",
        )
        self.assertEqual(
            hashlib.sha256(SUBSET_PATH.read_bytes()).hexdigest(),
            "33bb509c48144464a48d3b945cc44c14f880a1e6c6470c283dc0ed65e22b1f29",
        )

    def test_subset_derivation_is_reproducible(self) -> None:
        expected = np.arange(50000, dtype=np.int64)
        np.random.RandomState(0).shuffle(expected)
        expected = np.sort(expected[:1000])
        observed = np.loadtxt(SUBSET_PATH, dtype=np.int64)

        np.testing.assert_array_equal(observed, expected)

    def test_matched_config_differs_only_in_operational_name_and_size(self) -> None:
        candidate = PREFLIGHT.validate_config(CONFIG_PATH)
        differences = {
            ("experiment", "name"),
            ("dataset", "subset_size"),
        }
        observed_differences = {
            (section, key)
            for section in ANCHOR_CONFIG
            for key in ANCHOR_CONFIG[section]
            if ANCHOR_CONFIG[section][key] != candidate[section][key]
        }

        self.assertEqual(observed_differences, differences)

    def test_archive_and_source_hashes_are_declared(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["archive"]["sha256"],
            PREFLIGHT.EXPECTED_ARCHIVE_SHA256,
        )
        self.assertEqual(
            manifest["source"]["edm_commit"],
            PREFLIGHT.EXPECTED_EDM_COMMIT,
        )
        self.assertEqual(
            manifest["source"]["wrapper_sha256"],
            PREFLIGHT.EXPECTED_WRAPPER_SHA256,
        )
        PREFLIGHT.validate_manifest(CONFIG_PATH, MANIFEST_PATH)

    def test_launcher_has_valid_shell_syntax_and_safety_guards(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER_PATH)], check=True)
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertIn('if [ "${SLURM_JOB_ID:-}" = "" ]', launcher)
        self.assertIn('--mode "$MODE"', launcher)
        self.assertIn("Refusing to train outside Slurm", launcher)
        self.assertNotIn("curl ", launcher)
        self.assertNotIn("wget ", launcher)

    def test_fresh_mode_rejects_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = deepcopy(PREFLIGHT.EXPECTED_CONFIG)
            root = Path(temporary_directory)
            config["experiment"]["persistent_scratch_root"] = str(root / "scratch")
            config["experiment"]["persistent_data_root"] = str(root / "data")
            output = (
                Path(config["experiment"]["persistent_data_root"])
                / config["experiment"]["name"]
            )
            output.mkdir(parents=True)
            (output / "existing.txt").write_text("occupied", encoding="utf-8")

            with self.assertRaises(ValueError):
                PREFLIGHT.validate_output_state(config, "fresh")

    def test_resume_mode_requires_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = deepcopy(PREFLIGHT.EXPECTED_CONFIG)
            root = Path(temporary_directory)
            config["experiment"]["persistent_scratch_root"] = str(root / "scratch")
            config["experiment"]["persistent_data_root"] = str(root / "data")

            with self.assertRaises(ValueError):
                PREFLIGHT.validate_output_state(config, "resume")


if __name__ == "__main__":
    unittest.main()
