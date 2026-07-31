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

    def test_completed_edm50k_identity_is_declared(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(manifest["model"]["status"], "completed_validated")
        self.assertEqual(
            manifest["model"]["checkpoint_path"],
            "/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/"
            "network-snapshot-040000.pkl",
        )
        self.assertEqual(manifest["model"]["checkpoint_size_bytes"], 223159918)
        self.assertEqual(
            manifest["model"]["checkpoint_sha256"],
            "a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b",
        )
        self.assertEqual(
            manifest["run_artifacts"]["training_state_sha256"],
            "4af61f228ea5ca0f25897ba180e3e8c5466628fecffa039e98d3505d0bbfbcf9",
        )
        self.assertEqual(
            manifest["run_artifacts"]["config_used_sha256"],
            "464576709477f0ff74e12bbd66b8ac8afcb19dfa6f4127add42e3ac0e4efd106",
        )
        self.assertEqual(manifest["training"]["job_id"], 15315560)
        self.assertEqual(manifest["training"]["slurm_state"], "COMPLETED")
        self.assertEqual(manifest["training"]["exit_code"], "0:0")
        self.assertEqual(manifest["training"]["final_kimg"], 40000.0)
        self.assertEqual(manifest["smoke_test"]["result"], "pass")
        self.assertEqual(manifest["smoke_test"]["job_id"], 15425345)

    def test_launcher_uses_explicit_repository_contract(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER_PATH)], check=True)
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertIn('if [ -z "${SLURM_JOB_ID:-}" ]', launcher)
        self.assertIn('if [ -z "${E005_REPO_ROOT:-}" ]', launcher)
        self.assertIn('if [ -z "${E005_REPO_COMMIT:-}" ]', launcher)
        self.assertIn('--repo-root "$REPO_ROOT"', launcher)
        self.assertIn('--expected-repo-commit "$E005_REPO_COMMIT"', launcher)
        self.assertIn('python "$PREFLIGHT"', launcher)
        self.assertIn('echo "repository_root=${REPO_ROOT}"', launcher)
        self.assertIn('--mode "$MODE"', launcher)
        self.assertIn("error: SLURM_JOB_ID is unset", launcher)
        self.assertIn(
            'FALLBACK_TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp"',
            launcher,
        )
        self.assertIn('TMP_ROOT="${FALLBACK_TMP_ROOT}/e005_${SLURM_JOB_ID}"', launcher)
        self.assertIn('echo "slurm_tmpdir_source=${SLURM_TMPDIR_SOURCE}"', launcher)
        self.assertIn('echo "slurm_tmpdir=${SLURM_TMPDIR}"', launcher)
        self.assertIn('echo "slurm_tmpdir_writable=true"', launcher)
        self.assertIn("#SBATCH --time=2-00:00:00", launcher)
        self.assertIn("#SBATCH --gres=gpu:L40S:1", launcher)
        self.assertNotIn("BASH_SOURCE", launcher)
        self.assertNotIn('test -n "${SLURM_TMPDIR:-}"', launcher)
        self.assertNotIn("#SBATCH --gres=gpu:1", launcher)
        self.assertNotIn("#SBATCH --gpus=1", launcher)
        self.assertNotIn("curl ", launcher)
        self.assertNotIn("wget ", launcher)

    def run_launcher_contract_check(
        self,
        *,
        repo_root: str | None,
        repo_commit: str = "0" * 40,
        slurm_job_id: str | None = "unit-test",
    ) -> subprocess.CompletedProcess[str]:
        """Run only the launcher's pre-output repository checks."""
        environment = {
            "PATH": "/usr/bin:/bin",
            "E005_REPO_COMMIT": repo_commit,
        }
        if slurm_job_id is not None:
            environment["SLURM_JOB_ID"] = slurm_job_id
        if repo_root is not None:
            environment["E005_REPO_ROOT"] = repo_root
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER_PATH),
                "configs/e005_edm50k_matched_40000kimg.yaml",
                "fresh",
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )

    def test_launcher_rejects_missing_job_id_explicitly(self) -> None:
        result = self.run_launcher_contract_check(
            repo_root=str(REPO_ROOT),
            slurm_job_id=None,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error: SLURM_JOB_ID is unset", result.stderr)

    def test_launcher_rejects_absent_repository_root(self) -> None:
        result = self.run_launcher_contract_check(repo_root=None)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("E005_REPO_ROOT is required", result.stderr)

    def test_launcher_rejects_relative_repository_root(self) -> None:
        result = self.run_launcher_contract_check(repo_root="relative/checkout")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("E005_REPO_ROOT must be absolute", result.stderr)

    def test_launcher_rejects_incorrect_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self.run_launcher_contract_check(repo_root=temporary_directory)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Required repository path is missing", result.stderr)

    def test_repository_validation_rejects_wrong_commit(self) -> None:
        current_commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        with self.assertRaises(ValueError):
            PREFLIGHT.validate_repository(REPO_ROOT, f"{current_commit}wrong")

    def run_launcher_tmpdir_dry_run(
        self,
        *,
        slurm_tmpdir: Path | None,
        fallback_root: Path,
        preflight_exit: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        """Exercise launcher temporary setup without invoking training."""
        fake_bin = fallback_root.parent / "fake-bin"
        fake_bin.mkdir()
        fake_python = fake_bin / "python"
        fake_python.write_text(
            f"#!/bin/sh\nexit {preflight_exit}\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        environment = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SLURM_JOB_ID": "987654",
            "E005_REPO_ROOT": str(REPO_ROOT),
            "E005_REPO_COMMIT": "dry-run-commit",
            "E005_DRY_RUN_TMP_ROOT": str(fallback_root),
        }
        if slurm_tmpdir is not None:
            environment["SLURM_TMPDIR"] = str(slurm_tmpdir)
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER_PATH),
                "configs/e005_edm50k_matched_40000kimg.yaml",
                "fresh",
                "--launcher-dry-run",
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )

    def test_launcher_preserves_writable_provided_slurm_tmpdir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provided = root / "provided"
            provided.mkdir()
            result = self.run_launcher_tmpdir_dry_run(
                slurm_tmpdir=provided,
                fallback_root=root / "fallback",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("slurm_tmpdir_source=provided", result.stdout)
            self.assertIn(f"slurm_tmpdir={provided.resolve()}", result.stdout)
            self.assertIn("launcher_dry_run=pass", result.stdout)

    def test_launcher_uses_job_specific_fallback_when_tmpdir_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fallback_root = Path(temporary_directory) / "fallback"
            result = self.run_launcher_tmpdir_dry_run(
                slurm_tmpdir=None,
                fallback_root=fallback_root,
            )
            expected = fallback_root / "e005_987654"

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(expected.is_dir())
            self.assertIn("slurm_tmpdir_source=fallback", result.stdout)
            self.assertIn(f"slurm_tmpdir={expected.resolve()}", result.stdout)

    def test_launcher_rejects_unwritable_provided_tmpdir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provided = root / "unwritable"
            provided.mkdir(mode=0o500)
            try:
                result = self.run_launcher_tmpdir_dry_run(
                    slurm_tmpdir=provided,
                    fallback_root=root / "fallback",
                )
            finally:
                provided.chmod(0o700)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error: SLURM_TMPDIR is not writable", result.stderr)

    def test_failed_preflight_creates_no_tmpdir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fallback_root = Path(temporary_directory) / "fallback"
            result = self.run_launcher_tmpdir_dry_run(
                slurm_tmpdir=None,
                fallback_root=fallback_root,
                preflight_exit=1,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(fallback_root.exists())

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
