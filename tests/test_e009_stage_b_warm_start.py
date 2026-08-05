"""Tests for the frozen E009 Stage B warm-start smoke contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spectral_diffusion_playground.e009_warm_start import (
    STATE_SCHEMA_VERSION,
    StatefulInfiniteSampler,
    derive_rng_seeds,
    initialize_rngs,
    seeded_rng_digest,
    validate_extended_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_preflight():
    """Load the Stage B preflight as a test module."""
    path = REPO_ROOT / "scripts/e009_stage_b_preflight.py"
    spec = importlib.util.spec_from_file_location("e009_stage_b_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Stage B preflight")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageBWarmStartTests(unittest.TestCase):
    """Protect the amended Stage B scientific and execution boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = load_preflight()
        cls.protocol = json.loads(
            (REPO_ROOT / "configs/e009_stage_b_protocol.json").read_text()
        )

    def test_protocol_freezes_warm_start_seed_and_parent_hashes(self) -> None:
        amendment = self.protocol["warm_start_amendment"]
        self.assertTrue(amendment["warm_start"])
        self.assertFalse(amendment["exact_continuation"])
        self.assertEqual(amendment["rng_seed"], 1)
        self.assertEqual(amendment["starting_exposure_kimg"], 12000)
        anchor = self.protocol["resume_anchor"]
        self.assertEqual(
            anchor["training_state_sha256"],
            self.preflight.EXPECTED_PARENT_STATE_SHA256,
        )
        self.assertEqual(
            anchor["ema_snapshot_sha256"],
            self.preflight.EXPECTED_PARENT_EMA_SHA256,
        )

    def test_stage_b_configs_match_frozen_contract_and_hashes(self) -> None:
        for filename in self.preflight.APPROVED_CONFIGS:
            path = REPO_ROOT / "configs" / filename
            config = self.preflight.validate_config(REPO_ROOT, path)
            self.assertEqual(config["experiment"]["seed"], 1)
            self.assertTrue(config["experiment"]["warm_start"])
            self.assertEqual(config["training"]["workers"], 0)

    def test_seed_one_initialization_is_exactly_reproducible(self) -> None:
        first_generator, first_seeds = initialize_rngs(1, 0, 1)
        first_digest = seeded_rng_digest(first_generator)
        second_generator, second_seeds = initialize_rngs(1, 0, 1)
        second_digest = seeded_rng_digest(second_generator)
        self.assertEqual(first_seeds, derive_rng_seeds(1, 0, 1))
        self.assertEqual(first_seeds, second_seeds)
        self.assertEqual(first_digest, second_digest)
        self.assertNotEqual(first_seeds["stage_b_seed"], 0)

    def test_stateful_sampler_matches_reference_and_round_trips(self) -> None:
        dataset = list(range(17))
        sampler = StatefulInfiniteSampler(dataset, seed=1)
        iterator = iter(sampler)
        prefix = [next(iterator) for _ in range(23)]

        order = np.arange(len(dataset), dtype=np.int64)
        random_state = np.random.RandomState(1)
        random_state.shuffle(order)
        window = int(np.rint(order.size * 0.5))
        expected = []
        for index in range(23):
            position = index % order.size
            expected.append(int(order[position]))
            swap = (position - random_state.randint(window)) % len(order)
            order[position], order[swap] = order[swap], order[position]
        self.assertEqual(prefix, expected)

        state = sampler.state_dict()
        continuation = [next(iterator) for _ in range(20)]
        restored = StatefulInfiniteSampler(dataset, seed=1)
        restored.load_state_dict(state)
        restored_iterator = iter(restored)
        self.assertEqual(
            continuation,
            [next(restored_iterator) for _ in range(20)],
        )

    def test_extended_state_requires_ema_progress_and_rng(self) -> None:
        module = torch.nn.Linear(2, 2)
        sampler = StatefulInfiniteSampler(list(range(4)), seed=1)
        generator, _ = initialize_rngs(1, 0, 1)
        valid = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "net": module,
            "optimizer_state": {"state": {1: {}}, "param_groups": []},
            "ema": module,
            "progress": {"start_kimg": 12000, "cur_kimg": 13000},
            "rng_state": {
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": [],
                "sampler": sampler.state_dict(),
                "dataloader_generator": generator.get_state(),
                "unavailable": ["dataloader_worker_rng:not_applicable_num_workers_0"],
            },
            "warm_start": {"warm_start": True, "seed": 1},
        }
        validate_extended_state(valid)
        del valid["ema"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_extended_state(valid)

    def test_directory_manifest_rejects_symlink_and_detects_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage_a"
            root.mkdir()
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"frozen")
            before = self.preflight.directory_manifest(root)
            artifact.write_bytes(b"changed")
            after = self.preflight.directory_manifest(root)
            self.assertNotEqual(before, after)
            alias = Path(directory) / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                self.preflight.directory_manifest(alias)

    def test_smoke_launcher_cannot_start_full_run_or_evaluation(self) -> None:
        launcher = (REPO_ROOT / "scripts/e009_stage_b_resume_smoke.slurm").read_text()
        self.assertIn("e009_stage_b_edm5k_13000kimg_smoke.yaml", launcher)
        self.assertNotIn("e009_stage_b_edm5k_30000kimg.yaml", launcher)
        self.assertIn("full_stage_b_continuation=false", launcher)
        self.assertIn("evaluation=false", launcher)
        self.assertIn("e008_swaps=false", launcher)
        self.assertNotIn("20000..20127", launcher)
        self.assertNotIn("0..255", launcher)

    def test_training_code_loads_parent_components_separately(self) -> None:
        source = (
            REPO_ROOT / "src/spectral_diffusion_playground/e009_warm_start.py"
        ).read_text()
        self.assertIn('parent_state["net"]', source)
        self.assertIn('parent_state["optimizer_state"]', source)
        self.assertIn('snapshot["ema"]', source)
        self.assertIn('"exact_stage_a_continuation": False', source)
        for field in (
            '"numpy"',
            '"torch_cpu"',
            '"torch_cuda_all"',
            '"sampler"',
            '"dataloader_generator"',
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
