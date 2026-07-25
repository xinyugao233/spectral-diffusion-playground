"""Tests for the frozen Experiment 6 evaluation contract."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

import spectral_diffusion_playground.denoiser_trajectory as trajectory_module
from spectral_diffusion_playground.denoiser_trajectory import (
    CHECKPOINT_NAME,
    EVALUATION_RADII,
    GuidedDiffusionBundle,
    IMAGE_IDS,
    MODEL_CONFIG,
    NOISE_SEEDS,
    SCORE_FIELDS,
    TIMESTEPS,
    create_noise_batches,
    effective_sigma,
    evaluate_model_once,
    expected_alpha_bars,
    forward_diffuse,
    persistent_threshold_crossing,
)


class DenoiserTrajectoryTest(unittest.TestCase):
    """Keep implementation choices aligned with the frozen E006 protocol."""

    def test_frozen_grid_and_model_identity(self) -> None:
        """The implementation must not silently change scientific settings."""
        self.assertEqual(CHECKPOINT_NAME, "256x256_diffusion_uncond.pt")
        self.assertEqual(
            IMAGE_IDS, tuple(f"image_{index:03d}" for index in range(1, 7))
        )
        self.assertEqual(EVALUATION_RADII, (20.0, 40.0, 80.0))
        self.assertEqual(NOISE_SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(TIMESTEPS, tuple(range(0, 1_000, 25)) + (999,))
        self.assertFalse(MODEL_CONFIG["class_cond"])
        self.assertTrue(MODEL_CONFIG["learn_sigma"])
        self.assertFalse(MODEL_CONFIG["predict_xstart"])
        self.assertEqual(MODEL_CONFIG["timestep_respacing"], "")

    def test_score_schema_preserves_raw_errors(self) -> None:
        """Raw relative errors must remain alongside clipped scores."""
        self.assertIn("low_relative_error", SCORE_FIELDS)
        self.assertIn("high_relative_error", SCORE_FIELDS)
        self.assertIn("S_low", SCORE_FIELDS)
        self.assertIn("S_high", SCORE_FIELDS)
        self.assertIn("timestep", SCORE_FIELDS)
        self.assertIn("sigma", SCORE_FIELDS)

    def test_linear_schedule_matches_frozen_endpoints(self) -> None:
        """The native linear schedule should contain 1,000 valid alpha bars."""
        alpha_bars = expected_alpha_bars()
        self.assertEqual(alpha_bars.shape, (1_000,))
        self.assertTrue(np.all(np.diff(alpha_bars) < 0.0))
        self.assertAlmostEqual(alpha_bars[0], 0.9999)
        self.assertGreater(effective_sigma(float(alpha_bars[-1])), 100.0)

    def test_noise_batches_are_reproducible_and_seed_distinct(self) -> None:
        """Metadata-ordered noise must be paired across all timesteps."""
        first = create_noise_batches(num_images=2, image_shape=(3, 4, 4))
        second = create_noise_batches(num_images=2, image_shape=(3, 4, 4))
        self.assertEqual(set(first), set(NOISE_SEEDS))
        for seed in NOISE_SEEDS:
            np.testing.assert_array_equal(first[seed], second[seed])
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_forward_diffusion_uses_known_target_vp_formula(self) -> None:
        """Forward observations should vary only through alpha and fixed noise."""
        clean = torch.ones((1, 3, 2, 2), dtype=torch.float32)
        noise = torch.full_like(clean, 2.0)
        actual = forward_diffuse(clean, noise, alpha_bar=0.25)
        expected = 0.5 * clean + np.sqrt(0.75) * noise
        torch.testing.assert_close(actual, expected)

    def test_persistent_crossing_rejects_later_reversal(self) -> None:
        """A transient threshold hit must not count as persistent recovery."""
        sigmas = [10.0, 3.0, 1.0, 0.3, 0.1]
        scores = [0.1, 0.85, 0.7, 0.9, 0.95]
        crossing = persistent_threshold_crossing(sigmas, scores, threshold=0.8)
        self.assertEqual(crossing, 0.3)

    def test_persistent_crossing_can_be_unreached(self) -> None:
        """Curves that never remain above threshold should report no crossing."""
        crossing = persistent_threshold_crossing(
            [2.0, 1.0, 0.5],
            [0.1, 0.7, 0.79],
            threshold=0.8,
        )
        self.assertIsNone(crossing)

    def test_model_evaluation_records_raw_unclipped_metrics(self) -> None:
        """A minimal fake denoiser should exercise the raw score pipeline."""

        class IdentityDiffusion:
            """Return the noisy model input as the raw x-start prediction."""

            @staticmethod
            def p_mean_variance(
                model: object,
                noisy: torch.Tensor,
                timestep: torch.Tensor,
                *,
                clip_denoised: bool,
                model_kwargs: dict[str, object],
            ) -> dict[str, torch.Tensor]:
                del model, timestep, model_kwargs
                if clip_denoised:
                    raise AssertionError(
                        "Evaluation must request unclipped predictions."
                    )
                return {"pred_xstart": noisy}

        image = np.random.default_rng(123).random(
            (1, 256, 256, 3),
            dtype=np.float32,
        )
        noise = {0: np.zeros((1, 3, 256, 256), dtype=np.float32)}
        bundle = GuidedDiffusionBundle(
            model=object(),
            diffusion=IdentityDiffusion(),
            alpha_bars=np.asarray([0.9999], dtype=np.float64),
            device=torch.device("cpu"),
        )
        with (
            patch.object(trajectory_module, "TIMESTEPS", (0,)),
            patch.object(trajectory_module, "NOISE_SEEDS", (0,)),
            patch.object(trajectory_module, "EVALUATION_RADII", (20.0,)),
        ):
            records = evaluate_model_once(
                bundle,
                image,
                ("image_001",),
                noise,
                batch_size=1,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].trajectory, "direct_x0_prediction")
        self.assertEqual(records[0].axis_name, "vp_noise_to_signal_ratio")
        self.assertGreaterEqual(records[0].low_relative_error, 0.0)
        self.assertGreaterEqual(records[0].high_relative_error, 0.0)


if __name__ == "__main__":
    unittest.main()
