"""Run the frozen fixed-model frequency-band recovery baseline.

Experiment 6 evaluates raw clean-image predictions from known-target forward
diffusion observations. It deliberately does not run a reverse sampling chain
or make claims about learning time or memorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import torch

from spectral_diffusion_playground.dataset import (
    load_natural_image_metadata,
    load_preprocessed_natural_image,
    validate_natural_image_dataset,
)
from spectral_diffusion_playground.metrics import (
    frequency_band_components,
    image_frequency_band_recovery_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID: Final[str] = "experiment_06"
UPSTREAM_COMMIT: Final[str] = "22e0df8183507e13a7813f8d38d51b072ca1e67c"
CHECKPOINT_NAME: Final[str] = "256x256_diffusion_uncond.pt"
CHECKPOINT_SIZE: Final[int] = 2_211_383_297
CHECKPOINT_MD5: Final[str] = "fd9dd2335b8736d521de0aed54bd90ca"
DATASET_COMMIT: Final[str] = "a56e230"
IMAGE_IDS: Final[tuple[str, ...]] = tuple(f"image_{index:03d}" for index in range(1, 7))
TIMESTEPS: Final[tuple[int, ...]] = tuple(range(0, 1_000, 25)) + (999,)
EVALUATION_RADII: Final[tuple[float, ...]] = (20.0, 40.0, 80.0)
NOISE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
RECOVERY_THRESHOLD: Final[float] = 0.8
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 20_260_725
CONFIDENCE_LEVEL: Final[float] = 0.95
REPEATABILITY_TOLERANCE: Final[float] = 1e-6
SCHEDULE_TOLERANCE: Final[float] = 1e-14
PROJECTION_TOLERANCE: Final[float] = 1e-10
LOW_COLOR: Final[str] = "#0f766e"
HIGH_COLOR: Final[str] = "#d97706"

MODEL_CONFIG: Final[dict[str, object]] = {
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "attention_resolutions": "32,16,8",
    "dropout": 0.0,
    "diffusion_steps": 1_000,
    "noise_schedule": "linear",
    "timestep_respacing": "",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": False,
    "rescale_learned_sigmas": False,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "use_fp16": True,
    "use_new_attention_order": False,
}

SCORE_FIELDS: Final[tuple[str, ...]] = (
    "experiment_id",
    "image_id",
    "split",
    "checkpoint",
    "trajectory",
    "axis_name",
    "axis_value",
    "timestep",
    "alpha_bar",
    "sigma",
    "cutoff",
    "seed",
    "low_relative_error",
    "high_relative_error",
    "S_low",
    "S_high",
    "out_of_range_fraction",
)


@dataclass(frozen=True, slots=True)
class GuidedDiffusionBundle:
    """Loaded upstream model, diffusion process, and schedule."""

    model: Any
    diffusion: Any
    alpha_bars: np.ndarray
    device: torch.device


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """One image, timestep, cutoff, and noise-seed measurement."""

    experiment_id: str
    image_id: str
    split: str
    checkpoint: str
    trajectory: str
    axis_name: str
    axis_value: float
    timestep: int
    alpha_bar: float
    sigma: float
    cutoff: float
    seed: int
    low_relative_error: float
    high_relative_error: float
    S_low: float
    S_high: float
    out_of_range_fraction: float


def parse_args() -> argparse.Namespace:
    """Parse artifact locations while keeping scientific settings frozen."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 6 exactly as frozen in "
            "docs/experiment_06_fixed_model_denoising.md."
        )
    )
    parser.add_argument(
        "--guided-diffusion-root",
        type=Path,
        required=True,
        help="Clean checkout of the pinned OpenAI guided-diffusion revision.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Official 256x256_diffusion_uncond.pt checkpoint.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=REPO_ROOT / "assets" / "examples" / "metadata.csv",
    )
    parser.add_argument(
        "--image-directory",
        type=Path,
        default=REPO_ROOT / "assets" / "examples" / "natural",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Operational inference batch size; does not change noise pairing.",
    )
    return parser.parse_args()


def require_slurm() -> None:
    """Refuse the heavyweight model evaluation outside a Slurm allocation."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Experiment 6 refuses to run outside Slurm.")


def expected_alpha_bars() -> np.ndarray:
    """Return the pinned 1,000-step linear cumulative-alpha schedule."""
    betas = np.linspace(0.0001, 0.02, 1_000, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def effective_sigma(alpha_bar: float) -> float:
    """Return the VP process's additive-equivalent noise-to-signal ratio."""
    if not 0.0 < alpha_bar <= 1.0:
        raise ValueError("alpha_bar must lie in (0, 1].")
    return float(np.sqrt((1.0 - alpha_bar) / alpha_bar))


def create_noise_batches(
    *,
    num_images: int,
    image_shape: tuple[int, int, int],
) -> dict[int, np.ndarray]:
    """Create one frozen metadata-ordered Gaussian batch per seed."""
    if num_images <= 0:
        raise ValueError("num_images must be positive.")
    batches: dict[int, np.ndarray] = {}
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        batches[seed] = rng.standard_normal(
            (num_images, *image_shape),
            dtype=np.float32,
        )
    return batches


def forward_diffuse(
    clean_model_domain: torch.Tensor,
    noise: torch.Tensor,
    *,
    alpha_bar: float,
) -> torch.Tensor:
    """Construct a known-target variance-preserving noisy observation."""
    if clean_model_domain.shape != noise.shape:
        raise ValueError("clean image and noise tensors must have equal shapes.")
    return np.sqrt(alpha_bar) * clean_model_domain + np.sqrt(1.0 - alpha_bar) * noise


def persistent_threshold_crossing(
    sigmas: Sequence[float],
    scores: Sequence[float],
    *,
    threshold: float = RECOVERY_THRESHOLD,
) -> float | None:
    """Return the first persistent crossing while moving high to low noise.

    Inputs must be ordered from high to low noise. A crossing is accepted only
    when the score is at least ``threshold`` at that point and every later,
    lower-noise point.
    """
    sigma_array = np.asarray(sigmas, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if sigma_array.shape != score_array.shape or sigma_array.ndim != 1:
        raise ValueError("sigmas and scores must be equal-length vectors.")
    if sigma_array.size == 0:
        raise ValueError("sigmas and scores cannot be empty.")
    if np.any(np.diff(sigma_array) > 0.0):
        raise ValueError("sigmas must be ordered from high to low.")

    for index, sigma in enumerate(sigma_array):
        if np.all(score_array[index:] >= threshold):
            return float(sigma)
    return None


def file_identity(path: Path) -> dict[str, object]:
    """Compute checkpoint size, MD5, and SHA-256 in one streaming pass."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {
        "size_bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    """Return the exact Git revision for a checkout."""
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_clean(repository: Path) -> bool:
    """Return whether a Git checkout has no tracked or untracked changes."""
    result = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def verify_external_artifacts(
    guided_diffusion_root: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Verify pinned source and checkpoint identities before loading."""
    if git_revision(guided_diffusion_root) != UPSTREAM_COMMIT:
        raise RuntimeError(
            "guided-diffusion checkout does not match the frozen commit."
        )
    if not git_is_clean(guided_diffusion_root):
        raise RuntimeError("guided-diffusion checkout must be clean.")
    if checkpoint_path.name != CHECKPOINT_NAME:
        raise RuntimeError(f"Checkpoint must be named {CHECKPOINT_NAME}.")

    identity = file_identity(checkpoint_path)
    if identity["size_bytes"] != CHECKPOINT_SIZE:
        raise RuntimeError("Checkpoint byte size does not match the frozen artifact.")
    if identity["md5"] != CHECKPOINT_MD5:
        raise RuntimeError("Checkpoint MD5 does not match the published artifact.")
    return identity


def configure_determinism() -> None:
    """Enable deterministic inference settings before model execution."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_guided_diffusion(
    guided_diffusion_root: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> GuidedDiffusionBundle:
    """Load the exact upstream model and validate its native schedule."""
    source_path = str(guided_diffusion_root.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    script_util = importlib.import_module("guided_diffusion.script_util")
    model, diffusion = script_util.create_model_and_diffusion(**MODEL_CONFIG)

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Checkpoint state dictionary is not an exact model match.")

    model.to(device)
    model.convert_to_fp16()
    model.eval()

    alpha_bars = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    schedule_error = float(np.max(np.abs(alpha_bars - expected_alpha_bars())))
    if schedule_error > SCHEDULE_TOLERANCE:
        raise RuntimeError(
            f"Pinned diffusion schedule mismatch: max error {schedule_error:.3e}."
        )
    return GuidedDiffusionBundle(
        model=model,
        diffusion=diffusion,
        alpha_bars=alpha_bars,
        device=device,
    )


def load_evaluation_images(
    metadata_path: Path,
    image_directory: Path,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate and load the exact six-image calibration set in metadata order."""
    validate_natural_image_dataset(metadata_path, image_directory)
    metadata = load_natural_image_metadata(metadata_path)
    image_ids = tuple(row["image_id"] for row in metadata)
    if image_ids != IMAGE_IDS:
        raise RuntimeError("Natural-image identities or metadata order changed.")

    images = np.stack(
        [
            load_preprocessed_natural_image(image_directory / row["filename"])
            for row in metadata
        ],
        axis=0,
    )
    if images.shape != (6, 256, 256, 3):
        raise RuntimeError(f"Expected six 256x256 RGB images, got {images.shape}.")
    if images.dtype != np.float32:
        raise RuntimeError("Frozen preprocessing must return float32 images.")
    if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
        raise RuntimeError("Preprocessed images must be finite and in [0, 1].")
    return image_ids, images


def verify_frequency_decomposition(images: np.ndarray) -> float:
    """Verify complementary projections on every image and frozen cutoff."""
    maximum_error = 0.0
    for image in images:
        for cutoff in EVALUATION_RADII:
            low, high = frequency_band_components(
                image,
                radius=cutoff,
                exclude_dc=False,
            )
            error = float(np.max(np.abs(low + high - image)))
            maximum_error = max(maximum_error, error)
    if maximum_error > PROJECTION_TOLERANCE:
        raise RuntimeError(
            f"Frequency decomposition gate failed at {maximum_error:.3e}."
        )
    return maximum_error


def evaluate_model_once(
    bundle: GuidedDiffusionBundle,
    images: np.ndarray,
    image_ids: tuple[str, ...],
    noise_batches: dict[int, np.ndarray],
    *,
    batch_size: int,
) -> tuple[ScoreRecord, ...]:
    """Evaluate every frozen image, seed, timestep, and frequency cutoff."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    clean_nchw = np.transpose(images, (0, 3, 1, 2))
    clean_model_domain = torch.from_numpy(2.0 * clean_nchw - 1.0)
    records: list[ScoreRecord] = []

    with torch.inference_mode():
        for seed in NOISE_SEEDS:
            noise = torch.from_numpy(noise_batches[seed])
            for timestep in TIMESTEPS:
                alpha_bar = float(bundle.alpha_bars[timestep])
                sigma = effective_sigma(alpha_bar)
                predictions: list[np.ndarray] = []

                for start in range(0, len(image_ids), batch_size):
                    stop = min(start + batch_size, len(image_ids))
                    noisy = forward_diffuse(
                        clean_model_domain[start:stop],
                        noise[start:stop],
                        alpha_bar=alpha_bar,
                    ).to(bundle.device)
                    timestep_batch = torch.full(
                        (stop - start,),
                        timestep,
                        dtype=torch.long,
                        device=bundle.device,
                    )
                    output = bundle.diffusion.p_mean_variance(
                        bundle.model,
                        noisy,
                        timestep_batch,
                        clip_denoised=False,
                        model_kwargs={},
                    )
                    prediction = output["pred_xstart"]
                    expected_shape = (stop - start, 3, 256, 256)
                    if tuple(prediction.shape) != expected_shape:
                        raise RuntimeError(
                            f"Unexpected prediction shape {tuple(prediction.shape)}."
                        )
                    predictions.append(
                        prediction.detach().to(dtype=torch.float32).cpu().numpy()
                    )

                raw_model_predictions = np.concatenate(predictions, axis=0)
                raw_predictions = np.transpose(
                    (raw_model_predictions.astype(np.float64) + 1.0) / 2.0,
                    (0, 2, 3, 1),
                )
                if not np.isfinite(raw_predictions).all():
                    raise RuntimeError("Model produced non-finite raw predictions.")

                for image_index, image_id in enumerate(image_ids):
                    prediction = raw_predictions[image_index]
                    out_of_range = float(
                        np.mean((prediction < 0.0) | (prediction > 1.0))
                    )
                    for cutoff in EVALUATION_RADII:
                        scores = image_frequency_band_recovery_scores(
                            prediction,
                            images[image_index],
                            radius=cutoff,
                            exclude_dc=True,
                        )
                        record = ScoreRecord(
                            experiment_id=EXPERIMENT_ID,
                            image_id=image_id,
                            split="calibration",
                            checkpoint=CHECKPOINT_NAME,
                            trajectory="direct_x0_prediction",
                            axis_name="vp_noise_to_signal_ratio",
                            axis_value=sigma,
                            timestep=timestep,
                            alpha_bar=alpha_bar,
                            sigma=sigma,
                            cutoff=cutoff,
                            seed=seed,
                            low_relative_error=scores.low_relative_error,
                            high_relative_error=scores.high_relative_error,
                            S_low=scores.low_score,
                            S_high=scores.high_score,
                            out_of_range_fraction=out_of_range,
                        )
                        numeric_values = (
                            record.low_relative_error,
                            record.high_relative_error,
                            record.S_low,
                            record.S_high,
                            record.out_of_range_fraction,
                        )
                        if not np.isfinite(numeric_values).all():
                            raise RuntimeError(
                                "Non-finite recovery metric encountered."
                            )
                        records.append(record)
    return tuple(records)


def save_score_records(records: Sequence[ScoreRecord], output_path: Path) -> Path:
    """Write raw score records before any aggregation or plotting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=SCORE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for field in (
                "axis_value",
                "alpha_bar",
                "sigma",
                "cutoff",
                "low_relative_error",
                "high_relative_error",
                "S_low",
                "S_high",
                "out_of_range_fraction",
            ):
                row[field] = f"{float(row[field]):.10f}"
            writer.writerow(row)
    return output_path


def compare_repeated_records(
    first: Sequence[ScoreRecord],
    second: Sequence[ScoreRecord],
) -> float:
    """Return the maximum numeric difference between paired repeated runs."""
    if len(first) != len(second):
        raise RuntimeError("Repeated runs produced different row counts.")
    numeric_fields = (
        "axis_value",
        "alpha_bar",
        "sigma",
        "low_relative_error",
        "high_relative_error",
        "S_low",
        "S_high",
        "out_of_range_fraction",
    )
    identity_fields = (
        "experiment_id",
        "image_id",
        "split",
        "checkpoint",
        "trajectory",
        "axis_name",
        "timestep",
        "cutoff",
        "seed",
    )
    maximum_difference = 0.0
    for first_record, second_record in zip(first, second, strict=True):
        if any(
            getattr(first_record, field) != getattr(second_record, field)
            for field in identity_fields
        ):
            raise RuntimeError("Repeated runs produced differently ordered rows.")
        maximum_difference = max(
            maximum_difference,
            *(
                abs(
                    float(getattr(first_record, field))
                    - float(getattr(second_record, field))
                )
                for field in numeric_fields
            ),
        )
    return maximum_difference


def _record_lookup(
    records: Sequence[ScoreRecord],
) -> dict[tuple[str, int, int, float], ScoreRecord]:
    """Index records by image, seed, timestep, and cutoff."""
    return {
        (record.image_id, record.seed, record.timestep, record.cutoff): record
        for record in records
    }


def _hierarchical_interval(
    values: np.ndarray,
    image_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> tuple[float, float]:
    """Bootstrap a mean by resampling images and seeds hierarchically."""
    sampled = values[
        image_indices[:, :, np.newaxis],
        seed_indices,
    ]
    means = sampled.mean(axis=(1, 2))
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def build_summary(
    records: Sequence[ScoreRecord],
    image_ids: tuple[str, ...],
) -> dict[str, object]:
    """Aggregate image and seed uncertainty without treating timesteps as IID."""
    lookup = _record_lookup(records)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    image_indices = rng.integers(
        0,
        len(image_ids),
        size=(BOOTSTRAP_RESAMPLES, len(image_ids)),
    )
    seed_indices = rng.integers(
        0,
        len(NOISE_SEEDS),
        size=(BOOTSTRAP_RESAMPLES, len(image_ids), len(NOISE_SEEDS)),
    )
    aggregate_curves: list[dict[str, object]] = []

    metric_names = (
        "low_relative_error",
        "high_relative_error",
        "S_low",
        "S_high",
        "out_of_range_fraction",
    )
    for cutoff in EVALUATION_RADII:
        for timestep in TIMESTEPS:
            sample_record = lookup[(image_ids[0], NOISE_SEEDS[0], timestep, cutoff)]
            aggregate: dict[str, object] = {
                "cutoff": cutoff,
                "timestep": timestep,
                "alpha_bar": sample_record.alpha_bar,
                "sigma": sample_record.sigma,
            }
            for metric_name in metric_names:
                values = np.asarray(
                    [
                        [
                            getattr(
                                lookup[(image_id, seed, timestep, cutoff)], metric_name
                            )
                            for seed in NOISE_SEEDS
                        ]
                        for image_id in image_ids
                    ],
                    dtype=np.float64,
                )
                image_means = values.mean(axis=1)
                interval = _hierarchical_interval(
                    values,
                    image_indices,
                    seed_indices,
                )
                aggregate[metric_name] = {
                    "mean": float(image_means.mean()),
                    "image_sd": float(image_means.std(ddof=1)),
                    "mean_within_image_seed_sd": float(
                        values.std(axis=1, ddof=1).mean()
                    ),
                    "bootstrap_ci": [interval[0], interval[1]],
                }
            aggregate_curves.append(aggregate)

    crossings: list[dict[str, object]] = []
    for cutoff in EVALUATION_RADII:
        for image_id in image_ids:
            for seed in NOISE_SEEDS:
                ordered = [
                    lookup[(image_id, seed, timestep, cutoff)]
                    for timestep in reversed(TIMESTEPS)
                ]
                sigmas = [record.sigma for record in ordered]
                low_crossing = persistent_threshold_crossing(
                    sigmas,
                    [record.S_low for record in ordered],
                )
                high_crossing = persistent_threshold_crossing(
                    sigmas,
                    [record.S_high for record in ordered],
                )
                crossings.append(
                    {
                        "image_id": image_id,
                        "seed": seed,
                        "cutoff": cutoff,
                        "threshold": RECOVERY_THRESHOLD,
                        "low_sigma_crossing": low_crossing,
                        "high_sigma_crossing": high_crossing,
                    }
                )

    return {
        "experiment_id": EXPERIMENT_ID,
        "interpretation": (
            "Fixed-model known-target inference baseline; not learning time or "
            "memorization evidence."
        ),
        "configuration": {
            "image_ids": list(image_ids),
            "timesteps": list(TIMESTEPS),
            "cutoffs": list(EVALUATION_RADII),
            "noise_seeds": list(NOISE_SEEDS),
            "recovery_threshold": RECOVERY_THRESHOLD,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "images_then_seeds_within_images",
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "aggregate_curves": aggregate_curves,
        "persistent_crossings": crossings,
        "limitations": [
            "The six-image calibration set is small and not an ImageNet sample.",
            "Frequency cutoffs are operational and known to be cutoff-sensitive.",
            "Clipped recovery scores can saturate; raw relative errors are primary.",
            "Inference-time recovery does not identify learning or memorization time.",
        ],
    }


def save_json(payload: dict[str, object], output_path: Path) -> Path:
    """Write a stable, human-readable JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def build_manifest(
    args: argparse.Namespace,
    checkpoint_identity: dict[str, object],
    *,
    projection_max_error: float,
    repeatability_max_difference: float,
    device: torch.device,
) -> dict[str, object]:
    """Build the exact run-identity and numerical-gate manifest."""
    git_commit = git_revision(REPO_ROOT)
    return {
        "experiment_id": EXPERIMENT_ID,
        "command": shlex.join(sys.argv),
        "playground_commit": git_commit,
        "playground_clean_at_run_start": git_is_clean(REPO_ROOT),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_root": str(args.guided_diffusion_root.resolve()),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_name": CHECKPOINT_NAME,
        "dataset_commit": DATASET_COMMIT,
        "metadata_path": str(args.metadata_path.resolve()),
        "metadata_sha256": sha256_file(args.metadata_path),
        "preprocessing": (
            "RGB conversion;center crop to square;bicubic resize 256x256;"
            "float32;scale [0,1]"
        ),
        "model_config": MODEL_CONFIG,
        "timesteps": list(TIMESTEPS),
        "cutoffs": list(EVALUATION_RADII),
        "noise_seeds": list(NOISE_SEEDS),
        "batch_size": args.batch_size,
        "repeatability": {
            "runs": 2,
            "tolerance": REPEATABILITY_TOLERANCE,
            "maximum_difference": repeatability_max_difference,
            "passed": repeatability_max_difference <= REPEATABILITY_TOLERANCE,
        },
        "numerical_gates": {
            "schedule_tolerance": SCHEDULE_TOLERANCE,
            "projection_tolerance": PROJECTION_TOLERANCE,
            "projection_max_error": projection_max_error,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "not_applicable"
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def _summary_lookup(
    summary: dict[str, object],
) -> dict[tuple[float, int], dict[str, object]]:
    """Index aggregate summary rows for plotting."""
    curves = summary["aggregate_curves"]
    assert isinstance(curves, list)
    return {(float(row["cutoff"]), int(row["timestep"])): row for row in curves}


def plot_mean_recovery_curves(
    summary: dict[str, object],
    output_path: Path,
) -> Path:
    """Plot aggregate recovery curves with hierarchical bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = _summary_lookup(summary)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for axis, cutoff in zip(axes, EVALUATION_RADII, strict=True):
        rows = [lookup[(cutoff, timestep)] for timestep in TIMESTEPS]
        sigmas = np.asarray([row["sigma"] for row in rows], dtype=np.float64)
        for metric_name, label, color in (
            ("S_low", r"$S_{\mathrm{low}}$", LOW_COLOR),
            ("S_high", r"$S_{\mathrm{high}}$", HIGH_COLOR),
        ):
            metrics = [row[metric_name] for row in rows]
            means = np.asarray([metric["mean"] for metric in metrics])
            lower = np.asarray([metric["bootstrap_ci"][0] for metric in metrics])
            upper = np.asarray([metric["bootstrap_ci"][1] for metric in metrics])
            axis.plot(sigmas, means, color=color, linewidth=2.3, label=label)
            axis.fill_between(sigmas, lower, upper, color=color, alpha=0.18)
        axis.axhline(
            RECOVERY_THRESHOLD,
            color="#64748b",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_title(f"Cutoff r = {int(cutoff)}", fontsize=13)
        axis.set_xlabel(r"Effective noise $\sigma_t$ (high to low)", fontsize=11)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Recovery score", fontsize=11)
    axes[0].legend(frameon=False, fontsize=11)
    figure.suptitle(
        "Fixed-Model Frequency-Band Recovery",
        fontsize=16,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_per_image_curves(
    records: Sequence[ScoreRecord],
    image_ids: tuple[str, ...],
    output_path: Path,
) -> Path:
    """Plot seed-averaged low/high recovery for every image and cutoff."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = _record_lookup(records)
    figure, axes = plt.subplots(
        len(EVALUATION_RADII),
        len(image_ids),
        figsize=(18.0, 8.5),
        sharex=True,
        sharey=True,
    )
    for row_index, cutoff in enumerate(EVALUATION_RADII):
        for column_index, image_id in enumerate(image_ids):
            axis = axes[row_index, column_index]
            sigmas = np.asarray(
                [
                    lookup[(image_id, NOISE_SEEDS[0], timestep, cutoff)].sigma
                    for timestep in TIMESTEPS
                ]
            )
            for metric_name, color in (
                ("S_low", LOW_COLOR),
                ("S_high", HIGH_COLOR),
            ):
                values = np.asarray(
                    [
                        np.mean(
                            [
                                getattr(
                                    lookup[(image_id, seed, timestep, cutoff)],
                                    metric_name,
                                )
                                for seed in NOISE_SEEDS
                            ]
                        )
                        for timestep in TIMESTEPS
                    ]
                )
                axis.plot(sigmas, values, color=color, linewidth=1.5)
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.grid(alpha=0.18)
            if row_index == 0:
                axis.set_title(image_id, fontsize=11)
            if column_index == 0:
                axis.set_ylabel(f"r={int(cutoff)}\nscore", fontsize=10)
            if row_index == len(EVALUATION_RADII) - 1:
                axis.set_xlabel(r"$\sigma_t$", fontsize=10)
    figure.suptitle(
        "Per-Image Recovery Curves (Mean Across Five Noise Seeds)",
        fontsize=16,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_error_and_clipping_diagnostics(
    summary: dict[str, object],
    output_path: Path,
) -> Path:
    """Plot raw relative errors and unclipped out-of-range fractions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = _summary_lookup(summary)
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), sharex=True)
    for column, cutoff in enumerate(EVALUATION_RADII):
        rows = [lookup[(cutoff, timestep)] for timestep in TIMESTEPS]
        sigmas = np.asarray([row["sigma"] for row in rows], dtype=np.float64)
        top_axis = axes[0, column]
        bottom_axis = axes[1, column]

        for metric_name, label, color in (
            ("low_relative_error", r"$E_{\mathrm{low}}$", LOW_COLOR),
            ("high_relative_error", r"$E_{\mathrm{high}}$", HIGH_COLOR),
        ):
            means = np.asarray([row[metric_name]["mean"] for row in rows])
            top_axis.plot(sigmas, means, color=color, linewidth=2.0, label=label)
        clipping = np.asarray([row["out_of_range_fraction"]["mean"] for row in rows])
        bottom_axis.plot(sigmas, clipping, color="#be123c", linewidth=2.0)

        for axis in (top_axis, bottom_axis):
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.grid(alpha=0.2)
        top_axis.set_yscale("log")
        top_axis.set_title(f"Cutoff r = {int(cutoff)}", fontsize=13)
        bottom_axis.set_xlabel(r"Effective noise $\sigma_t$", fontsize=11)
    axes[0, 0].set_ylabel("Raw relative error", fontsize=11)
    axes[1, 0].set_ylabel("Raw values outside [0, 1]", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=11)
    figure.suptitle(
        "Raw-Error and Clipping Diagnostics",
        fontsize=16,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_experiment(args: argparse.Namespace) -> tuple[Path, ...]:
    """Execute the frozen Experiment 6 protocol from acquisition to figures."""
    require_slurm()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 6 requires a CUDA Slurm allocation.")

    configure_determinism()
    device = torch.device("cuda")
    checkpoint_identity = verify_external_artifacts(
        args.guided_diffusion_root,
        args.checkpoint_path,
    )
    image_ids, images = load_evaluation_images(
        args.metadata_path,
        args.image_directory,
    )
    projection_error = verify_frequency_decomposition(images)
    noise_batches = create_noise_batches(
        num_images=len(image_ids),
        image_shape=(3, 256, 256),
    )
    bundle = load_guided_diffusion(
        args.guided_diffusion_root,
        args.checkpoint_path,
        device=device,
    )

    first_records = evaluate_model_once(
        bundle,
        images,
        image_ids,
        noise_batches,
        batch_size=args.batch_size,
    )
    scores_path = save_score_records(
        first_records,
        args.results_dir / "experiment_06_scores.csv",
    )

    second_records = evaluate_model_once(
        bundle,
        images,
        image_ids,
        noise_batches,
        batch_size=args.batch_size,
    )
    repeatability_difference = compare_repeated_records(
        first_records,
        second_records,
    )
    if repeatability_difference > REPEATABILITY_TOLERANCE:
        repeat_path = args.results_dir / "experiment_06_repeat_scores.csv"
        save_score_records(second_records, repeat_path)
        raise RuntimeError(
            "Repeatability gate failed: "
            f"{repeatability_difference:.3e} > {REPEATABILITY_TOLERANCE:.3e}. "
            f"Second-run evidence saved to {repeat_path}."
        )

    summary = build_summary(first_records, image_ids)
    summary_path = save_json(
        summary,
        args.results_dir / "experiment_06_summary.json",
    )
    manifest = build_manifest(
        args,
        checkpoint_identity,
        projection_max_error=projection_error,
        repeatability_max_difference=repeatability_difference,
        device=device,
    )
    manifest_path = save_json(
        manifest,
        args.results_dir / "experiment_06_manifest.json",
    )

    mean_figure = plot_mean_recovery_curves(
        summary,
        args.output_dir / "experiment_06_mean_recovery_curves.png",
    )
    per_image_figure = plot_per_image_curves(
        first_records,
        image_ids,
        args.output_dir / "experiment_06_per_image_recovery_curves.png",
    )
    diagnostic_figure = plot_error_and_clipping_diagnostics(
        summary,
        args.output_dir / "experiment_06_raw_error_and_clipping_diagnostics.png",
    )
    return (
        scores_path,
        summary_path,
        manifest_path,
        mean_figure,
        per_image_figure,
        diagnostic_figure,
    )


def main() -> int:
    """Run Experiment 6 and print exact artifact paths."""
    outputs = run_experiment(parse_args())
    print("Experiment 6 completed.")
    for output in outputs:
        print(f"  {output}")
    print(
        "Interpretation: fixed-model inference recovery only; "
        "not learning-time or memorization evidence."
    )
    return 0
