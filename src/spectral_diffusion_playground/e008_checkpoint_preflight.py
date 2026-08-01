"""Pure utilities for the E008 baseline-only checkpoint preflight."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from spectral_diffusion_playground.e006_transition_swaps import (
    clopper_pearson_interval,
)

EXPERIMENT_ID: Final[str] = "experiment_08_preflight"
PILOT_SEEDS: Final[tuple[int, ...]] = tuple(range(10_000, 10_128))
CONFIRMATORY_SEEDS: Final[tuple[int, ...]] = tuple(range(256))
ELIGIBLE_COUNT_MIN: Final[int] = 13
ELIGIBLE_COUNT_MAX: Final[int] = 115
SNAPSHOT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^network-snapshot-(?P<kimg>\d{6})\.pkl$"
)


def parse_training_kimg(filename: str) -> int:
    """Parse training duration from an exact EDM snapshot filename."""
    match = SNAPSHOT_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Malformed EDM snapshot filename: {filename}")
    return int(match.group("kimg"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoint_paths(root: Path) -> tuple[list[Path], list[Path]]:
    """Discover accepted-name and malformed snapshot candidates deterministically."""
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    candidates = sorted(path for path in root.iterdir() if path.is_file())
    accepted = [path for path in candidates if SNAPSHOT_PATTERN.fullmatch(path.name)]
    rejected = [
        path
        for path in candidates
        if path.name.startswith("network-snapshot-") and path not in accepted
    ]
    durations = [parse_training_kimg(path.name) for path in accepted]
    duplicates = sorted({value for value in durations if durations.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate snapshot training durations: {duplicates}")
    return accepted, rejected


def validate_no_swap_only(
    *, donor_checkpoint: str | None = None, swap_window: str | None = None
) -> None:
    """Reject any request that could execute a partial-trajectory swap."""
    if donor_checkpoint is not None:
        raise ValueError("E008 preflight forbids donor checkpoints")
    if swap_window is not None:
        raise ValueError("E008 preflight forbids swap windows")


def candidate_is_eligible(memorized_count: int, n_samples: int) -> bool:
    """Apply the frozen 13..115 count rule for exactly 128 pilot samples."""
    if n_samples != len(PILOT_SEEDS):
        raise ValueError("Eligibility requires exactly 128 pilot samples")
    return ELIGIBLE_COUNT_MIN <= memorized_count <= ELIGIBLE_COUNT_MAX


def summarize_candidate(
    *,
    model_role: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    training_kimg: int,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate one checkpoint with exact descriptive binomial uncertainty."""
    failures = [row for row in rows if row["status"] != "ok"]
    successes = [row for row in rows if row["status"] == "ok"]
    count = sum(int(row["memorized"]) for row in successes)
    complete = len(rows) == len(PILOT_SEEDS) and not failures
    if complete:
        low, high = clopper_pearson_interval(count, len(PILOT_SEEDS))
        eligible = candidate_is_eligible(count, len(PILOT_SEEDS))
        reason = (
            "count_in_13_through_115" if eligible else "count_outside_13_through_115"
        )
        status = "complete"
    else:
        low = high = float("nan")
        eligible = False
        reason = "incomplete_or_failed_pilot"
        status = "failed" if failures else "incomplete"
    return {
        "model_role": model_role,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "training_kimg": training_kimg,
        "n_samples": len(successes),
        "memorized_count": count,
        "memorization_rate": count / len(PILOT_SEEDS) if complete else float("nan"),
        "ci95_low": low,
        "ci95_high": high,
        "eligible": eligible,
        "eligibility_reason": reason,
        "n_failures": len(failures),
        "status": status,
    }


def select_model_pair(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Select the closest eligible cross-role pair with the frozen SHA tie-break."""
    eligible = [row for row in summaries if bool(row["eligible"])]
    left = [row for row in eligible if row["model_role"] == "edm_1k"]
    right = [row for row in eligible if row["model_role"] == "edm_50k"]
    if not left or not right:
        return None
    ranked: list[tuple[float, str, str, Mapping[str, object], Mapping[str, object]]] = (
        []
    )
    for first in left:
        for second in right:
            difference = abs(
                float(first["memorization_rate"]) - float(second["memorization_rate"])
            )
            ranked.append(
                (
                    difference,
                    str(first["checkpoint_sha256"]),
                    str(second["checkpoint_sha256"]),
                    first,
                    second,
                )
            )
    difference, _, _, first, second = min(ranked, key=lambda item: item[:3])
    return {
        "absolute_pilot_rate_difference": difference,
        "edm_1k": dict(first),
        "edm_50k": dict(second),
    }


def independent_seed_latents(
    seeds: Iterable[int], shape: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Create independent deterministic NumPy test latents for seed-contract tests."""
    return {
        int(seed): np.random.RandomState(int(seed)).standard_normal(shape)
        for seed in seeds
    }


def merge_resume_rows(
    existing: Sequence[Mapping[str, object]],
    incoming: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge resumable rows without duplicate checkpoint/seed keys."""
    merged: dict[tuple[str, int], dict[str, object]] = {}
    for row in list(existing) + list(incoming):
        key = (str(row["checkpoint_sha256"]), int(row["sample_seed"]))
        value = dict(row)
        if key in merged and merged[key] != value:
            raise ValueError(f"Conflicting resume row for {key}")
        merged[key] = value
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row["model_role"]),
            int(row["training_kimg"]),
            str(row["checkpoint_sha256"]),
            int(row["sample_seed"]),
        ),
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file or return an empty list when it does not exist."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
