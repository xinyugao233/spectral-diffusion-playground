"""Frozen numerical utilities for Experiment 6 transition-window swaps."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

EXPERIMENT_ID: Final[str] = "experiment_06"
RUN_ID: Final[str] = "e006_transition_window_swaps"
PROTOCOL_COMMIT: Final[str] = "068c7e3a745fb51b1d2416524b7e29f70b0b5f08"
PAIR_BOOTSTRAP_SEED: Final[int] = 20260730
PAIR_BOOTSTRAP_RESAMPLES: Final[int] = 10_000
EFFECT_THRESHOLD: Final[float] = 0.10
SAMPLE_SEEDS: Final[tuple[int, ...]] = tuple(range(256))
SIGMA_GRID: Final[tuple[float, ...]] = (
    80.0,
    57.58598472124816,
    40.785573796507961,
    28.374584604156844,
    19.352452980325229,
    12.91008238075732,
    8.4009353090998165,
    5.3151945217963821,
    3.2568215197655368,
    1.9233398370400518,
    1.088170636545279,
    0.58534812319454221,
    0.29644228447915727,
    0.13951646873101678,
    0.05994731123547159,
    0.022934518372333384,
    0.0075280199627840785,
    0.0020000000000000031,
)
TERMINAL_SIGMA: Final[float] = 0.0

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class WindowSpec:
    """One inclusive whole-denoiser swap window."""

    name: str
    start_index: int
    end_index: int

    @property
    def indices(self) -> tuple[int, ...]:
        """Return all included denoiser-call indices."""
        return tuple(range(self.start_index, self.end_index + 1))

    @property
    def start_sigma(self) -> float:
        """Return the sigma at the first included call."""
        return SIGMA_GRID[self.start_index]

    @property
    def end_sigma(self) -> float:
        """Return the sigma at the last included call."""
        return SIGMA_GRID[self.end_index]

    def contains(self, step_index: int) -> bool:
        """Return whether ``step_index`` is inside the inclusive window."""
        return self.start_index <= step_index <= self.end_index


@dataclass(frozen=True)
class ConditionSpec:
    """One no-swap or whole-denoiser swap condition."""

    name: str
    base_model: str
    donor_model: str | None
    window: WindowSpec | None

    def model_for_step(self, step_index: int) -> str:
        """Return the frozen model identity for one denoiser call."""
        if self.window is not None and self.window.contains(step_index):
            if self.donor_model is None:
                raise RuntimeError("Swap window requires a donor model")
            return self.donor_model
        return self.base_model


WINDOWS: Final[tuple[WindowSpec, ...]] = (
    WindowSpec("low_transition", 5, 11),
    WindowSpec("high_transition", 11, 14),
    WindowSpec("combined_transition", 5, 14),
    WindowSpec("paper_medium_reference", 6, 13),
    WindowSpec("low_pre_control", 0, 6),
    WindowSpec("low_post_control", 11, 17),
    WindowSpec("high_pre_control", 7, 10),
    WindowSpec("high_post_control", 14, 17),
)
WINDOW_BY_NAME: Final[dict[str, WindowSpec]] = {
    window.name: window for window in WINDOWS
}


def frozen_conditions() -> tuple[ConditionSpec, ...]:
    """Return the two baselines followed by all 16 frozen swap conditions."""
    conditions = [
        ConditionSpec("edm_1k_no_swap", "edm_1k", None, None),
        ConditionSpec("edm_50k_no_swap", "edm_50k", None, None),
    ]
    for base_model, donor_model in (
        ("edm_1k", "edm_50k"),
        ("edm_50k", "edm_1k"),
    ):
        prefix = f"{base_model}_base__{donor_model}_donor"
        conditions.extend(
            ConditionSpec(
                name=f"{prefix}__{window.name}",
                base_model=base_model,
                donor_model=donor_model,
                window=window,
            )
            for window in WINDOWS
        )
    return tuple(conditions)


def euler_update(
    state: np.ndarray,
    denoised: np.ndarray,
    sigma: float,
    sigma_next: float,
) -> FloatArray:
    """Apply exactly one first-order Euler denoising update."""
    if sigma <= 0.0:
        raise ValueError("Euler denoiser sigma must be positive")
    x = np.asarray(state, dtype=np.float64)
    prediction = np.asarray(denoised, dtype=np.float64)
    derivative = (x - prediction) / np.float64(sigma)
    return x + np.float64(sigma_next - sigma) * derivative


def pure_euler_sample_numpy(
    latent: np.ndarray,
    condition: ConditionSpec,
    model_calls: Mapping[str, object],
) -> FloatArray:
    """Run the frozen pure-Euler schedule with NumPy-compatible callables.

    This implementation is used by focused tests. Production sampling follows
    the same update in Torch and makes exactly one model call per step.
    """
    state = np.asarray(latent, dtype=np.float64) * SIGMA_GRID[0]
    schedule = SIGMA_GRID + (TERMINAL_SIGMA,)
    for step_index, (sigma, sigma_next) in enumerate(
        zip(schedule[:-1], schedule[1:])
    ):
        model_name = condition.model_for_step(step_index)
        model = model_calls[model_name]
        denoised = model(state, sigma)  # type: ignore[operator]
        state = euler_update(state, denoised, sigma, sigma_next)
    return state


def generated_sample_hash(sample: np.ndarray) -> str:
    """Hash a generated sample in canonical little-endian float64 NCHW form."""
    value = np.asarray(sample, dtype="<f8", order="C")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def nearest_two(
    generated: np.ndarray,
    reference: np.ndarray,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Return sorted nearest-two reference positions and Euclidean distances."""
    queries = np.asarray(generated, dtype=np.float64)
    bank = np.asarray(reference, dtype=np.float64)
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError(
            f"Expected compatible 2D arrays, got {queries.shape} and {bank.shape}"
        )
    squared = (
        np.sum(queries * queries, axis=1, keepdims=True)
        + np.sum(bank * bank, axis=1)[None, :]
        - 2.0 * (queries @ bank.T)
    )
    np.maximum(squared, 0.0, out=squared)
    candidate = np.argpartition(squared, kth=1, axis=1)[:, :2]
    candidate_distances = np.take_along_axis(squared, candidate, axis=1)
    order = np.argsort(candidate_distances, axis=1, kind="stable")
    indices = np.take_along_axis(candidate, order, axis=1).astype(np.int64)
    distances = np.sqrt(
        np.take_along_axis(candidate_distances, order, axis=1)
    ).astype(np.float64, copy=False)
    return indices, distances


def memorization_flags(distances: np.ndarray) -> BoolArray:
    """Apply the strict frozen criterion ``d1NN < d2NN / 3``."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected nearest-two distances, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Nearest-neighbor distances must be finite and nonnegative")
    return np.asarray(values[:, 0] < values[:, 1] / 3.0, dtype=np.bool_)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Return ``P[X <= k]`` for ``X ~ Binomial(n, probability)``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    return float(
        sum(
            math.comb(n, index)
            * probability**index
            * (1.0 - probability) ** (n - index)
            for index in range(k + 1)
        )
    )


def _bisect_probability(
    predicate,
    *,
    iterations: int = 80,
) -> float:
    """Bisect a monotone probability predicate on ``[0, 1]``."""
    low = 0.0
    high = 1.0
    for _ in range(iterations):
        midpoint = (low + high) / 2.0
        if predicate(midpoint):
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the exact two-sided Clopper-Pearson binomial interval."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    alpha = 1.0 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lower = _bisect_probability(
            lambda probability: 1.0
            - _binomial_cdf(successes - 1, trials, probability)
            < alpha / 2.0
        )
    if successes == trials:
        upper = 1.0
    else:
        upper = _bisect_probability(
            lambda probability: _binomial_cdf(
                successes, trials, probability
            )
            > alpha / 2.0
        )
    return float(lower), float(upper)


def exact_sign_test_p_value(negative: int, positive: int) -> float:
    """Return the exact two-sided sign-test p-value for discordant pairs."""
    if negative < 0 or positive < 0:
        raise ValueError("Discordant counts must be nonnegative")
    total = negative + positive
    if total == 0:
        return 1.0
    smaller = min(negative, positive)
    one_sided = sum(math.comb(total, index) for index in range(smaller + 1))
    return float(min(1.0, 2.0 * one_sided / (2**total)))


def paired_bootstrap_interval(
    deltas: np.ndarray,
    *,
    seed: int = PAIR_BOOTSTRAP_SEED,
    resamples: int = PAIR_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Bootstrap the mean paired seed-level change."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Paired deltas must be one nonempty finite vector")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    offset = 0
    while offset < resamples:
        count = min(1000, resamples - offset)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[offset : offset + count] = values[indices].mean(axis=1)
        offset += count
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_effect_summary(
    baseline: np.ndarray,
    swapped: np.ndarray,
    *,
    seed: int = PAIR_BOOTSTRAP_SEED,
    resamples: int = PAIR_BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int | bool]:
    """Summarize one paired swap-versus-baseline memorization comparison."""
    base = np.asarray(baseline, dtype=np.int8)
    swap = np.asarray(swapped, dtype=np.int8)
    if base.shape != swap.shape or base.ndim != 1:
        raise ValueError("Baseline and swap labels must be matching 1D arrays")
    if not np.all(np.isin(base, (0, 1))) or not np.all(np.isin(swap, (0, 1))):
        raise ValueError("Memorization labels must be binary")
    deltas = swap - base
    ci_low, ci_high = paired_bootstrap_interval(
        deltas, seed=seed, resamples=resamples
    )
    negative = int(np.sum(deltas == -1))
    positive = int(np.sum(deltas == 1))
    zero = int(np.sum(deltas == 0))
    rate_difference = float(swap.mean() - base.mean())
    direction_supported = bool(
        (rate_difference > 0.0 and ci_low > 0.0)
        or (rate_difference < 0.0 and ci_high < 0.0)
    )
    return {
        "n_pairs": int(base.size),
        "baseline_rate": float(base.mean()),
        "swap_rate": float(swap.mean()),
        "rate_difference": rate_difference,
        "paired_mean_delta": float(deltas.mean()),
        "paired_ci95_low": ci_low,
        "paired_ci95_high": ci_high,
        "discordant_negative": negative,
        "discordant_positive": positive,
        "discordant_zero": zero,
        "sign_test_p_value": exact_sign_test_p_value(negative, positive),
        "direction_supported": direction_supported,
    }


def effect_gap_bootstrap_interval(
    transition_deltas: np.ndarray,
    control_deltas: np.ndarray,
    *,
    seed: int,
    resamples: int = PAIR_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Bootstrap ``|transition effect| - |control effect|`` over paired seeds."""
    transition = np.asarray(transition_deltas, dtype=np.float64)
    control = np.asarray(control_deltas, dtype=np.float64)
    if transition.shape != control.shape or transition.ndim != 1:
        raise ValueError("Transition/control deltas must be matching 1D arrays")
    generator = np.random.default_rng(seed)
    gaps = np.empty(resamples, dtype=np.float64)
    offset = 0
    while offset < resamples:
        count = min(1000, resamples - offset)
        indices = generator.integers(
            0, transition.size, size=(count, transition.size)
        )
        gaps[offset : offset + count] = np.abs(
            transition[indices].mean(axis=1)
        ) - np.abs(control[indices].mean(axis=1))
        offset += count
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def transition_influence(
    baseline: np.ndarray,
    transition: np.ndarray,
    pre_control: np.ndarray,
    post_control: np.ndarray,
    *,
    seed: int = PAIR_BOOTSTRAP_SEED,
    resamples: int = PAIR_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Apply the frozen practical threshold and uncertainty safeguards."""
    base = np.asarray(baseline, dtype=np.int8)
    transition_values = np.asarray(transition, dtype=np.int8)
    pre = np.asarray(pre_control, dtype=np.int8)
    post = np.asarray(post_control, dtype=np.int8)
    if not (
        base.shape == transition_values.shape == pre.shape == post.shape
        and base.ndim == 1
    ):
        raise ValueError("Transition and control labels must align by seed")

    transition_summary = paired_effect_summary(
        base, transition_values, seed=seed, resamples=resamples
    )
    pre_summary = paired_effect_summary(
        base, pre, seed=seed + 1, resamples=resamples
    )
    post_summary = paired_effect_summary(
        base, post, seed=seed + 2, resamples=resamples
    )
    transition_delta = transition_values - base
    pre_delta = pre - base
    post_delta = post - base
    pre_gap_ci = effect_gap_bootstrap_interval(
        transition_delta,
        pre_delta,
        seed=seed + 3,
        resamples=resamples,
    )
    post_gap_ci = effect_gap_bootstrap_interval(
        transition_delta,
        post_delta,
        seed=seed + 4,
        resamples=resamples,
    )
    effect = abs(float(transition_summary["rate_difference"]))
    pre_effect = abs(float(pre_summary["rate_difference"]))
    post_effect = abs(float(post_summary["rate_difference"]))
    point_threshold = bool(
        effect >= EFFECT_THRESHOLD
        and effect >= pre_effect + EFFECT_THRESHOLD
        and effect >= post_effect + EFFECT_THRESHOLD
    )
    uncertainty_support = bool(
        transition_summary["direction_supported"]
        and pre_gap_ci[0] > 0.0
        and post_gap_ci[0] > 0.0
    )
    return {
        "transition": transition_summary,
        "pre_control": pre_summary,
        "post_control": post_summary,
        "transition_effect_magnitude": effect,
        "pre_control_effect_magnitude": pre_effect,
        "post_control_effect_magnitude": post_effect,
        "pre_effect_gap_ci95": list(pre_gap_ci),
        "post_effect_gap_ci95": list(post_gap_ci),
        "passes_point_threshold": point_threshold,
        "uncertainty_support": uncertainty_support,
        "influential": bool(point_threshold and uncertainty_support),
    }


def classify_outcome(
    influential: Mapping[tuple[str, str], bool],
    *,
    invalid: bool = False,
    baseline_degenerate: bool = False,
) -> str:
    """Return one frozen E006 outcome label from the four primary tests."""
    required = {
        ("edm_1k_to_edm_50k", "low_transition"),
        ("edm_1k_to_edm_50k", "high_transition"),
        ("edm_50k_to_edm_1k", "low_transition"),
        ("edm_50k_to_edm_1k", "high_transition"),
    }
    if set(influential) != required:
        raise ValueError("Outcome classification requires exactly four primary tests")
    if invalid or baseline_degenerate:
        return "INCONCLUSIVE"
    passed = {key for key, value in influential.items() if value}
    if passed == required:
        return "YES"
    if not passed:
        return "NO"
    directions = {direction for direction, _ in passed}
    windows = {window for _, window in passed}
    if len(passed) == 1 or len(directions) == 1 or len(windows) == 1:
        return "PARTIAL"
    return "MIXED"


def deterministic_qualitative_selection(
    baseline: Sequence[bool],
    swapped: Sequence[bool],
    seeds: Sequence[int],
) -> dict[str, list[int]]:
    """Apply the frozen first-two-per-category qualitative selection rule."""
    base = np.asarray(baseline, dtype=np.bool_)
    swap = np.asarray(swapped, dtype=np.bool_)
    seed_values = np.asarray(seeds, dtype=np.int64)
    if base.shape != swap.shape or base.shape != seed_values.shape:
        raise ValueError("Qualitative selection arrays must have matching shapes")
    categories = {
        "newly_memorized": (~base) & swap,
        "no_longer_memorized": base & (~swap),
        "unchanged_memorized": base & swap,
        "unchanged_non_memorized": (~base) & (~swap),
    }
    return {
        name: [int(seed) for seed in seed_values[mask][:2]]
        for name, mask in categories.items()
    }
