"""Experiment stub for noise versus frequency analysis."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="02_noise_vs_frequency.py",
    title="Noise versus frequency",
    question="How do noise realizations that look similar in pixel space differ spectrally?",
)


def main() -> int:
    """Run the scaffolded noise versus frequency experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
