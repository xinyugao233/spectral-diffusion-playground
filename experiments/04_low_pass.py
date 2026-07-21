"""Experiment stub for low-pass filtering analysis."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="04_low_pass.py",
    title="Low-pass filtering",
    question="Which image structures remain after progressively suppressing high frequencies?",
)


def main() -> int:
    """Run the scaffolded low-pass filtering experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
