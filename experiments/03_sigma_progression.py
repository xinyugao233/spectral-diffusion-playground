"""Experiment stub for sigma progression analysis."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="03_sigma_progression.py",
    title="Sigma progression",
    question="How does spectral structure change as the noise scale increases?",
)


def main() -> int:
    """Run the scaffolded sigma progression experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
