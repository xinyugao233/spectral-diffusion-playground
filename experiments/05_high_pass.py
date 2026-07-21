"""Experiment stub for high-pass filtering analysis."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="05_high_pass.py",
    title="High-pass filtering",
    question="Which details are emphasized when low-frequency content is removed?",
)


def main() -> int:
    """Run the scaffolded high-pass filtering experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
