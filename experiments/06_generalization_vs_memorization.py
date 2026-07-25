"""Planned experiment for training-time generalization and memorization dynamics."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="06_generalization_vs_memorization.py",
    title="When does memorization manifest?",
    question=(
        "Across training checkpoints, when do matched training, held-out, and "
        "deliberately oversampled images develop different low- and "
        "high-frequency recovery curves?"
    ),
)


def main() -> int:
    """Run the scaffolded checkpoint-dynamics experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
