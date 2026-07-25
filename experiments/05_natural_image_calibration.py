"""Planned natural-image calibration of frequency-band recovery metrics."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="05_natural_image_calibration.py",
    title="Natural image calibration of S_low and S_high",
    question=(
        "How stable are low- and high-frequency recovery metrics across "
        "provenance-recorded natural images, cutoffs, and controlled trajectories?"
    ),
)


def main() -> int:
    """Run the scaffolded natural-image calibration experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
