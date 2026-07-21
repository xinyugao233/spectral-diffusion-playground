"""Experiment stub for FFT visualization."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="01_fft_visualization.py",
    title="FFT visualization",
    question="What becomes obvious when a reference image is viewed through centered magnitude and phase plots?",
)


def main() -> int:
    """Run the scaffolded FFT visualization experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
