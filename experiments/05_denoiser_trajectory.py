"""Planned experiment for fixed-model denoising trajectories."""

from __future__ import annotations

from _bootstrap import ExperimentStub, run_experiment_stub

SPEC = ExperimentStub(
    script_name="05_denoiser_trajectory.py",
    title="Fixed-model denoising baseline",
    question=(
        "For a fixed pretrained denoiser, when are low- and high-frequency "
        "bands recovered across noise levels or sampling steps? This measures "
        "inference dynamics, not learning or memorization."
    ),
)


def main() -> int:
    """Run the scaffolded fixed-model trajectory experiment."""
    return run_experiment_stub(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
