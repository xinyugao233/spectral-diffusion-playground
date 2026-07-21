"""Smoke tests for the repository scaffold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.utils import ExperimentStub, format_experiment_stub


class ExperimentStubFormattingTest(unittest.TestCase):
    """Verify that scaffold messaging stays consistent."""

    def test_format_includes_core_fields(self) -> None:
        spec = ExperimentStub(
            script_name="01_fft_visualization.py",
            title="FFT visualization",
            question="What becomes obvious in the frequency domain?",
        )

        message = format_experiment_stub(spec)

        self.assertIn("[01_fft_visualization.py] FFT visualization", message)
        self.assertIn("Question: What becomes obvious in the frequency domain?", message)
        self.assertIn("Status: Scaffold only; experiment implementation pending.", message)


if __name__ == "__main__":
    unittest.main()
