"""Run the frozen fixed-model frequency-band recovery baseline."""

from __future__ import annotations

# Import the shared bootstrap so this script can be run directly from the repo root.
import _bootstrap  # noqa: F401

from spectral_diffusion_playground.denoiser_trajectory import main


if __name__ == "__main__":
    raise SystemExit(main())
