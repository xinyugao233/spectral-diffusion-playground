"""Run the frozen natural-image calibration of `S_low` and `S_high`."""

from __future__ import annotations

# Import the shared bootstrap so this script can be run directly from the repo root.
import _bootstrap  # noqa: F401

from spectral_diffusion_playground.natural_image_calibration import main


if __name__ == "__main__":
    raise SystemExit(main())
