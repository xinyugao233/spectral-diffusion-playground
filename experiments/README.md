# Experiments

Each script in this directory should answer one narrow question and remain
independently executable from the repository root.

Conventions:

- keep experiment-specific logic in the script or a dedicated future submodule
- move reusable code into `src/spectral_diffusion_playground/`
- write generated figures under `figures/` and raw numerical outputs under `results/`
- avoid hidden cross-experiment state

The numbered filenames are intentional: they define a reading order for people
new to the repository.

Experiments 1–3 are the completed Fourier foundations. Experiments 4–6 form a
paper-derived clean-room reimplementation:

- Experiment 4 generates a frozen CIFAR-10 cutoff review packet. Independent
  human review is pending and no cutoff has been selected.
- Experiment 5 will decompose fixed-sigma denoising residual energy into
  orthogonal low- and high-frequency components.
- Experiment 6 will test whole-denoiser swaps around the resulting transition
  windows.

The original executed paper code was unavailable. Experiment 4 currently
contains numerical projection checks and review materials, not a cutoff result
or semantic claim. No Experiment 5–6 result exists.
