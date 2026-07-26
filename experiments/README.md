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

Experiments 1–3 are the completed Fourier foundations. Experiments 4–6 are
being redesigned as a paper-derived clean-room reimplementation:

- Experiment 4 will freeze an operational CIFAR-10 frequency cutoff.
- Experiment 5 will decompose fixed-sigma denoising residual energy into
  orthogonal low- and high-frequency components.
- Experiment 6 will test whole-denoiser swaps around the resulting transition
  windows.

The original executed paper code was unavailable. No Experiment 4–6 result or
claim currently exists in this repository.
