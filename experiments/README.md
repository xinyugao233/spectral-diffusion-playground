# Experiments

Each script in this directory should answer one narrow question and remain
independently executable from the repository root.

Conventions:

- keep experiment-specific logic in the script or a dedicated future submodule
- move reusable code into `src/spectral_diffusion_playground/`
- write generated figures to a dedicated subdirectory under `figures/`
- avoid hidden cross-experiment state

The numbered filenames are intentional: they define a reading order for people
new to the repository.
