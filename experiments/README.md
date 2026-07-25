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

The sequence has two distinct roles:

- Experiments 1–4 define and calibrate the frequency-space measurements.
- Experiment 5 calibrates those measurements across natural images.
- Experiment 6 is a fixed-model inference baseline.
- Experiment 7 is the checkpoint-aligned memorization study.

Experiment 5 has passed its provenance, uncertainty, cutoff-sweep, and failure
analysis gate. Experiment 6 has a frozen known-target protocol but remains
unimplemented; no checkpoint has been downloaded. It must preserve the
distinction between fixed-model inference dynamics and checkpoint-aligned
learning dynamics.

Low-frequency recovery is a coarse/global-structure proxy; high-frequency
recovery is a fine-detail proxy. Do not treat either operational band as a
semantic category.

Do not interpret high-frequency recovery by itself as memorization. A
memorization analysis requires matched training and held-out controls across
checkpoints, with distributional differences treated as potential confounders.
