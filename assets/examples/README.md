# Curated Example Images

This directory is reserved for provenance-recorded real images that tell a
stronger story than the synthetic fallback.

Experiment 5 calibration images belong under `natural/`. Every image must have
a matching row in `metadata.csv` containing its exact source, creator, license,
URL, download date, original resolution, and preprocessing.

Only add images that are original, public domain, or distributed under a
license compatible with this repository. Do not infer a license from the
hosting platform; record the exact license attached to the source.

Example usage:

```bash
python experiments/01_fft_visualization.py --image-path assets/examples/castle.png
python experiments/02_noise_vs_frequency.py --image-path assets/examples/castle.png
python experiments/03_frequency_decomposition.py --image-path assets/examples/castle.png
python experiments/04_structure_detail_metrics.py --image-path assets/examples/castle.png
```

Until these files are added, Experiments 1–4 fall back to the deterministic
`assets/default_fft_reference.png`. Experiment 5 must remain a stub.
