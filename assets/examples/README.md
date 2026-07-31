# Curated Example Images

This directory is reserved for provenance-recorded real images that tell a
stronger story than the synthetic fallback.

The images under `natural/` are provenance-recorded examples retained for
general Fourier visualization. Every image has a matching row in `metadata.csv`
with its source, creator, license, URL, download date, original resolution, and
documented preprocessing.

Only add images that are original, public domain, or distributed under a
license compatible with this repository. Do not infer a license from the
hosting platform; record the exact license attached to the source.

Example usage:

```bash
python experiments/01_fft_visualization.py \
    --image-path assets/examples/natural/image_005.jpg
python experiments/02_noise_vs_frequency.py \
    --image-path assets/examples/natural/image_002.jpg
python experiments/03_frequency_decomposition.py \
    --image-path assets/examples/natural/image_005.jpg
```

Experiments 1–3 also support the deterministic
`assets/default_fft_reference.png` fallback.
