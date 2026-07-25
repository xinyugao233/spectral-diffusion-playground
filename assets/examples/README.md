# Curated Example Images

This directory is reserved for real images that tell a stronger story than the
synthetic fallback.

Suggested additions:

- `castle.png`
- `forest.png`
- `portrait.png`
- `brain_mri.png`
- `face.png`

Only add images that are original, public domain, or distributed under a
license compatible with this repository. Record the source URL, creator,
license, and any required attribution in this file when an image is added.

Example usage:

```bash
python experiments/01_fft_visualization.py --image-path assets/examples/castle.png
python experiments/02_noise_vs_frequency.py --image-path assets/examples/castle.png
python experiments/03_frequency_decomposition.py --image-path assets/examples/castle.png
```

Until these files are added, Experiments 1–3 fall back to the deterministic
`assets/default_fft_reference.png`.
