# Curated Example Images

This directory is reserved for real images that tell a stronger story than the
synthetic fallback.

Suggested additions:

- `castle.png`
- `forest.png`
- `portrait.png`
- `brain_mri.png`
- `face.png`

Example usage:

```bash
python experiments/01_fft_visualization.py --image-path assets/examples/castle.png
python experiments/02_noise_vs_frequency.py --image-path assets/examples/castle.png
```

Until these files are added, `01_fft_visualization.py` falls back to the
deterministic `assets/default_fft_reference.png`.
