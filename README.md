# Spectral Diffusion Playground

![Understanding Images in Fourier Space](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png)

Spectral Diffusion Playground is a small research repository for understanding diffusion-model behavior through Fourier analysis, controlled perturbations, and clean visualizations.

**Current status:** Experiments 1 and 2 are complete. Later experiments remain planned.

## Scope

This repository is designed for readers who want intuition, not another end-to-end diffusion training stack.

It is meant to provide:

- small experiments with one clear question each
- reproducible scripts rather than notebook-only workflows
- shared utilities collected in a real Python package
- figures that are suitable for research notes, talks, and portfolio review

It is not meant to be:

- a benchmark suite
- a production diffusion library
- a claim-heavy research release before the evidence exists

## Why Fourier Analysis Matters for Diffusion

Diffusion models are usually discussed in pixel space: add noise, predict noise, denoise step by step. That view is useful, but incomplete.

The frequency domain exposes different questions:

- Which structures disappear first as noise increases?
- How do low-frequency semantics and high-frequency detail degrade differently?
- When two perturbations look similarly strong in pixel space, do they have the same spectral signature?
- What does a denoiser implicitly need to recover at different frequency bands?

A Fourier view does not replace the standard diffusion formulation. It provides a complementary lens that is often easier to visualize and reason about.

## Design Principles

- One experiment, one question.
- Every experiment should run independently.
- Shared code belongs in `src/spectral_diffusion_playground/`.
- Outputs should be easy to trace back to the script that produced them.
- The repository should stay readable to someone skimming it for five minutes.

## Featured Experiment

### Understanding Images in Fourier Space

Status: Complete.

This experiment turns one image into a compact story:

- original image in pixel space
- centered Fourier magnitude
- log-scaled Fourier magnitude
- inverse FFT reconstruction

Run it with a curated real image once `assets/examples/` is populated:

```bash
python experiments/01_fft_visualization.py \
    --image-path assets/examples/castle.png
```

Today the script still includes a deterministic synthetic fallback so the repo stays runnable even before the curated example set is added.

Display normalization:

- linear magnitude uses exact max normalization after channel averaging
- log magnitude uses `log1p(magnitude + 1e-12)` before the same max normalization

### How Gaussian Noise Changes Frequency Content

Status: Complete.

Motivation: diffusion models add Gaussian noise in pixel space, but the same perturbation becomes easier to reason about when it is inspected in Fourier space. This experiment fixes one image, varies only `sigma` in `x_sigma = x + sigma * epsilon`, and shows both the noisy observations and their spectral summaries.

![How Gaussian Noise Changes Frequency Content](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png)

![Normalized Radial Spectral Distribution](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_normalized_radial_distribution.png)

Run it with a curated real image once `assets/examples/` is populated:

```bash
python experiments/02_noise_vs_frequency.py \
    --image-path assets/examples/castle.png
```

Display normalization:

- noisy images are clipped only for visualization; the additive perturbation itself is not clipped
- log spectra use `log1p(magnitude + 1e-12)` followed by one shared global 99.5th-percentile normalization after channel averaging
- the raw radial analysis also saves annulus-averaged power `E(r)` on a log-scaled y-axis
- the normalized radial spectral-distribution figure excludes the centered DC bin, then plots `E(r) / \sum_{r>0} E(r)` with a dashed white-noise reference line

Takeaway:

- larger `sigma` values visibly erase image structure in pixel space
- the log spectra become more uniformly elevated across the frequency plane as noise dominates the image
- after DC exclusion and normalization, larger `sigma` values spread relative radial power more uniformly across frequency bands

## Planned Experiments

| Script | Title | Question | Planned output | Status |
| --- | --- | --- | --- | --- |
| `01_fft_visualization.py` | Understanding Images in Fourier Space | What becomes obvious when an image is viewed through centered magnitude and phase plots? | Reversible pixel-space to frequency-space walkthrough | [x] |
| `02_noise_vs_frequency.py` | How Gaussian Noise Changes Frequency Content | How does additive Gaussian noise change both image-space structure and Fourier-space energy? | Spatial/spectral grid plus radial energy curves | [x] |
| `03_sigma_progression.py` | Noise Scale Progression in Fourier Space | How does increasing noise scale change spectral structure? | Multi-panel progression over sigma values | [ ] |
| `04_low_pass.py` | What Low-Pass Filtering Preserves | What survives aggressive removal of high frequencies? | Low-pass reconstructions and spectra | [ ] |
| `05_high_pass.py` | What High-Pass Filtering Emphasizes | What is emphasized when low frequencies are suppressed? | High-pass reconstructions and residual views | [ ] |

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Running an Experiment

Each experiment is an independent script:

```bash
python experiments/01_fft_visualization.py
```

To compare RGB and grayscale views in the same artifact:

```bash
python experiments/01_fft_visualization.py --grayscale
```

## Repository Layout

```text
spectral-diffusion-playground/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── .gitignore
├── src/
│   └── spectral_diffusion_playground/
│       ├── __init__.py
│       ├── fft.py
│       ├── filters.py
│       ├── noise.py
│       ├── visualization.py
│       └── utils.py
├── experiments/
│   ├── _bootstrap.py
│   ├── README.md
│   ├── 01_fft_visualization.py
│   ├── 02_noise_vs_frequency.py
│   ├── 03_sigma_progression.py
│   ├── 04_low_pass.py
│   └── 05_high_pass.py
├── assets/
├── figures/
├── docs/
└── tests/
```

## Future Research Directions

- Compare spectral behavior across datasets or semantic classes.
- Replace the synthetic fallback with a curated set of real example images in `assets/examples/`.
- Add animated FFT walkthroughs and GIF exports for recruiting and tutorial use.
- Study whether different noise schedules induce distinct spectral trajectories.
- Connect score estimation and denoising behavior to frequency recovery.
- Examine spectral effects of conditioning, guidance, or architecture choices.
- Use this repository as a base for paper-ready diagnostic figures.

## Citation

If this repository is used in research, cite it as software and include the exact commit hash used for the reported results.

## License

Released under the MIT License. See [LICENSE](LICENSE).
