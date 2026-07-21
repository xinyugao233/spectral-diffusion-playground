# Spectral Diffusion Playground

Spectral Diffusion Playground is a small research repository for understanding diffusion-model behavior through Fourier analysis, controlled perturbations, and clean visualizations.

**Current status:** repository scaffold only. The experiment entrypoints and shared package structure are in place, but the actual experiments are not implemented yet.

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

## Planned Experiments

| Script | Question | Planned output | Status |
| --- | --- | --- | --- |
| `01_fft_visualization.py` | What becomes obvious when an image is viewed through centered magnitude and phase plots? | Baseline FFT figures for reference images | [ ] |
| `02_noise_vs_frequency.py` | How do different noise realizations appear in the frequency domain? | Side-by-side spatial and spectral comparisons | [ ] |
| `03_sigma_progression.py` | How does increasing noise scale change spectral structure? | Multi-panel progression over sigma values | [ ] |
| `04_low_pass.py` | What survives aggressive removal of high frequencies? | Low-pass reconstructions and spectra | [ ] |
| `05_high_pass.py` | What is emphasized when low frequencies are suppressed? | High-pass reconstructions and residual views | [ ] |

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

At the current stage, the scripts intentionally act as stubs and print the question they will eventually answer.

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
- Study whether different noise schedules induce distinct spectral trajectories.
- Connect score estimation and denoising behavior to frequency recovery.
- Examine spectral effects of conditioning, guidance, or architecture choices.
- Use this repository as a base for paper-ready diagnostic figures.

## Citation

If this repository is used in research, cite it as software and include the exact commit hash used for the reported results.

## License

Released under the MIT License. See [LICENSE](LICENSE).
