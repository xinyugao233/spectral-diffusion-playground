# Spectral Diffusion Playground

Frequency-resolved experiments for studying denoising and memorization in
diffusion models.

![EDM-1K spectral residual curves](figures/experiment_05/experiment_05_edm1k_low_high_residual_curves.png)

## Research Question

When a diffusion denoiser moves from noisy inputs toward clean images, how do
low- and high-frequency residual errors change, and are the resulting
transition windows especially influential for trajectory-level memorization?

This repository develops that question in six auditable steps. Experiments
E001-E003 establish the Fourier foundations. E004 selects an operational
CIFAR-10 frequency cutoff. E005 decomposes fixed-sigma denoising residual
energy into exact complementary bands. E006 intervenes on the resulting
windows by swapping the entire denoiser between matched EDM-1K and EDM-50K
models.

Frequency bands are measurement proxies, not semantic definitions. Low
frequency is not assumed to mean understanding, and high frequency is not
assumed to mean memorization.

## Connection To The Paper

Experiments E004-E006 are a paper-derived clean-room extension of
[*Two Calm Ends and the Wild Middle: A Geometric Picture of Memorization in
Diffusion Models*](https://arxiv.org/abs/2602.17846).

The paper motivates fixed-sigma denoising error and whole-denoiser swaps. This
repository adds an orthogonal Fourier decomposition of the fixed-sigma
residual, freezes transition windows before swap evaluation, and tests those
windows with matched clean-room models.

The original executed paper evaluator, swap implementation, checkpoint
identities, subset ordering, and sampling seeds were unavailable. E004-E006
therefore do **not** claim code identity or exact numerical reproduction of the
paper.

## Key Findings

- **E004:** A disclosed single-reviewer visual decision selected the
  operational CIFAR-10 cutoff `r = 4`, with `r = 3, 5` retained for primary
  sensitivity analysis and `r = 6` as an optional extended check.
- **E005:** At `r = 4`, the EDM-1K test residual showed an ordered transition:
  low-frequency residual energy changed over indices `5..11`
  (`sigma = 12.9101..0.585348`), followed by high-frequency residual energy
  over indices `11..14` (`sigma = 0.585348..0.0599473`).
- **E006:** The formal outcome is **`INCONCLUSIVE`** because the EDM-50K
  no-swap baseline was degenerate at `0/256` memorized samples under the frozen
  decision rule.
- **E006 descriptive finding:** The low-frequency transition window was the
  tested window most strongly associated with changes in the pixel-space
  memorization criterion. It passed the frozen influence test in both swap
  directions; the high-frequency transition window passed in neither.

E006 does not support assigning a causal memorization label to any sigma
interval.

![E006 transition windows versus controls](figures/experiment_06/experiment_06_transition_vs_controls.png)

## Experiment Roadmap

| ID | Purpose | Main artifact | Status | Result |
| --- | --- | --- | --- | --- |
| E001 | Explain the reversible image-to-Fourier transformation | [FFT visualization](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png) | Complete | The inverse FFT reconstructs the input to numerical precision |
| E002 | Show how Gaussian noise changes spectral content | [Noise/frequency grid](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png) | Complete | White noise raises energy broadly across spatial frequencies |
| E003 | Establish exact complementary low/high projections | [Frequency decomposition](figures/where_image_information_lives_grid.png) | Complete | Low- and high-band reconstructions sum to the original image |
| E004 | Select an operational cutoff on frozen CIFAR-10 examples | [Decision record](docs/experiment_04_frequency_cutoff_decision.md) | Complete | Reference `r = 4`; sensitivity `r = 3, 5`; optional `r = 6` |
| E005 | Split the paper-derived fixed-sigma residual into orthogonal band energies | [Residual-curve results](docs/experiment_05_spectral_residual_results.md) | Complete | Low-band transition precedes the high-band transition at `r = 4` |
| E006 | Test whole-denoiser swaps over E005 windows and matched controls | [Swap results](docs/experiment_06_transition_window_swap_results.md) | Complete; `INCONCLUSIVE` | Low-transition influence is descriptively strong, but baseline degeneracy blocks a directional conclusion |

## Methods In Brief

E005 applies complementary Fourier projections directly to the denoising
residual

```text
e_sigma = m_sigma(X + sigma Z) - X
```

and measures

```text
E_full = ||e_sigma||_2^2
E_low  = ||P_low,r e_sigma||_2^2
E_high = ||P_high,r e_sigma||_2^2
```

The channelwise 2D FFT uses `norm="ortho"`; the centered high-frequency mask
is the exact complement of the low-frequency mask. Consequently,
`E_full = E_low + E_high` holds within the frozen numerical tolerance.

E006 uses a pure 18-call Euler sampler and swaps the **whole denoiser** during
predeclared index windows. It does not splice frequency components of model
outputs. Memorization is evaluated in unquantized `[-1, 1]` RGB pixel space
using the strict criterion `d1NN < d2NN / 3` against the frozen clean-room
CIFAR-10 1K subset.

## Documentation And Artifacts

### E004: Operational frequency cutoff

- [Frozen protocol](docs/experiment_04_frequency_cutoff_protocol.md)
- [Reviewer instructions](docs/experiment_04_reviewer_instructions.md)
- [Final decision](docs/experiment_04_frequency_cutoff_decision.md)
- [Machine-readable results](results/README.md#e004-operational-frequency-cutoff)
- [Canonical montages](figures/README.md#e004-operational-frequency-cutoff)

### E005: Spectral residual curves

- [Frozen protocol](docs/experiment_05_spectral_residual_protocol.md)
- [Clean-room model provenance](docs/experiment_05_clean_room_models.md)
- [Validated results](docs/experiment_05_spectral_residual_results.md)
- [Compact results](results/experiment_05/)
- [Figures](figures/experiment_05/)

### E006: Transition-window swaps

- [Frozen protocol](docs/experiment_06_transition_swap_protocol.md)
- [Validated results](docs/experiment_06_transition_window_swap_results.md)
- [Compact results](results/experiment_06/)
- [Figures](figures/experiment_06/)

See the [documentation index](docs/README.md),
[results index](results/README.md), and [figures index](figures/README.md) for
the complete navigation map.

## Reproduction

Python 3.11 or newer is required for the reusable local experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the educational foundations independently from the repository root:

```bash
python experiments/01_fft_visualization.py
python experiments/02_noise_vs_frequency.py
python experiments/03_frequency_decomposition.py
```

Generate the deterministic E004 review packet from an existing
torchvision-compatible CIFAR-10 root:

```bash
python experiments/04_frequency_cutoff.py --dataset-root /path/to/cifar10
```

E005 and E006 require the frozen external CIFAR-10 archive, matched EDM
checkpoints, and the recorded Hellbender environment. Their exact hashes,
paths, configurations, and Slurm commands are recorded in the
[E005 model provenance](docs/experiment_05_clean_room_models.md),
[E005 results](docs/experiment_05_spectral_residual_results.md), and
[E006 protocol](docs/experiment_06_transition_swap_protocol.md). No script
downloads data or checkpoints implicitly.

Run repository validation with:

```bash
python -m unittest discover tests
git diff --check
```

## Reproducibility And Provenance

Scientific choices were frozen before the corresponding evaluations. Important
commits include:

| Milestone | Commit |
| --- | --- |
| E004 cutoff implementation | `a745cf1805deea0691fc3c43a591315b8a63984a` |
| E004 cutoff decision | `59b558e` |
| E005 evaluator | `b16c3a9c8224755cc2a5a52b0f1aacff44a63da7` |
| E005 results | `52d6889` |
| E006 frozen protocol | `068c7e3a745fb51b1d2416524b7e29f70b0b5f08` |
| E006 executed implementation | `ae0febb9b983c50c5946d61423fda72358887523` |
| E006 results | `df06e4fe3d9350988a5882b8d17db45c8ef6645f` |

Frozen model identities:

```text
EDM-1K SHA-256:
8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1

EDM-50K SHA-256:
a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
```

Only compact summaries, validation records, manifests, and final figures are
committed. This keeps review and cloning practical while preserving exact
provenance. Large per-sample artifacts remain on the research storage system:

```text
E005: /home/xggh8/data/zw-lab/e005_spectral_residual_curves
E006: /home/xggh8/data/zw-lab/e006_transition_window_swaps
```

Their identities and reproduction commands are recorded in the committed run
manifests and result documents. Raw generated samples and per-sample CSV files
must not be added to Git.

## Repository Layout

```text
spectral-diffusion-playground/
├── assets/       # deterministic examples and documented image provenance
├── configs/      # frozen E005/E006 execution configurations
├── data/         # small versioned manifests, never downloaded datasets
├── docs/         # protocols, provenance records, and result narratives
├── experiments/  # independently executable E001-E006 entry points
├── figures/      # curated, reviewable figures
├── results/      # compact machine-readable outputs
├── scripts/      # guarded preflight and Slurm launchers
├── src/          # reusable FFT, filtering, evaluation, and plotting code
└── tests/        # numerical identities, determinism, schemas, and safeguards
```

## Limitations

- E004 used one disclosed qualitative reviewer; the planned two-reviewer
  scoring procedure was not completed.
- The cutoff is operational and CIFAR-10-specific, not a universal semantic
  boundary between structure and detail.
- E005 transition windows come from the clean-room EDM-1K test residual curves
  and depend on the frozen schedule and cutoff family.
- E006 uses 256 seeds and a strict pixel-space nearest-neighbor criterion. Its
  EDM-50K baseline was exactly zero, triggering the frozen degeneracy guard.
- E004-E006 are paper-derived clean-room experiments, not exact reproductions
  of the paper's unavailable executed code.

## Citation And License

If you use this repository, cite it as software with the exact Git commit and
cite the grounding paper separately. The code is released under the
[MIT License](LICENSE).
