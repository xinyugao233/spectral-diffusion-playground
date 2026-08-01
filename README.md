# Spectral Diffusion Playground

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests: 69 passing](https://img.shields.io/badge/tests-69%20passing-2ea44f)](#installation-and-reproduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-0b7285)](LICENSE)
[![Tag: portfolio-v1](https://img.shields.io/badge/tag-portfolio--v1-c2410c)](https://github.com/xinyugao233/spectral-diffusion-playground/tree/portfolio-v1)

A research portfolio of frequency-resolved experiments for understanding
denoising and memorization in diffusion models.

The central question is: **how do low- and high-frequency denoising residuals
change across noise levels, and are those transition windows especially
influential for trajectory-level memorization?**

Experiments E001-E003 build the Fourier foundation. E004-E006 form a
paper-derived clean-room extension of
[*Two Calm Ends and the Wild Middle: A Geometric Picture of Memorization in
Diffusion Models*](https://arxiv.org/abs/2602.17846).

## Key Visual Highlights

E005 reveals two ordered residual-energy transitions in a fixed denoiser: the
low-frequency residual changes at higher noise, followed by the high-frequency
residual at lower noise.

![EDM-1K low- and high-frequency residual-energy curves](figures/experiment_05/experiment_05_edm1k_low_high_residual_curves.png)

**Headline result:** under the frozen fixed-denoiser evaluation, low-frequency
residual recovery transitions at higher noise than high-frequency residual
recovery, producing an ordered coarse-to-fine residual pattern.

E006 then tests those windows with whole-denoiser swaps. Its formal outcome is
**`INCONCLUSIVE`** because the EDM-50K no-swap baseline is degenerate, while
the low-transition window remains the strongest descriptive association.

![E006 transition windows versus width-matched controls](figures/experiment_06/experiment_06_transition_vs_controls.png)

## Start Here

- **See the main measurement:** [E005 residual curves and transition windows](#e005-low--and-high-frequency-residual-transitions).
- **See the intervention and its limits:** [E006 whole-denoiser swaps](#e006-transition-window-whole-denoiser-swaps).
- **Audit the evidence:** [E005 result record](docs/experiment_05_spectral_residual_results.md)
  and [E006 result record](docs/experiment_06_transition_window_swap_results.md).
- **Reproduce the foundations locally:** [installation and commands](#installation-and-reproduction).

## Experiment Map

| Experiment | Question | Main figure | Result | Status |
| --- | --- | --- | --- | --- |
| E001 | What does a 2D Fourier transform represent? | [FFT visualization](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png) | Pixel-to-frequency conversion is reversible | Complete |
| E002 | How does Gaussian noise change spectral content? | [Noise/frequency grid](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png) | Noise raises energy broadly across frequency | Complete |
| E003 | Where does image information appear across frequency radii? | [Decomposition grid](figures/where_image_information_lives_grid.png) | Complementary low/high reconstructions exactly recover the image | Complete |
| E004 | Which cutoff is a useful operational CIFAR-10 split? | [Cutoff montage](figures/experiment_04_cutoff_montage_classes_0_1.png) | Reference `r = 4`; sensitivity `r = 3, 5` | Complete |
| E005 | When do low/high residual energies transition across noise levels? | [Two residual curves](figures/experiment_05/experiment_05_edm1k_low_high_residual_curves.png) | Low-frequency transition precedes high-frequency transition | Complete |
| E006 | Do swaps over those windows alter the memorization criterion? | [Swap/control chart](figures/experiment_06/experiment_06_transition_vs_controls.png) | Formal outcome `INCONCLUSIVE`; low-transition result is descriptively strongest | Complete |

## E001: Understanding Images In Fourier Space

**Question.** What information does a channelwise 2D Fourier transform expose,
and can the original image be recovered? E001 computes the centered FFT,
linear and log-magnitude spectra, and inverse FFT with orthonormal
normalization. The reconstruction matches the input up to numerical precision.

![Original image, Fourier spectra, and inverse reconstruction](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png)

- **Purpose:** Build intuition for the reversible pixel-space to frequency-space
  transformation.
- **Main artifact:** [RGB FFT visualization](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png)
  and [RGB/grayscale comparison](figures/understanding_images_in_fourier_space_default_fft_reference_rgb_and_grayscale.png).
- **Takeaway:** Fourier space reorganizes image variation by spatial frequency;
  it does not discard information by itself.

```bash
python experiments/01_fft_visualization.py
python experiments/01_fft_visualization.py --grayscale
```

## E002: How Gaussian Noise Changes Frequency Content

**Question.** How does additive white Gaussian noise alter an image and its
spectrum? E002 evaluates five frozen noise levels and compares pixel-space
corruption with log spectra and radial energy distributions. Increasing noise
raises energy broadly across spatial frequencies and makes the normalized
DC-excluded profile more uniform.

![Gaussian noise in pixel and frequency space](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png)

- **Purpose:** Connect pixel-space corruption to frequency-space behavior.
- **Main artifact:** [Noise/frequency grid](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png)
  and [normalized radial distribution](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_normalized_radial_distribution.png).
- **Takeaway:** White noise contributes broadly rather than occupying one
  narrow frequency band.

```bash
python experiments/02_noise_vs_frequency.py
```

## E003: Where Does Image Information Live?

**Question.** What becomes visible as progressively larger centered frequency
regions are retained? E003 applies nested circular low-pass masks and exact
complementary high-pass masks. The low-pass reconstruction and signed residual
sum back to the input, while the reconstruction-error curve quantifies how
rapidly image content is recovered.

![Low-pass masks and cumulative image reconstruction](figures/where_image_information_lives_grid.png)

![Complementary high-frequency residuals](figures/high_frequency_residuals.png)

- **Purpose:** Establish the complementary projection operators used later.
- **Main artifact:** [Reconstruction grid](figures/where_image_information_lives_grid.png),
  [high-frequency residuals](figures/high_frequency_residuals.png), and
  [error curve](figures/reconstruction_error_vs_frequency_radius.png).
- **Takeaway:** Increasing radius progressively restores image variation, but
  frequency bands remain operational measurements rather than semantic labels.

```bash
python experiments/03_frequency_decomposition.py
```

## E004: Selecting An Operational CIFAR-10 Cutoff

**Question.** Which centered radial cutoff provides a useful and auditable split
on 32 x 32 CIFAR-10 images? E004 freezes 20 examples, six candidate radii, exact
display conventions, and reconstruction/energy diagnostics before review. A
single disclosed reviewer selected `r = 4` as the smallest visually acceptable
cutoff; `r = 3` and `r = 5` remain the primary sensitivity cutoffs.

![Representative E004 cutoff montage for CIFAR-10 classes 0 and 1](figures/experiment_04_cutoff_montage_classes_0_1.png)

- **Purpose:** Freeze an operational frequency split before examining denoiser
  curves.
- **Main artifact:** [Five class-grouped cutoff montages](figures/README.md#e004-operational-frequency-cutoff)
  and the [decision record](docs/experiment_04_frequency_cutoff_decision.md).
- **Takeaway:** `r = 4` is a documented experimental choice, not a universal
  boundary between structure and detail.

The originally planned two-independent-reviewer scoring procedure was not
completed. No inter-rater or blinded-review claim is made.

## E005: Low- And High-Frequency Residual Transitions

**Question.** How does the paper-derived fixed-sigma denoising residual divide
between complementary frequency bands? E005 projects the residual directly,
uses float64 spectral calculations after float32 inference, and verifies
`E_full = E_low + E_high` numerically. This is the central two-curve diagnostic
for the repository.

For a clean image `X`, Gaussian noise `Z`, noise level `sigma`, and fixed
denoiser `m_sigma`, the residual is

```text
e_sigma = m_sigma(X + sigma Z) - X
```

E005 measures its summed squared magnitude in two exact complementary Fourier
bands:

```text
E_low(sigma)  = ||P_low,r e_sigma||_2^2
E_high(sigma) = ||P_high,r e_sigma||_2^2
```

Here, **residual energy** is the summed squared error remaining after denoising,
projected into either the low- or high-frequency band. The curves are not
image-quality scores or memorization rates; they measure how much denoising
error remains in each band at each fixed noise level.

![EDM-1K low- and high-frequency residual-energy curves](figures/experiment_05/experiment_05_edm1k_low_high_residual_curves.png)

![EDM-50K low- and high-frequency residual-energy curves](figures/experiment_05/experiment_05_edm50k_low_high_residual_curves.png)

### How The Transition Windows Are Extracted

The rule is applied **independently** to each aggregated low- and
high-frequency residual-energy curve on the frozen 18-point sigma grid. It
does not use the intersection of the two curves. The grid is traversed from
large sigma (high noise) to small sigma (low noise), without smoothing or
interpolation.

For one band, let `E_i` be its aggregated residual energy at grid index `i`.
The **high-noise endpoint** is the median of indices `0, 1, 2`; the
**low-noise endpoint** is the median of indices `15, 16, 17`. These endpoints
convert energy into a normalized recovery quantity:

```text
E_high-noise = median(E_0, E_1, E_2)
E_low-noise  = median(E_15, E_16, E_17)

R_i = (E_high-noise - E_i)
      / (E_high-noise - E_low-noise)

entry = first two consecutive points with R_i >= 0.20
exit  = first two consecutive points with R_i >= 0.80
```

Approximately, `R = 0` represents the high-noise residual level and `R = 1`
the low-noise residual level. Values are not clipped, so endpoint overshoot
remains visible. The **transition entry** is the first index beginning a pair
of consecutive values at or above `0.20`. Starting from that entry, the
**transition exit** is the first index beginning a pair at or above `0.80`.
The inclusive entry-to-exit interval is the extracted window.

The implementation returns `no_clear_transition` for a nonfinite curve, a
nonpositive or nonfinite endpoint difference, a missing crossing, an exit
before entry, a zero-width window, or a later two-point recrossing below either
threshold. It never widens or manually moves a failed window. At the reference
cutoff `r = 4`, entry and exit must each be within one grid index of the
corresponding `r = 3` and `r = 5` boundaries to be marked
`adjacent_cutoff_stable=true`.

![Low- and high-frequency transition windows extracted with the frozen 20%-to-80% rule](figures/experiment_05/experiment_05_transition_windows.png)

The trajectory proceeds from large sigma to small sigma. At the reference
cutoff `r = 4`, the frozen EDM-1K test result is:

| Band | Transition indices | Sigma window | Adjacent-cutoff stable |
| --- | --- | --- | --- |
| Low-frequency residual | `5..11` | `12.9101..0.585348` | `true` |
| High-frequency residual | `11..14` | `0.585348..0.0599473` | `true` |

The low-frequency transition therefore occurs first, at higher sigma; the
high-frequency transition follows later, at lower sigma. Their shared boundary
near `sigma = 0.585348` is a descriptive coarse-to-fine handoff under this
fixed-denoiser measurement. These are fixed-sigma denoising residual
transitions, not training-time learning transitions.

```mermaid
flowchart LR
    A[Fixed-sigma denoising residual] --> B[Low/high Fourier projection]
    B --> C[Two residual-energy curves]
    C --> D[Independent 20%-80% transition extraction]
    D --> E[Low and high candidate windows]
    E --> F[Whole-denoiser swap experiment]
    F --> G[Final memorization criterion]
```

- **Purpose:** Turn the structure/detail intuition into exact, complementary
  residual-energy measurements.
- **Main artifact:** [Two-curve figures](figures/experiment_05/),
  [transition-window visualization](figures/experiment_05/experiment_05_transition_windows.png),
  and [validated results](docs/experiment_05_spectral_residual_results.md).
- **Takeaway:** Low-frequency residual recovery transitions earlier and
  high-frequency residual recovery transitions later under this fixed-denoiser
  setup. This is a descriptive coarse-to-fine residual pattern, not a learning
  dynamic or memorization result.

### From Transition Windows To A Candidate Danger Zone

E005 identifies candidate trajectory intervals from residual-energy dynamics;
by itself, it does **not** identify a memorization danger zone. E006 tests the
low- and high-transition intervals by swapping the whole denoiser between
EDM-1K and EDM-50K and comparing each interval with width-matched controls.

The low-frequency transition window is the strongest candidate danger-zone
interval in the present experiment. This is a descriptive trajectory-level
result, not proof that low-frequency features themselves cause memorization.
The high-transition interval did not pass the frozen influence criterion. The
formal E006 outcome remains **`INCONCLUSIVE`** because the EDM-50K no-swap
baseline was degenerate at `0/256`.

E006 swaps the **whole denoiser** during a frequency-aligned time window. It
does not swap only the low-frequency or high-frequency component of the
denoiser output. Neither the space between the curves nor their intersection
defines a candidate interval.

## E006: Transition-Window Whole-Denoiser Swaps

**Question.** Are the E005 transition windows more influential for final
trajectory-level memorization than width-matched controls? E006 runs whole-
denoiser swaps in both EDM-1K/EDM-50K directions with the same 256 latent seeds
and evaluates the strict pixel-space criterion `d1NN < d2NN / 3`.

![E006 transition windows and width-matched controls](figures/experiment_06/experiment_06_transition_vs_controls.png)

The generated/nearest-neighbor pairs below make the strict pixel-space
memorization criterion visually concrete. They are examples from the frozen
evaluation, not a substitute for the aggregate decision rule.

![Representative E006 generated samples and nearest training neighbors](figures/experiment_06/experiment_06_generated_nn_pairs.png)

- **Purpose:** Test whether the E005 windows matter under a causal model-swap
  intervention.
- **Main artifact:** [Transition/control chart](figures/experiment_06/experiment_06_transition_vs_controls.png),
  [memorization-rate chart](figures/experiment_06/experiment_06_memorization_rates.png),
  [paired-change chart](figures/experiment_06/experiment_06_paired_changes.png),
  and [validated results](docs/experiment_06_transition_window_swap_results.md).
- **Takeaway:** The formal outcome is **`INCONCLUSIVE`** because the EDM-50K
  baseline is degenerate at `0/256`. Descriptively, the low-transition window
  passes the frozen influence criterion in both swap directions; the
  high-transition window passes in neither.

The descriptive result does not override the frozen outcome or establish a
causal memorization interpretation for any sigma interval.

E006 swaps the **whole denoiser** during windows aligned with the E005
frequency-band transitions; it does not intervene on only the low- or
high-frequency part of a denoiser output. The justified descriptive statement
is therefore about the trajectory interval, not a frequency-specific causal
mechanism.

## Mathematical Core

E005 applies complementary Fourier projections directly to

```text
e_sigma = m_sigma(X + sigma Z) - X
```

and measures

```text
E_full = ||e_sigma||_2^2
E_low  = ||P_low,r e_sigma||_2^2
E_high = ||P_high,r e_sigma||_2^2
```

The channelwise 2D FFT uses `norm="ortho"`; the high-frequency mask is the exact
complement of the centered low-frequency mask. The resulting band energies sum
to the full residual energy within the frozen tolerance.

## Documentation And Results

| Experiment | Protocol and result narrative | Compact results | Figures |
| --- | --- | --- | --- |
| E004 | [Protocol](docs/experiment_04_frequency_cutoff_protocol.md) · [Reviewer instructions](docs/experiment_04_reviewer_instructions.md) · [Decision](docs/experiment_04_frequency_cutoff_decision.md) | [Results index](results/README.md#e004-operational-frequency-cutoff) | [Montages](figures/README.md#e004-operational-frequency-cutoff) |
| E005 | [Protocol](docs/experiment_05_spectral_residual_protocol.md) · [Model provenance](docs/experiment_05_clean_room_models.md) · [Results](docs/experiment_05_spectral_residual_results.md) | [`results/experiment_05/`](results/experiment_05/) | [`figures/experiment_05/`](figures/experiment_05/) |
| E006 | [Protocol](docs/experiment_06_transition_swap_protocol.md) · [Results](docs/experiment_06_transition_window_swap_results.md) | [`results/experiment_06/`](results/experiment_06/) | [`figures/experiment_06/`](figures/experiment_06/) |

See the [documentation index](docs/README.md), [results index](results/README.md),
and [figures index](figures/README.md) for the complete navigation map.

## Installation And Reproduction

Python 3.11 or newer is required for the reusable local experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m unittest discover tests
```

E004 requires an existing torchvision-compatible CIFAR-10 root. E005 and E006
require the frozen external archive, matched EDM checkpoints, and recorded
Hellbender environment. Exact hashes, configurations, paths, and Slurm commands
are recorded in the E005/E006 protocol and result documents. No experiment
downloads data or checkpoints implicitly.

## Reproducibility And Provenance

Scientific choices were frozen before evaluation. Key commits are:

| Milestone | Commit |
| --- | --- |
| E004 cutoff implementation | `a745cf1805deea0691fc3c43a591315b8a63984a` |
| E004 cutoff decision | `59b558e` |
| E005 evaluator | `b16c3a9c8224755cc2a5a52b0f1aacff44a63da7` |
| E005 results | `52d6889` |
| E006 protocol | `068c7e3a745fb51b1d2416524b7e29f70b0b5f08` |
| E006 executed implementation | `ae0febb9b983c50c5946d61423fda72358887523` |
| E006 results | `df06e4fe3d9350988a5882b8d17db45c8ef6645f` |

Frozen checkpoint hashes:

```text
EDM-1K:  8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1
EDM-50K: a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
```

Only compact summaries, manifests, validation records, and final figures are
committed. Large raw artifacts remain external so the repository stays
reviewable and practical to clone:

```text
E005: /home/xggh8/data/zw-lab/e005_spectral_residual_curves
E006: /home/xggh8/data/zw-lab/e006_transition_window_swaps
```

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
- The cutoff is CIFAR-10-specific and operational, not a universal semantic
  boundary.
- E005 transition windows depend on the frozen clean-room model, schedule, and
  cutoff family; they do not identify when a model learned either band.
- E006 uses 256 seeds and a strict pixel-space criterion. Its degenerate
  EDM-50K baseline triggers the frozen safeguard and prevents a directional
  conclusion.
- E004-E006 are paper-derived clean-room experiments, not exact reproductions
  of the paper's unavailable executed code.

## Citation And License

If you use this repository, cite it as software using [`CITATION.cff`](CITATION.cff)
and record the exact Git commit. Cite the grounding paper separately. Released
under the [MIT License](LICENSE).
