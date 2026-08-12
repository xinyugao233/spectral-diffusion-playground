# Spectral Diffusion Playground

A research portfolio using frequency-resolved data geometry to identify
**when along a diffusion trajectory an intervention can change memorization**.
The central result is a controlled whole-denoiser swap at geometry-selected
noise levels on CIFAR-10.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests: 206 passing](https://img.shields.io/badge/tests-206%20passing-2ea44f)](#reproduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-0b7285)](LICENSE)

**Quick links:** [Main result](#4-main-result) ·
[Method](#the-experiment-in-one-picture) · [Reproduce](#reproduction) ·
[Full documentation](docs/README.md)

## 4. Main Result

> **Swapping an empirically generalizing denoiser into a memorizing trajectory
> at the high-frequency-derived sampled noise levels
> `sigma = {1.9233, 1.0882}` reduced final memorization to `55.08%`, compared
> with `65.23%` and `66.41%` at the neighboring width-matched controls.**

![Low- and high-frequency-derived whole-denoiser swap results](figures/experiment_10/low_high_directional_swap_table.png)

[Open the publication-quality PDF](figures/experiment_10/low_high_directional_swap_table.pdf)

The high target's suppression effect was `28.91%`, versus `18.75%` and
`17.58%` for its controls. The target-minus-mean-control contrast was `10.74%`
with paired bootstrap 95% CI `[6.84%, 14.84%]`.

**Supported:** `HIGH_DERIVED_SUPPRESSION_SUPPORTED`

The low-derived candidate did not pass the preregistered influence criterion.
The reverse induction direction was floor-limited because the EDM-50K
recipient had `0/256` memorized samples under the no-swap baseline.

![Target-minus-control contrasts with paired confidence intervals](figures/experiment_10/target_control_contrasts.png)

## The Experiment In One Picture

```text
E001-E004: freeze the Fourier split at r = 4
                         |
              +----------+----------+
              |                     |
        low: r <= 4            high: r > 4
              |                     |
       C_low(sigma),           C_high(sigma),
       W_low(sigma)            W_high(sigma)
              |                     |
       sigma = 3.2568       sigma = 1.9233, 1.0882
              +----------+----------+
                         |
        E010: swap the whole denoiser at those sigma values
                         |
        only high-derived suppression is supported
```

In experiment notation:

```text
-> E004: freeze r = 4
-> E004B: low/high coverage and posterior geometry
-> E010: whole-denoiser swaps
-> high-derived suppression supported
```

Fourier radius `r` and diffusion noise `sigma` play different roles. Radius
defines the measurement subspace; sigma locates the intervention along the
denoising trajectory. The intervention swaps the **whole denoiser**, not an
individual Fourier component.

## 1. Separate Coarse And Fine Frequencies

E001-E003 establish the Fourier representation, show how Gaussian noise
changes spectral content, and verify exact complementary low/high projections.
E004 then freezes `r = 4` as an operational CIFAR-10 split, with sensitivity
checks at `r = 3,5`.

![Coarse-to-fine Fourier reconstruction across increasing radii](figures/where_image_information_lives_grid.png)

- **Low frequency:** `r <= 4`, retaining broad, slowly varying structure.
- **High frequency:** `r > 4`, the exact complementary residual containing
  progressively finer variation.

This is a documented measurement choice, not a universal semantic boundary.
The low- and high-pass reconstructions sum back to the original image.

**Foundation experiments:**
[E001 FFT](experiments/01_fft_visualization.py) ·
[E002 noise spectra](experiments/02_noise_vs_frequency.py) ·
[E003 decomposition](experiments/03_frequency_decomposition.py) ·
[E004 cutoff decision](docs/experiment_04_frequency_cutoff_decision.md)

## 2. Frequency Geometry Selects Candidate Sigma Locations

Following the paper's geometric picture, the project measures:

- **Coverage `C_sigma`:** how much noisy held-out data lies inside regions
  covered by training-centered Gaussian shells.
- **Maximum posterior weight `W_sigma`:** how strongly the empirical posterior
  concentrates on a single training example.

The full-space E004A baseline reconstructs this geometry in a deterministic
clean-room setting. Its assumptions, definitions, numerical validation, and
reproduction evidence are recorded in the
[source audit](docs/paper_geometry_source_audit.md),
[protocol](docs/experiment_04a_paper_geometry_protocol.md), and
[results](docs/experiment_04a_paper_geometry_results.md).

E004B then evaluates the geometry independently inside the frozen Fourier
split. It draws two curves in the low-frequency subspace and two curves in the
high-frequency subspace:

```text
E004B = frequency-restricted data geometry
E005  = frequency-restricted denoising residual energy

C_low_sigma(p,D)      W_low_sigma(D)
C_high_sigma(p,D)     W_high_sigma(D)
```

![Frequency-restricted geometry and frozen candidate sigma locations](figures/experiment_10/geometry_targets.png)

On the frozen 18-point schedule, a sampled level is selected only when the 95%
lower confidence bounds for both quantities reach the preregistered
`q_C=q_W=0.8` thresholds.

| Subspace | Fourier band | Candidate sigma | Sampler calls |
| --- | --- | ---: | ---: |
| Low | `r <= 4` | `3.2568` | `{8}` |
| High | `r > 4` | `1.9233, 1.0882` | `{9,10}` |

The low candidate is one sampled point. The high candidate is two sampled
levels, not a claim that every continuous sigma between them passed the rule.
Their exact real projector ranks, `147` and `2925`, differ sharply. E004B
therefore does not isolate frequency from
subspace dimension, covariance, or energy structure.
Coverage alone does not exhibit the same ordering as the joint
coverage-and-concentration candidates.

Full details: [E004B protocol](docs/experiment_04b_frequency_restricted_geometry_protocol.md) ·
[E004B results](docs/experiment_04b_frequency_restricted_geometry_results.md)

## 3. Test The Candidates With Whole-Denoiser Swaps

For each geometry-derived candidate, E010 temporarily replaces the recipient's
entire denoiser with a donor and then resumes the recipient immediately after
the listed sigma values:

```text
recipient trajectory
        -> enter frozen candidate sigma value(s)
        -> donor replaces the whole denoiser
        -> recipient resumes
        -> evaluate final-sample memorization
```

Each target is compared with preregistered neighboring controls using the same
256 latent seeds. Sampler calls are implementation mappings only: the low
target is `i=8`, and the high target is `i={9,10}`.

The primary result figure is shown at the top of this page. The full
[E010 protocol](docs/experiment_10_directional_memorization_transfer_protocol.md),
[analysis plan](docs/experiment_10_directional_analysis_plan.md), and
[validated results](docs/experiment_10_directional_memorization_transfer_results.md)
preserve both suppression and induction directions.

## What This Does And Does Not Show

- It shows a selective timing association for one asymmetric model pair under
  a whole-denoiser intervention.
- It does not establish high-frequency-component or fine-detail causality;
  frequency geometry selects **when** to intervene, not **what component** is
  replaced.
- It does not establish dataset-size causality or a universal memorization
  danger zone.
- It does not establish a continuous danger interval between sparsely sampled
  sigma levels.
- It does not generalize beyond the tested models, sampler, seeds, and strict
  pixel-space memorization criterion.

## Supporting And Historical Experiments

These experiments document how the final E010 design was reached. They are
retained for scientific provenance but are not prerequisites for the main
argument.

| Experiment | Role | Outcome |
| --- | --- | --- |
| [E005](docs/experiment_05_spectral_residual_results.md) | Supporting residual-dynamics diagnostic | Coarse-to-fine pattern |
| [E006](docs/experiment_06_transition_window_swap_results.md) | Historical exploratory swap | `INCONCLUSIVE` |
| [E007](docs/experiment_07_geometry_aligned_swap_protocol.md) | Proposed full-space intervention | Blocked, not executed |
| [E008](docs/experiment_08_retirement_decision.md) | Symmetric-pair design | `RETIRED_UNEXECUTED` |
| [E009](docs/experiment_09_stage_b_results.md) | Larger-data pair search | No eligible 5K model through 30K kimg |

**E006** was formally `INCONCLUSIVE` and did not identify a
memorization danger zone. **E007-E009** document blocked, retired, or negative
paths rather than completed substitutes for E010.

## Reproduction

Python 3.11 or newer is required for the reusable local experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m unittest discover tests
```

Core local entry points:

```bash
python experiments/01_fft_visualization.py
python experiments/02_noise_vs_frequency.py
python experiments/03_frequency_decomposition.py
python experiments/04a_paper_geometry_curves.py --help
python experiments/04b_frequency_restricted_geometry.py --help
```

E004, E004A, and E004B require an existing CIFAR-10 root. Model-based
experiments require the frozen external archive and checkpoints; nothing
downloads datasets or model weights implicitly. Exact seeds, hashes, commits,
and cluster execution records are versioned in [`docs/`](docs/README.md) and
[`results/`](results/README.md).

## Repository Structure

```text
spectral-diffusion-playground/
├── assets/       # deterministic examples and image provenance
├── configs/      # frozen experiment configurations
├── docs/         # protocols, provenance, and result narratives
├── experiments/  # executable experiment entry points
├── figures/      # curated, reviewable figures
├── results/      # compact machine-readable outputs
├── scripts/      # guarded preflight and cluster launchers
├── src/          # reusable analysis and evaluation code
└── tests/        # numerical, schema, and safeguard tests
```

## Documentation

- **Core geometry:** [E004](docs/experiment_04_frequency_cutoff_decision.md) ·
  [E004A](docs/experiment_04a_paper_geometry_results.md) ·
  [E004B](docs/experiment_04b_frequency_restricted_geometry_results.md)
- **Main intervention:** [protocol](docs/experiment_10_directional_memorization_transfer_protocol.md) ·
  [analysis](docs/experiment_10_directional_analysis_plan.md) ·
  [results](docs/experiment_10_directional_memorization_transfer_results.md)
- **Historical experiments:** [E005-E009 documentation](docs/README.md)
- **Indexes:** [Docs](docs/README.md) · [Results](results/README.md) ·
  [Figures](figures/README.md)

## Citation And License

If you use this repository, cite it as software using [`CITATION.cff`](CITATION.cff)
and record the exact Git commit. Cite the grounding paper,
[*Two Calm Ends and the Wild Middle: A Geometric Picture of Memorization in
Diffusion Models*](https://arxiv.org/abs/2602.17846), separately.

Released under the [MIT License](LICENSE).
