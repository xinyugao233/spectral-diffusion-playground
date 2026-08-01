# Experiment Entry Points

Each numbered script answers one narrow question and is executable from the
repository root. Reusable numerical and plotting logic belongs in
`src/spectral_diffusion_playground/`; scripts orchestrate frozen inputs and
write explicit outputs.

| Script | Role | Status |
| --- | --- | --- |
| `01_fft_visualization.py` | Reversible image/Fourier visualization | Complete |
| `02_noise_vs_frequency.py` | Gaussian noise and radial spectral energy | Complete |
| `03_frequency_decomposition.py` | Complementary low/high reconstruction | Complete |
| `04_frequency_cutoff.py` | Deterministic CIFAR-10 cutoff review packet | Complete |
| `04a_paper_geometry_curves.py` | Compute, validate, or plot the paper-derived clean-room coverage/concentration baseline | Complete |
| `05_spectral_residual_curves.py` | Orthogonal fixed-sigma residual energies | Complete |
| `06_transition_window_swaps.py` | Whole-denoiser transition-window swaps | Complete; formal outcome `INCONCLUSIVE` |
| E007 protocol only | Geometry-aligned whole-denoiser swaps | Proposed; not executed |

## Local Foundations

```bash
python experiments/01_fft_visualization.py
python experiments/02_noise_vs_frequency.py
python experiments/03_frequency_decomposition.py
python experiments/04_frequency_cutoff.py --dataset-root /path/to/cifar10
python experiments/04a_paper_geometry_curves.py
python experiments/04a_paper_geometry_curves.py --validate-only
python experiments/04a_paper_geometry_curves.py --compute \
  --dataset-root /path/to/cifar10 \
  --output-dir results/experiment_04a_reproduction \
  --device auto
python experiments/04a_paper_geometry_curves.py --compute-e006-grid \
  --dataset-root /path/to/cifar10 \
  --device auto
```

## Model Experiments

E005 and E006 require the frozen external archive, clean-room checkpoints, and
recorded execution environment. Use the guarded Slurm launchers only after
verifying every hash and output-path collision gate:

```text
scripts/e005_eval_spectral_residuals.slurm
scripts/e006_eval_transition_swaps.slurm
```

Exact commands and identities are in the
[E005 results](../docs/experiment_05_spectral_residual_results.md) and
[E006 protocol](../docs/experiment_06_transition_swap_protocol.md).

E007 is protocol-only and has not been executed. Its proposed geometry-aligned
window is documented in
[`docs/experiment_07_geometry_aligned_swap_protocol.md`](../docs/experiment_07_geometry_aligned_swap_protocol.md).

## Output Discipline

- Write compact numerical summaries under `results/`.
- Write final figures under `figures/`.
- Keep downloaded datasets, checkpoints, generated samples, and large raw
  tables outside Git.
- Never adjust frozen cutoffs, windows, seeds, or decision rules after seeing
  model results.
