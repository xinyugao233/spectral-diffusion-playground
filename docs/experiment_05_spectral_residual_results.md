# Experiment 5: Spectral Residual Curve Results

## Status

**Completed as a paper-derived clean-room reimplementation.**

The original executed paper evaluator, exact paper checkpoint identities,
CIFAR-10 1K permutation, test selection, and random seeds were unavailable.
This result follows the frozen clean-room protocol and matched clean-room model
pair. It is not a code-identical or numerically exact reproduction of the
paper.

Experiment 6 has not started. These transition windows are descriptive
fixed-sigma residual-energy summaries, not memorization danger zones.
Memorization relevance remains an E006 hypothesis.

## Run Identity

```text
implementation commit: b16c3a9c8224755cc2a5a52b0f1aacff44a63da7
Slurm job:             15425473
Slurm state:           COMPLETED
exit code:             0:0
elapsed:               00:22:10
node:                  g040
remote output:         /home/xggh8/data/zw-lab/e005_spectral_residual_curves
```

Frozen checkpoints:

```text
EDM-1K:
/home/xggh8/data/zw-lab/exp_004_standard_edm_n1000_40000kimg_20260415/network-snapshot-040000.pkl
SHA-256: 8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1

EDM-50K:
/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/network-snapshot-040000.pkl
SHA-256: a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
```

## Imported Artifacts

The repository keeps compact, reviewable results:

```text
results/experiment_05/experiment_05_aggregated_curves.csv
results/experiment_05/experiment_05_identity_validation.json
results/experiment_05/experiment_05_manifest.json
results/experiment_05/experiment_05_transition_windows.json

figures/experiment_05/experiment_05_edm1k_low_high_residual_curves.png
figures/experiment_05/experiment_05_edm50k_low_high_residual_curves.png
figures/experiment_05/experiment_05_train_test_comparison.png
figures/experiment_05/experiment_05_cutoff_sensitivity.png
figures/experiment_05/experiment_05_transition_windows.png
figures/experiment_05/experiment_05_additivity_diagnostics.png
```

The full raw per-sample CSV is intentionally not committed:

```text
path:    /home/xggh8/data/zw-lab/e005_spectral_residual_curves/experiment_05_bandwise_residuals.csv
rows:    2,304,000
SHA-256: 2600a81bb6f1d8d5bc442dd6c9147629d61231bb958a28bf7a5e704de38ffe88
```

Other remote artifact hashes:

```text
aggregated curves SHA-256:  0b19d3f1209d752e1c2df3e5f3e1e46678f3100b04e690cf1fd9f6f50f48f220
identity report SHA-256:   911126ea61ddc02a6440762df9066117b49703c5b3960b8d60032553f4c3a940
transition JSON SHA-256:   aa588d071716e81694ea467f282947cc9949834ffdc8011abe847d0344cbd6bf
run manifest SHA-256:      3d5b1ade232eee5ba80150e35c53b2f33eb354c7b6bb06a21bdf39dfc34c99e8
```

Reproduction command:

```bash
E005_REPO_ROOT=/cluster/pixstor/zwggh-lab/xinyu/projects/spectral-diffusion-playground \
E005_REPO_COMMIT=b16c3a9c8224755cc2a5a52b0f1aacff44a63da7 \
sbatch --export=ALL,E005_REPO_ROOT,E005_REPO_COMMIT \
  scripts/e005_eval_spectral_residuals.slurm \
  /home/xggh8/data/zw-lab/e005_spectral_residual_curves full
```

## Numerical Validation

The identity report passed:

```text
expected rows:                     2,304,000
observed rows:                     2,304,000
nonfinite rows:                    0
failed rows:                       0
max reconstruction error:          1.7763568394002505e-15
max additivity absolute error:     3.922195901395753e-12
max additivity relative error:     1.3215668922493663e-15
max orthogonality relative error:  2.9090600661927896e-17
```

The required residual-energy identity

```text
E_full = E_low + E_high
```

holds within the frozen tolerance. Scans of the imported aggregate table and
external raw table found no NaN or nonfinite numeric values.

## Primary Findings

Primary transition extraction uses `edm_1k/test` at the reference cutoff
`r=4`.

Low-frequency residual energy — general-structure proxy:

```text
transition indices:        5..11
sigma window:              12.9101 to 0.585348
adjacent-cutoff stable:    true
```

High-frequency residual energy — fine-detail proxy:

```text
transition indices:        11..14
sigma window:              0.585348 to 0.0599473
adjacent-cutoff stable:    true
```

Low-frequency residual recovery transitions at higher noise. High-frequency
residual recovery transitions later at lower noise. The shared endpoint near
`sigma=0.585` is a descriptive coarse-to-fine handoff in the fixed-sigma
residual-energy curves.

## Cutoff Sensitivity

Low-frequency residual transition windows:

| Cutoff | Status | Indices | Sigma Window |
| --- | --- | --- | --- |
| `r=3` | `ok` | `5..10` | `12.9101` to `1.08817` |
| `r=4` | `ok` | `5..11` | `12.9101` to `0.585348` |
| `r=5` | `ok` | `5..11` | `12.9101` to `0.585348` |
| `r=6` | `ok` | `5..11` | `12.9101` to `0.585348` |

High-frequency residual transition windows:

| Cutoff | Status | Indices | Sigma Window |
| --- | --- | --- | --- |
| `r=3` | `ok` | `11..13` | `0.585348` to `0.139516` |
| `r=4` | `ok` | `11..14` | `0.585348` to `0.0599473` |
| `r=5` | `ok` | `12..14` | `0.296442` to `0.0599473` |
| `r=6` | `ok` | `12..14` | `0.296442` to `0.0599473` |

No transition result is `no_clear_transition`.

## Interpretation

The fixed EDM-1K denoiser shows an ordered residual-energy transition under
the frozen clean-room protocol: low-frequency residual energy falls through
its transition window at higher noise, while high-frequency residual energy
falls later at lower noise. This supports using the E005 windows as candidate
transition windows for the next whole-denoiser swap experiment.

This does not establish when a model learned either band. It also does not
show memorization. E006 must test whether whole-denoiser swaps around these
windows affect trajectory-level memorization behavior under the clean-room
setup.

## Limitations

- The result is clean-room and paper-derived, not an exact reproduction.
- The E004 cutoff was selected by a single-reviewer qualitative decision, not
  a completed two-reviewer scoring protocol.
- Frequency bands are operational proxies; they are not semantic definitions.
- The primary transition windows come from EDM-1K held-out test curves only,
  by protocol.
- Raw per-sample rows are stored externally because the CSV is large.
