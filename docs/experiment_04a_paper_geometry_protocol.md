# E004A: Paper Coverage-Concentration Geometry Protocol

## Objective

Reproduce the paper's two original geometric quantities before comparing them
with the repository's separate spectral residual analysis:

```text
maximum posterior weight W_sigma(D)
Gaussian-shell coverage C_sigma(p,D)
```

E004A is inserted without renumbering E004-E006. It is a paper-derived
clean-room baseline, not an exact reproduction.

## Frozen Data Domain

- CIFAR-10 Python batches in canonical order.
- First 1,000 training and first 1,000 test images.
- RGB `32 x 32`, flattened to dimension `3072`.
- Conversion: `x / 127.5 - 1`, yielding `[-1,1]`.
- Training examples define `D`; held-out test examples estimate `p` for
  coverage.

These are clean-room subsets. They are not claimed to match the paper's random
1K training subset or exact held-out set.

## Frozen Estimators

Posterior queries use `X ~ p_D`, `Z ~ N(0,I)`, and paper Eq. (3). For each
query, weights are normalized using float64 log-sum-exp and the maximum is
recorded. `W_sigma(D)` is their mean.

Coverage queries use held-out `X`, independent Gaussian `Z`, and the exact
union event from Definition 4.6. A query is covered when at least one training
center has distance in the inclusive annulus

```text
[sigma r_in(5,3072), sigma r_out(5,3072)].
```

Nearest-neighbor distance is not substituted for this event.

## Frozen Clean-Room Configuration

```text
sigma grid = {0.02, 0.05, 0.1, 0.14, 0.2, 0.3, 0.4, 0.6, 0.8,
              1, 1.5, 2, 3, 4, 5, 8, 12, 20, 40, 80}
seed = 0
posterior corruption draws = 4
coverage corruption draws = 8
shell c = 5
bootstrap replicates = 500
```

Corruption draws are fixed and reused across sigma. Hierarchical bootstrap
intervals resample corruption draws and examples. All computations are
float64 except the originally generated standard-normal arrays, which used
NumPy float32 before float64 distance evaluation.

The executable configuration, including all ordered subset indices and their
hashes, is [`configs/e004a_local_geometry.json`](../configs/e004a_local_geometry.json).
The local evaluator verifies the six canonical CIFAR-10 batch hashes before
computation and refuses to overwrite a nonempty output directory.

## End-To-End Local Regeneration

The committed curves can be regenerated numerically rather than treated as
plotting inputs:

```bash
python experiments/04a_paper_geometry_curves.py \
  --compute \
  --dataset-root /path/to/cifar10 \
  --output-dir results/experiment_04a_reproduction \
  --device auto
```

The reference backend is NumPy/SciPy float64 on CPU. `--device auto` resolves
to that oracle; an accelerator backend must not be substituted until it agrees
with the oracle on the numerical tests. Pairwise clean distances and scalar
noise cross-terms are batched along query and reference axes. No
`queries x references x 3072` tensor is constructed.

The modes are deliberately separate:

- `--compute` regenerates numerical curves, uncertainty, validation,
  comparison, manifests, and figures from CIFAR-10 and fresh deterministic
  Gaussian corruptions;
- `--plot-only` reads one explicit result directory and regenerates figures;
- `--validate-only` checks schemas, ranges, finiteness, sigma identity, and the
  sampled high-high set without computing or plotting.

## Reproduction Agreement Rule

The tolerance was frozen before the artifact-independent deterministic rerun.
For each metric and sigma, define the committed standard-error proxy as

```text
SE_committed = (CI_high - CI_low) / (2 * 1.96)
```

and accept agreement when

```text
|fresh - committed| <= max(
    metric_absolute_floor,
    3 * sqrt(SE_fresh^2 + SE_committed^2)
)
```

The absolute floor is `0.01` for both coverage and maximum posterior weight.
`SE_fresh` is the standard error across independent corruption-draw means.
This rule allows expected Monte Carlo variation but remains strict enough to
expose implementation, subset, normalization, or RNG discrepancies. Agreement
is reported separately by metric and sigma. The sampled high-high set and the
qualitative three-regime pattern are validated independently and are never
forced to match.

## Paper-Guided Danger Region

The paper defines the qualitative mechanism as simultaneous high coverage and
high posterior concentration. It does not provide a universal threshold or an
exact Figure 3 boundary.

The clean-room run preregistered exploratory thresholds `q_W = q_C = 0.8` and
sensitivity values `0.7` and `0.9`. At the primary threshold, the full-space
point estimates are high-high at sampled sigma values `{2,3,4,5}`. Figures may
shade the continuous interval `2 <= sigma <= 5` only as a **paper-guided
clean-room high-high region**. The continuous shading is visual; decisions use
the observed grid points without interpolation.

The spectral E005 transition windows never define or revise this region.

## Acceptance Gates

- posterior weights are finite, nonnegative, and sum to one;
- max posterior weight lies in `[1/N,1]` up to tolerance;
- direct and stabilized posterior calculations agree on safe examples;
- shell coverage lies in `[0,1]` and matches exact union events;
- batching leaves shell membership unchanged;
- sigma/config/subset identities are recorded;
- curve values and intervals are finite;
- imported metrics match the validated hub source hash;
- fresh local estimates satisfy the frozen reproduction rule or discrepancies
  remain explicitly reported;
- paper geometry and spectral curves remain visually and conceptually distinct.
