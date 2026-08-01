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
- paper geometry and spectral curves remain visually and conceptually distinct.

