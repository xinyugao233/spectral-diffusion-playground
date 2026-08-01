# E004B: Frequency-Restricted Gaussian-Shell Geometry Protocol

## Status

Frozen before the primary CIFAR-10 computation. E004B is a paper-derived
clean-room extension, not an exact reconstruction of unavailable paper code.

## Objective

Compute the paper's two geometric quantities separately after projecting the
same clean images and Gaussian corruptions into complementary low- and
high-frequency subspaces. E004A remains the original full-space baseline.
E005 remains a distinct denoising-residual experiment.

## Frozen Data And Schedule

- CIFAR-10 canonical Python batches, hash verified.
- First 1,000 training examples and first 1,000 test examples.
- Flattened channel-first RGB values normalized as `x / 127.5 - 1`.
- Exact descending 18-point EDM sampler schedule recorded in
  [`configs/e004b_frequency_restricted_geometry.json`](../configs/e004b_frequency_restricted_geometry.json).
- Seed `0`; four posterior corruption draws; eight coverage corruption draws.
- 500 hierarchical bootstrap replicates.
- Float64 NumPy CPU oracle after source-compatible float32 Gaussian draws.

## Fourier Projectors

For cutoff `r`, the low mask includes every centered Fourier bin whose radius
is `<= r`, including DC. The high mask is its exact binary complement. Both
masks are conjugate symmetric and are applied independently to each channel
with `norm="ortho"`.

The primary cutoff is `r=4`; `r=3,5` are sensitivity checks. Their exact real
ranks are:

| Cutoff | Low rank | High rank |
| --- | ---: | ---: |
| `3` | 87 | 2985 |
| `4` | 147 | 2925 |
| `5` | 243 | 2829 |

Projected vectors are transformed back into real spatial coordinates before
distances are computed. The following identities must pass numerically:

```text
P_low + P_high = I
P_low P_high = 0
P_low^2 = P_low
P_high^2 = P_high
||x||^2 = ||P_low x||^2 + ||P_high x||^2
```

## Posterior Concentration

For band `b` in `{low, high}`:

```text
w_i^b(y,sigma)
  = exp(-||P_b y - P_b x_i||^2 / (2 sigma^2))
    / sum_j exp(-||P_b y - P_b x_j||^2 / (2 sigma^2))

W_sigma^b(D) = E[max_i w_i^b(P_b(X + sigma Z), sigma)]
```

Queries are noisy training examples. Distances are neither divided by band
rank nor otherwise normalized. Sigma is not rescaled.

## Gaussian-Shell Coverage

Coverage uses noisy held-out queries and the exact union of shells centered at
the projected training examples. For real projector rank `d_b` and `c=5`:

```text
r_in(d_b)  = sqrt(max(d_b - 2 sqrt(c d_b), 0))
r_out(d_b) = sqrt(d_b + 2 sqrt(c d_b) + 2c)
```

The shell radii use `d_b`, never the 3072-value storage width. The estimator
checks whether any projected training center lies within the inclusive shell.

## Paired Randomness

For every cutoff, one seeded sequence of full-dimensional Gaussian draws is
generated in the E004A order: posterior draws first, coverage draws second.
The same draw is projected into both bands. Reinitializing the same generator
for each cutoff preserves the same underlying draws across `r=3,4,5`.

## Target Selection

For each band and evaluated sigma point, record point-estimate and lower-bound
classifications at `q_C=q_W=0.8`. The primary target uses both 95% lower
confidence bounds at `r=4`:

```text
C_sigma^b lower bound >= 0.8
W_sigma^b lower bound >= 0.8
```

Only observed grid points are eligible. No interpolation, gap filling, manual
widening, or conversion of noncontiguous components into one interval is
allowed. An empty set remains empty. E005 residual curves and swap outcomes
cannot modify either target. Results at `r=3,5` report sensitivity only and
cannot revise the primary cutoff.

## Outputs And Interpretation

The computation writes one long-format table, numerical validation, target
summary, cutoff sensitivity, provenance manifest, and five review figures.
Interpretation is descriptive: E004B may locate different geometric regimes
in different subspaces, but it cannot establish that a frequency band causes
memorization. Any intervention over the selected temporal windows belongs to
the proposed, unexecuted E008 protocol.
