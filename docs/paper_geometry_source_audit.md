# Paper Geometry Source Audit

## Status

This audit supports a **paper-derived clean-room reproduction** of the geometric
curves in *Two Calm Ends and the Wild Middle: A Geometric Picture of
Memorization in Diffusion Models* (arXiv:2602.17846v1). The original executed
Figure 3 implementation and exact subset identities were not recovered.

Paper PDF audited: `2602.17846v1.pdf`, SHA-256
`1dd6b436878c74327dab0c289e57335a915cdb35845fb75967882ba62375d2d8`.
The PDF is not committed because it is an external paper artifact.

## Exact Paper Definitions

### Maximum posterior weight

Paper Eq. (3) assigns training example `x_i` the empirical posterior weight

```text
w_i(y, sigma) = exp(-||y - x_i||^2 / (2 sigma^2))
                / sum_j exp(-||y - x_j||^2 / (2 sigma^2)).
```

Definition 4.1 then defines

```text
W_sigma(D) = E_{X ~ p_D, Z ~ N(0,I)} max_i w_i(X + sigma Z, sigma).
```

The expectation is over training examples and Gaussian corruption. It is not
evaluated on held-out queries, model attention, or nearest-neighbor ratios.
The clean-room implementation uses float64 log-sum-exp stabilization.

### Gaussian-shell coverage

Lemma 4.5 defines, for dimension `d` and shell constant `c > 0`,

```text
r_in(c,d)  = sqrt(d - 2 sqrt(c d))
r_out(c,d) = sqrt(d + 2 sqrt(c d) + 2c).
```

The shell in paper Eq. (6) is

```text
S_sigma(x) = {x': sigma r_in <= ||x' - x|| <= sigma r_out}.
```

Definition 4.6 defines

```text
C_sigma(p,D) = P(X + sigma Z is in union_i S_sigma(x_i)),  X ~ p.
```

Here `p` is the underlying **data distribution**, not a scalar shell-probability
parameter. The paper uses `c = 5` unless stated otherwise, which gives shell
mass at least `1 - 2 exp(-5)`, approximately `0.9865`.

Coverage is an exact binary union-of-annuli event per noisy held-out query and
then an average. It is not replaced by nearest-neighbor distance. Theorem 4.8
provides nearest-neighbor and pairwise-distance bounds, but Figure 3 reports
empirical coverage itself.

## Figure 3 And Regime Logic

Figure 3 plots `C_sigma(p,D)` and `W_sigma(D)` for a 1K CIFAR-10 training
subset and 1K held-out images on a shared log-scaled sigma axis. Section 4.2
describes:

- small noise: high posterior concentration, low coverage;
- medium noise: both quantities transition, with a high-high danger region;
- large noise: low posterior concentration, high coverage.

The paper explicitly says that precise regime estimation requires further
work. It does not provide an algorithmic Figure 3 boundary or a universal
numeric threshold. Figure 5 and Appendix E.4 identify dataset-specific regions
empirically from the two geometric curves.

## Experimental Details Recovered

- Domain: CIFAR-10 RGB, flattened to `d = 3072`.
- Value range: linearly rescaled to `[-1, 1]` (Appendix E.2).
- Distance: Euclidean distance in flattened normalized pixel space.
- Training/query counts in Figure 3 caption: 1,000 / 1,000.
- Training subset: randomly sampled, but exact indices and ordering unavailable.
- Shell constant: `c = 5`.
- Appendix E.2 coverage study: 350 trials, a fresh 1K subset per trial, one
  held-out image, and 10 Gaussian corruptions per sigma.
- Appendix E.3 posterior diagnostic: one fixed random 1K subset, 100 training
  base points, and 400 corruptions per sigma.
- Appendix E.2 schedule: 40-step EDM polynomial schedule from `0.002` to `80`.
- Exact Figure 3 sigma values, random seeds, subset identities, and aggregation
  code are not specified.

The Appendix E.2/E.3 procedures are related diagnostics; the paper does not
fully state that their exact draw counts generated Figure 3.

## Code And History Search

Searches covered the active playground tree, every local Git branch/tag and
history, the research-context hub, likely local research repositories from the
earlier provenance audit, arXiv metadata, and targeted public GitHub/web
queries for the title, `C_sigma`, `W_sigma`, Gaussian-shell coverage, and max
posterior weight. No official or author-provided Figure 3 implementation was
located. The arXiv record does not link code.

The playground's previous E005 posterior-mean implementation computes a
different object and did not contain Figure 3 coverage curves. Git history
contains no deleted paper-geometry baseline.

## Clean-Room Source Selected

The imported result comes from the validated 1K/1K full-space gate in the
shared research-context hub:

```text
hub code commit: 5de3c5ef23333484a1406fca712206105b9f0bd4
script SHA-256:  37ea80919f5f92c09ce4dfad01ddc3bbf19f6912d8de60bd26100cbffb60cd7c
config SHA-256:  b4bf4dfa6d18440605083f7eb2ccc49c22e77f6542c1be02b7c5c4997a90b6a2
```

Clean-room deviations from the unavailable Figure 3 execution are explicit:

- first 1,000 canonical training and test examples, rather than the unknown
  random paper subset;
- seed `0`;
- 4 posterior and 8 coverage corruption draws reused across sigma;
- a frozen 20-point grid from `0.02` to `80`;
- 500 hierarchical bootstrap replicates;
- exploratory high-high thresholds `q_W = q_C = 0.8`, with `0.7` and `0.9`
  sensitivity, frozen before the hub run.

These choices qualitatively reproduce the paper's three-regime geometry but do
not support code-identical or numerically exact reproduction claims.

