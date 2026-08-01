# E004A: Paper Geometry Baseline Results

## Status

**Completed as a paper-derived clean-room reproduction.** The compact result is
imported from the validated full-space gate in the research-context hub. The
paper's original executed code, exact random subset, Figure 3 sigma grid, and
random seeds were unavailable.

## Result

The clean-room curves reproduce the paper's qualitative three-regime picture:

- `sigma <= 0.4`: maximum posterior weight is `1`, while coverage is `0`;
- `sigma in {2,3,4,5}`: both point estimates exceed the preregistered
  exploratory threshold `0.8`;
- `sigma >= 8`: coverage remains near `1`, while maximum posterior weight
  decreases from `0.376` at `sigma=8` to `0.00289` at `sigma=80`.

At `sigma=5`, the posterior point estimate is `0.814`, but its 95% lower bound
is `0.799`; therefore the lower-bound high-high set is narrower than the
point-estimate region. Thresholds are clean-room diagnostics, not paper-defined
universal boundaries.

## Relation To E005 And E006

E005's low-frequency residual transition (`sigma=12.9101..0.585348`) overlaps
the clean-room geometry's sampled high-high values `{2,3,4,5}`. E005's
high-frequency transition (`sigma=0.585348..0.0599473`) does not. Thus the
geometric high-high region lies inside the low-frequency residual transition
on the shared sigma axis, while the high-frequency transition occurs later.

This is a descriptive overlap between different measurements on different
frozen grids. It is not a significance test and does not imply that low
frequencies cause memorization. E006 swaps the whole denoiser over
spectral-aligned windows; its formal outcome remains `INCONCLUSIVE`.

## Validation And Provenance

The source hub run passed posterior normalization, full-space recovery,
Parseval, projection, and projected-noise gates. Maximum errors were:

```text
posterior normalization: 1.142419492349286e-13
full-space recovery:     1.4210854715202004e-14
projected-noise relative error: 0.002770896511353082
```

Durable identities and clean-room deviations are recorded in
[`geometry_manifest.json`](../results/experiment_04a/geometry_manifest.json)
and the [source audit](paper_geometry_source_audit.md).

## End-To-End Local Regeneration

The committed curves were regenerated end to end from the hash-verified
CIFAR-10 Python batches using implementation commit `432c371489961be3b55dea187bf8ea6951ffb9be`.
This run loaded and normalized the frozen first 1,000 training and first 1,000
test examples, generated fresh deterministic corruption arrays from seed `0`,
and recomputed both metrics rather than reading committed estimates.

The rerun is independent of the committed curve estimates, but uses the same
frozen subset, seed, sigma grid, estimator, normalization, and corruption
draws. Its near-exact agreement demonstrates deterministic reproducibility,
not robustness across alternative subsets or random seeds.

```text
device: CPU, NumPy/SciPy float64 oracle
runtime: 3.3108 seconds
peak resident memory: 441.8 MiB
config SHA-256: cd1ab1afbde72d66465e3208c7c3627b69e93b87b614e2b7a613e4d08ec4be4a
posterior normalization error max: 1.141309269314661e-13
```

Every sigma passes the frozen uncertainty-aware reproduction tolerance.
Maximum absolute differences are:

```text
Gaussian-shell coverage:    1.1102230246251565e-16
maximum posterior weight:   4.718447854656915e-16
```

The fresh sampled high-high set remains `{2,3,4,5}`, and the qualitative
small-/medium-/large-noise picture is unchanged. This numerical agreement is
specific to the frozen clean-room configuration and does not convert the
result into an exact reproduction of the paper's unavailable execution.

The fresh curves, per-sigma comparison, validation, manifest, and regenerated
figures are committed under
[`results/experiment_04a_reproduction/`](../results/experiment_04a_reproduction/).

## Exact E006-Grid Evaluation

The E004A estimator was additionally run directly at all 18 E006 sampler
sigmas. At the preregistered clean-room thresholds `q_C=q_W=0.8`, indices
`{8,9}` qualify, corresponding to `sigma={3.2568215,1.9233398}`. The 95% lower
confidence bounds produce the same set. Classification uses evaluated points
only, with no interpolation or gap filling.

The set overlaps the paper-reported medium reference `6..13` and the E005
low-frequency spectral transition `5..11`, but not the E005 high-frequency
spectral transition `11..14`. These are descriptive set relationships among
different definitions. The result does not retroactively alter E006 and does
not establish a universal paper boundary.

```bash
python experiments/04a_paper_geometry_curves.py \
  --compute-e006-grid \
  --dataset-root /path/to/cifar10 \
  --device auto
```

See the [alignment figure](../figures/experiment_04a/e006_grid_geometry_alignment.png),
[manifest](../results/experiment_04a/e006_grid_geometry_manifest.json), and
[region-definition audit](danger_zone_definition_audit.md).
