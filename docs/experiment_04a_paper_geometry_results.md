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

