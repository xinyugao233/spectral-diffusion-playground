# E007: Geometry-Aligned Whole-Denoiser Swap Protocol

> **PROPOSED — NOT EXECUTED**

## Objective

Test whether swapping the whole denoiser at the E004A clean-room geometric
high-high points changes the frozen pixel-space memorization criterion more
than width-matched neighboring controls. This is a new experiment. It does not
replace or reinterpret historical E006.

## Frozen Geometry Target

E004A was evaluated directly on the 18-point E006 EDM schedule using the first
1,000 canonical CIFAR-10 training and test images, normalization `[-1,1]`,
seed `0`, shell constant `c=5`, four posterior draws, eight coverage draws,
and 500 hierarchical-bootstrap replicates.

The primary target uses the conservative lower-confidence-bound rule:

```text
C_sigma lower 95% bound >= 0.8
W_sigma lower 95% bound >= 0.8
```

It selects exactly indices `8..9`, with sigma values
`3.256821519765537` and `1.9233398370400518`. The point-estimate rule selects
the same indices and is retained as a reported sensitivity result. Qualifying
points are evaluated independently; interpolation and gap filling are
prohibited.

## Conditions

Use the frozen pure-Euler 18-step sampler and the matched clean-room EDM-1K and
EDM-50K checkpoints from E006. A swap condition uses the donor denoiser for
every inclusive index in its window and the base denoiser elsewhere.

Run both directions:

```text
EDM-1K base -> EDM-50K donor
EDM-50K base -> EDM-1K donor
```

For each direction, evaluate:

| Role | Inclusive indices | Sigma values |
| --- | --- | --- |
| Geometry target | `8..9` | `3.2568215..1.9233398` |
| Width-matched pre-control | `6..7` | `8.4009353..5.3151945` |
| Width-matched post-control | `10..11` | `1.0881706..0.5853481` |

Also evaluate the unchanged EDM-1K and EDM-50K no-swap baselines. Reuse the
same 256 latent seeds, `0..255`, for every condition. No additional window may
be introduced after results are inspected.

## Memorization Criterion

Use the same reference subset and representation as E006: unquantized,
unclamped RGB tensors in `[-1,1]`, flattened in channel-major order, with
Euclidean pixel-space distances to the frozen clean-room 1K training subset.
A generated sample is counted as memorized exactly when

```text
d_1NN < d_2NN / 3.
```

## Practical Influence Rule

For each direction, let `effect(T)` be the paired memorization-rate change
from its no-swap baseline. The geometry target passes the point threshold only
when all three conditions hold:

```text
abs(effect(T)) >= 0.10
abs(effect(T)) >= abs(effect(pre-control)) + 0.10
abs(effect(T)) >= abs(effect(post-control)) + 0.10
```

Paired bootstrap intervals and paired seed-level direction evidence must also
support the reported direction. Point estimates alone are insufficient.

## Degeneracy Guard And Outcomes

If either no-swap baseline has exactly `0/256` or `256/256` memorized samples,
the formal outcome is `INCONCLUSIVE`, regardless of target effect size. Model,
sampler, nearest-neighbor, or uncertainty failures also force
`INCONCLUSIVE`.

Assign exactly one outcome using the frozen E006 meanings:

- `YES`: the target passes against both controls in both directions;
- `PARTIAL`: the target passes in only one direction;
- `MIXED`: direction or control evidence conflicts;
- `NO`: the target does not pass the frozen practical threshold;
- `INCONCLUSIVE`: a degeneracy, uncertainty, or execution safeguard applies.

No outcome may be called a universal paper boundary. The phrase *candidate
memorization danger zone* is allowed only for a `YES` or `PARTIAL` result and
only with the clean-room geometry and intervention limitations stated nearby.

## Stop Conditions

Stop without interpreting results if any checkpoint, source, subset, sampler,
schedule, seed, tensor-domain, or output-quantization identity differs from
the frozen E006 setup. Stop if generated samples differ across batch sizes,
if a condition fails, if fewer than 256 paired seeds complete, or if any
window differs from the table above.

## Required Disclosure

E006 tested spectral-aligned windows and a literature-derived paper medium
reference. It did not preregister a window from locally computed coverage and
posterior-weight curves, because E004A did not yet exist. E007 is proposed to
close that gap and has not been executed.
