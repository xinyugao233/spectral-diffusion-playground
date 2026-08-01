# E007: Geometry-Aligned Whole-Denoiser Swap Protocol

> **PROPOSED — BLOCKED BY KNOWN BASELINE DEGENERACY**

## Objective

Test whether swapping the whole denoiser at the E004A clean-room geometric
high-high points changes the frozen pixel-space memorization criterion more
than width-matched neighboring controls. This is a new experiment. It does not
replace or reinterpret historical E006.

## Known Design Blocker

The original E006 model pair cannot support the proposed bidirectional E007
decision rule. Under the frozen sampler and seeds, the EDM-50K no-swap baseline
is already known to be `0/256`. Because the protocol's degeneracy guard treats
this as `INCONCLUSIVE`, executing E007 unchanged would not produce an
informative formal outcome.

Rerunning the same baselines cannot resolve this blocker. The historical E006
model pair is not execution-ready for the primary bidirectional E007 test and
must not be permitted to produce a non-`INCONCLUSIVE` classification.

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

## Proposed Conditions After Preflight

Only after the model-pair preflight below selects and freezes an eligible pair,
use the frozen pure-Euler 18-step sampler from E006. A swap condition uses the
donor denoiser for every inclusive index in its window and the base denoiser
elsewhere. The historical E006 checkpoints are not eligible by default merely
because their provenance is complete.

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

Evaluate both selected-model no-swap baselines and reuse the same 256
confirmatory latent seeds, `0..255`, for every final condition. No additional
window may be introduced after results are inspected.

## Required Model-Pair Preflight

Before any geometry-aligned swap is executed:

1. Freeze separate candidate checkpoint pools for the 1K-trained and
   50K-trained model roles, including paths, hashes, training configuration,
   subset size, duration, and EMA status.
2. Freeze pilot seeds `10000..10127`. These 128 seeds are disjoint from the
   confirmatory seeds `0..255`.
3. During selection, evaluate only no-swap baseline memorization under the
   frozen sampler and strict `d_1NN < d_2NN / 3` criterion.
4. Do not generate or inspect geometry-target, control-window, or any other
   swap effect during model-pair selection.
5. Use a prospective default eligibility interval of `[0.10, 0.90]` for each
   pilot baseline memorization rate. For 128 pilot seeds, this requires an
   inclusive memorized count from `13` through `115`.
6. Treat `[0.10, 0.90]` as a new E007 design choice, not a paper or E006 rule.
   It excludes near-floor and near-ceiling baselines so both increases and
   decreases remain empirically measurable.
7. Record every candidate and pilot result, including rejected checkpoints.
8. If multiple pairs are eligible, select the pair with the smallest absolute
   difference between pilot baseline rates; break exact ties by the
   lexicographically ordered pair of checkpoint SHA-256 values.
9. Freeze the selected pair, hashes, baseline-only selection record, target,
   controls, and decision rule before using any confirmatory seed.
10. Use confirmatory seeds only after the complete design is frozen. Pilot
    seeds must not appear in the confirmatory analysis.

If either role has no eligible checkpoint, E007 remains blocked. The
eligibility interval may not be relaxed after pilot results are observed.

## Provenance-Backed Candidate Search Space

Repository provenance identifies possible candidates, not eligible models:

- The matched clean-room EDM-50K run records EMA snapshots every `2,000 kimg`
  from `network-snapshot-000000.pkl` through
  `network-snapshot-040000.pkl` under
  `/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/`.
- The EDM-1K anchor configuration records the same snapshot cadence under
  `/home/xggh8/data/zw-lab/exp_004_standard_edm_n1000_40000kimg_20260415/`,
  but this repository freezes only the final checkpoint identity. Any
  intermediate files must be inventoried and hashed before entering the pool.
- The active repository contains no provenance-complete intermediate-subset
  model that can be presumed eligible. Any such model must first receive the
  same checkpoint and training-data provenance treatment.

Intermediate duration, intermediate subset size, and other clean-room models
are plausible sources of nondegenerate baselines, but none is suitable until
it passes the frozen baseline-only pilot. No checkpoint evaluation or training
is part of this protocol update.

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

The known historical EDM-50K count of `0/256` therefore guarantees
`INCONCLUSIVE` for the original E006 pair. The classification below applies
only to a newly preregistered pair that passes the model-pair preflight.

Assign exactly one outcome using the frozen E006 meanings:

- `YES`: the target passes against both controls in both directions;
- `PARTIAL`: the target passes in only one direction;
- `MIXED`: direction or control evidence conflicts;
- `NO`: the target does not pass the frozen practical threshold;
- `INCONCLUSIVE`: a degeneracy, uncertainty, or execution safeguard applies.

No outcome may be called a universal paper boundary. The phrase *candidate
memorization danger zone* is allowed only for a `YES` or `PARTIAL` result and
only with the clean-room geometry and intervention limitations stated nearby.

## Optional One-Direction Descriptive Analysis

An `EDM-1K base -> EDM-50K donor` analysis over indices `8..9` may be specified
separately because the historical EDM-1K baseline is `247/256`, not exactly
degenerate. It would be a one-direction descriptive experiment, not the
primary bidirectional E007 test. It must not receive a
`YES/PARTIAL/MIXED/NO` classification, and the reverse direction remains
blocked by the historical EDM-50K `0/256` baseline. This optional analysis has
not been executed.

## Stop Conditions

Stop before swap execution unless the candidate pools, pilot eligibility
interval, pilot seeds, confirmatory seeds, and selected pair have all been
frozen. Stop if no eligible pair exists. After selection, stop without
interpreting results if any selected checkpoint, source, subset, sampler,
schedule, seed, tensor-domain, or output-quantization identity differs from
its preregistered value. Stop if generated samples differ across batch sizes,
if a condition fails, if fewer than 256 paired confirmatory seeds complete, or
if any window differs from the table above.

## Required Disclosure

E006 tested spectral-aligned windows and a literature-derived paper medium
reference. It did not preregister a window from locally computed coverage and
posterior-weight curves, because E004A did not yet exist. E007 is proposed to
close that gap, but its primary bidirectional design is blocked until a
nondegenerate model pair is preregistered. E007 has not been executed.
