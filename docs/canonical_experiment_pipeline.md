# Canonical Experimental Pipeline

## Scientific Order

The repository's central experiment follows one directional chain:

```text
freeze Fourier cutoff
        -> compute full-space paper geometry
        -> compute geometry in each frozen frequency band
        -> select full-space and band-specific targets
        -> compare them with spectral residual transitions
        -> propose target-aligned interventions
```

Coverage and posterior concentration select the geometry targets. E005
residual-energy curves do not select or revise them.

## Stage 1: Operational Frequency Cutoff

**Source:** E004.

E004 freezes reference cutoff `r=4`, with sensitivity cutoffs `r=3,5`, before
model-curve interpretation. This radius defines the centered complementary
Fourier projections `P_low,r` and `P_high,r` used by E004B and E005.

The cutoff does not enter the definitions of `C_sigma(p,D)` or `W_sigma(D)`.
It is frozen first so the later frequency-band interpretation is
preregistered, not so it can determine the candidate interval.

## Stage 2: Paper Geometry And Candidate Selection

**Source:** E004A.

E004A computes Gaussian-shell coverage `C_sigma(p,D)` and maximum posterior
concentration `W_sigma(D)` directly on the exact 18-point sampler schedule. The
primary clean-room rule is

```text
coverage lower 95% bound >= 0.8
posterior lower 95% bound >= 0.8
```

with `q_C=q_W=0.8`. Evaluated points only are eligible; interpolation and gap
filling are prohibited. The rule selects indices `8..9`, at sigma values
`3.256821519765537` and `1.9233398370400518`.

This is the **E004A clean-room full-space geometry target**. It is not a
universal boundary or an exact reconstruction of the paper's unavailable
executed Figure 3 configuration.

## Stage 3: Frequency-Restricted Geometry

**Source:** E004B.

E004B computes the same coverage and maximum-posterior-weight definitions on
the same data and Gaussian corruptions after projection into the frozen
complementary subspaces. At `r=4`, the primary lower-confidence-bound rule
selects low target `{8}` and high target `{9,10}`.

These are joint coverage/posterior targets. Coverage alone does not have the
same ordering: high-band coverage persists to lower sigma, while low-band
posterior concentration persists farther toward high noise.

The low/high ranks at `r=4` are `147/2925`. E004B is completed as a
descriptive measurement of the operational Fourier decomposition, but rank,
covariance, and energy differences prevent a frequency-only attribution.

The low target is unchanged at `r=3,4,5`. The high target is `{9,10}` at
`r=3,4` and narrows to `{10}` under the `r=5` lower-confidence-bound
sensitivity check. This scale dependence is reported and cannot revise the
primary cutoff after inspection.

## Stage 4: Spectral Residual Interpretation

**Source:** E005.

Using the already frozen `r=4` projections, E005 measures complementary
low- and high-frequency residual energies. The geometry target `8..9` lies
inside the E005 low-frequency spectral transition `5..11` and does not overlap
the E005 high-frequency spectral transition `11..14`.

The justified interpretation is:

> The independently full-space geometry target occurs during the
> low-frequency residual transition and before the high-frequency residual
> transition. The E004B low and high geometry targets both lie within that
> same E005 low-frequency residual transition.

E005 does not define the candidate zone. Its transition rule, curve
intersection, and low-frequency band cannot replace the E004A coverage and
posterior-concentration selection rule. The overlap does not establish that
low frequencies cause memorization.

## Stage 5: Historical Exploratory Swaps

**Source:** E006.

E006 is the historical spectral-window swap experiment. It tested the E005
low transition `5..11`, E005 high transition `11..14`, their union `5..14`,
and the literature-derived paper-reported medium reference `6..13`.

E006 did not test a window selected from locally computed coverage and
posterior-weight curves because E004A did not yet exist. Its formal outcome
remains `INCONCLUSIVE`. E006 is useful exploratory evidence about
spectral-aligned trajectory intervals, but it is not the final
geometry-aligned intervention.

## Stage 6: Proposed Full-Space Geometry Swap

**Source:** proposed E007.

E007 asks:

> Does swapping the whole denoiser exactly over the independently
> geometry-defined high-high interval change final memorization more than
> equally wide neighboring intervals?

The primary target is `8..9`; the width-matched pre-control is `6..7`; and the
width-matched post-control is `10..11`. This is the required final intervention
connecting the paper geometry to trajectory-level memorization.

E007 is **PROPOSED — BLOCKED BY KNOWN BASELINE DEGENERACY**. The historical
model pair cannot support an informative bidirectional test because its
EDM-50K no-swap baseline is already `0/256`. A nondegenerate model pair must be
selected and frozen through the preregistered baseline-only preflight before
E007 can run.

## Stage 7: Proposed Frequency-Geometry Swaps

**Source:** proposed E008.

E008 preregistered separate whole-denoiser interventions over E004B low target
`8..8` and high target `9..10`, with immediately adjacent width-matched
controls. It is not a Fourier coefficient swap. E008 is
**RETIRED_UNEXECUTED**.

The separately frozen E008 checkpoint preflight evaluated only full no-swap
trajectories on pilot seeds `10000..10127`. It inventoried the existing
matched-run checkpoint pools before inference, applied the immutable eligible
count range `13..115`, and reserved confirmatory seeds `0..255`. The completed
gate returned `BLOCKED_NO_ELIGIBLE_PAIR`: six EDM-1K checkpoints were eligible,
but all 21 EDM-50K checkpoints produced `0/128`. No E008 condition was run.

## Canonical Region Table

| Object | Indices | Sigma | How selected | Role |
| --- | --- | --- | --- | --- |
| E004A full-space target | `8..9` | `3.2568..1.9233` | Full-space `C,W` lower bounds >= `0.8` | Full-space geometry target |
| E004B low target | `8` | `3.2568` | Low-subspace `C,W` lower bounds >= `0.8` | Band-specific geometry target |
| E004B high target | `9..10` | `1.9233..1.0882` | High-subspace `C,W` lower bounds >= `0.8` | Band-specific geometry target |
| E005 low transition | `5..11` | `12.9101..0.5853` | Spectral 20%-to-80% rule | Spectral interpretation |
| E005 high transition | `11..14` | `0.5853..0.05995` | Spectral 20%-to-80% rule | Later spectral transition |
| Paper reference | `6..13` | `8.4009..0.1395` | Literature convention | Historical context |
| E007 pre-control | `6..7` | `8.4009..5.3152` | Width matched | Control |
| E007 post-control | `10..11` | `1.0882..0.5853` | Width matched | Control |

## Stage 8: E009 Model-Pair Search

E009 tested whether intermediate data sizes could provide the eligible
baseline-matched larger-data endpoint required by E008. Stage A found only an
eligible 2K endpoint; Stage B extended the 5K lineage through 30K kimg, where
all 18 checkpoints remained `0/128`. E008 was therefore retired unexecuted;
no further E008 model search or swap execution is planned.

## Stage 9: E010 Directional Asymmetric-Baseline Test

E010 asks a different question and explicitly accepts asymmetric baselines.
It swaps whole denoisers bidirectionally over E004B low target `{8}`, high
target `{9,10}`, and neighboring controls. The frozen result supports only
high-derived suppression; induction and low-derived suppression fail their
criteria. E010 addresses a separate asymmetric-baseline directional question;
it does not replace or count as execution of retired E008.

## Non-Substitution Rules

- `r` defines E005 Fourier projections; it does not define `C_sigma` or
  `W_sigma` in E004A. It does define E004B's projected subspaces.
- E004A selects the clean-room candidate zone; E005 only interprets it.
- E004B computes geometry inside bands; it is not an E005 residual analysis.
- E006 cannot be relabeled as testing the E004A target.
- E007 cannot be described as executed or informative until its baseline-only
  model-pair preflight passes.
- E008 cannot be described as executed, and its temporal whole-denoiser swaps
  cannot be called Fourier coefficient swaps.
- E010 cannot be described as baseline matched, as a frequency-component
  intervention, or as identifying dataset-size causality.
