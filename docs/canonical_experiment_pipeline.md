# Canonical Experimental Pipeline

## Scientific Order

The repository's central experiment follows one directional chain:

```text
freeze Fourier cutoff
        -> compute paper geometry
        -> select geometry-defined candidate interval
        -> interpret that interval spectrally
        -> intervene exactly on the geometry-defined interval
```

Coverage and posterior concentration select the candidate interval. The
spectral curves do not select or revise it.

## Stage 1: Operational Frequency Cutoff

**Source:** E004.

E004 freezes reference cutoff `r=4`, with sensitivity cutoffs `r=3,5`, before
model-curve interpretation. This radius defines the centered complementary
Fourier projections `P_low,r` and `P_high,r` used by E005.

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

This is the **E004A clean-room geometry-defined candidate danger zone**. It is
not a universal boundary or an exact reconstruction of the paper's unavailable
executed Figure 3 configuration.

## Stage 3: Spectral Interpretation

**Source:** E005.

Using the already frozen `r=4` projections, E005 measures complementary
low- and high-frequency residual energies. The geometry target `8..9` lies
inside the E005 low-frequency spectral transition `5..11` and does not overlap
the E005 high-frequency spectral transition `11..14`.

The justified interpretation is:

> The independently geometry-defined candidate danger zone occurs during the
> low-frequency residual transition and before the high-frequency residual
> transition.

E005 does not define the candidate zone. Its transition rule, curve
intersection, and low-frequency band cannot replace the E004A coverage and
posterior-concentration selection rule. The overlap does not establish that
low frequencies cause memorization.

## Stage 4: Historical Exploratory Swaps

**Source:** E006.

E006 is the historical spectral-window swap experiment. It tested the E005
low transition `5..11`, E005 high transition `11..14`, their union `5..14`,
and the literature-derived paper-reported medium reference `6..13`.

E006 did not test a window selected from locally computed coverage and
posterior-weight curves because E004A did not yet exist. Its formal outcome
remains `INCONCLUSIVE`. E006 is useful exploratory evidence about
spectral-aligned trajectory intervals, but it is not the final
geometry-aligned candidate-zone intervention.

## Stage 5: Final Geometry-Aligned Swap

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

## Canonical Region Table

| Object | Indices | Sigma | How selected | Role |
| --- | --- | --- | --- | --- |
| E004A geometry target | `8..9` | `3.2568..1.9233` | `C,W` lower bounds >= `0.8` | Primary candidate danger-zone target |
| E005 low transition | `5..11` | `12.9101..0.5853` | Spectral 20%-to-80% rule | Spectral interpretation |
| E005 high transition | `11..14` | `0.5853..0.05995` | Spectral 20%-to-80% rule | Later spectral transition |
| Paper reference | `6..13` | `8.4009..0.1395` | Literature convention | Historical context |
| E007 pre-control | `6..7` | `8.4009..5.3152` | Width matched | Control |
| E007 post-control | `10..11` | `1.0882..0.5853` | Width matched | Control |

## Non-Substitution Rules

- `r` defines E005 Fourier projections; it does not define `C_sigma` or
  `W_sigma`.
- E004A selects the clean-room candidate zone; E005 only interprets it.
- E006 cannot be relabeled as testing the E004A target.
- E007 cannot be described as executed or informative until its baseline-only
  model-pair preflight passes.
