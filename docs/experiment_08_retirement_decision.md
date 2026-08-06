# E008 Retirement Decision

Status: **RETIRED_UNEXECUTED**

Historical outcome: **`BLOCKED_NO_ELIGIBLE_PAIR`**

## Decision

E008 is retired without executing its preregistered frequency-geometry swap
conditions. No further E008 training, checkpoint search, or swap execution is
planned.

## Evidence

The frozen E008 baseline preflight found six eligible EDM-1K checkpoints but
no eligible EDM-50K checkpoint: all 21 EDM-50K candidates scored `0/128`.
E009 then searched intermediate dataset sizes. Its eligible 2K endpoint did
not satisfy E008's frozen minimum larger-data role of 5K, and every tested 5K
checkpoint through 30K kimg remained `0/128`.

The required nondegenerate, baseline-matched cross-role pair was therefore not
found. The E008 protocol, preflight, and negative model-search evidence remain
preserved as historical artifacts.

## Disposition

E010 addressed the directional scientific question separately using an
explicitly asymmetric model pair. E010 does not count as E008 execution and
does not revise E008's frozen eligibility rule or historical outcome.
