# Configured fictional generator

The first generator samples one event for each configured variable per episode,
with one patient per episode. Each variable has categorical values, probabilities,
and event and recording days. Variables are independent; the index variable must
be configured exactly once. Unconfigured study variables remain absent. This
contract is separate from the study and the resulting fixture.

SplitMix64 maps its upper 53 bits to a floating-point sample in [0, 1). The
manifest retains the full generator configuration, algorithm and software version;
the output also retains the parsed study configuration. Results
are tested for repeatability in the same environment, not guaranteed across
platforms or versions. This explicit mechanism avoids an unrecorded dependency
or fitted model; it does not represent clinical relationships or causal effects.
Future models can replace this contract when their assumptions and evaluation
are defined.
## Conditional tables

Version 2 extends categorical generation with ordered parent variables and a
complete table for their value combinations. This expresses associations only;
continuous distributions and missingness mechanisms remain future work.
Roots specify `outcomes`, while dependent variables specify `parents` and a
`table` whose `when` maps every parent to one categorical value. The Cartesian
product uses the union of values across each parent's distributions, including
zero-probability values. This keeps assumptions explicit and rejects incomplete
tables. An alternative implicit default distribution would hide unspecified
combinations and was not chosen. Parent order prevents cycles but does not prove
temporal causation, privacy, plausibility, or external validity.
