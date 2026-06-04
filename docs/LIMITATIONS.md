# Limitations

This repo is intentionally selective, and so are its claims.

## Technical Limits

- The deployed student is blind at inference and therefore still bounded by the
  quality of proprioceptive history alone.
- The strongest remaining weakness is lateral-push behavior in the
  continuous corridor setting.
- Cross-runtime validation is useful, but it is not a substitute for complete
  hardware validation.
- This repo does not claim that the student dominates every possible
  disturbance family.

## Claim Limits

This repo claims:

- a strong blind-history locomotion path
- a teacher-guided training story
- a validated export surface
- deployment-oriented evaluation discipline

This repo does not claim:

- solved online adaptation
- final hardware robustness closure
- full rough-terrain generalization
- one policy that wins every robustness axis

## Why Publish It This Way

A narrower claim is more useful than a broader, less credible one.

This repo is meant to be:

- understandable
- reproducible
- technically defensible

That keeps the scope narrow and the claim defensible.
