# Results

This repo focuses on one clear result family: a blind history-conditioned C1
student with a validated teacher lineage and a deployment-facing export path.

## What Worked

- the teacher path produced useful privileged supervision
- the student kept a blind deployable inference contract
- the history pathway remained behaviorally important
- the export path could be validated to near machine precision
- the deployment rehearsal and cross-runtime checks stayed viable

## Snapshot

Representative metrics:

| Check | Value | Read |
| --- | --- | --- |
| Export parity mean abs diff | `6.0e-09` | Export matched source almost perfectly |
| Export parity max abs diff | `4.77e-07` | No meaningful runtime drift from export |
| Isaac deploy velocity error mean | `0.0679` | Stable deployment-surface tracking |
| MuJoCo nominal velocity error mean | `0.1363` | Cross-runtime nominal behavior stayed viable |
| MuJoCo control saturation frac mean | `0.0` | The controller was not saturating constantly |

## Qualitative Story

The policy is strongest when framed honestly:

- good nominal locomotion
- useful tolerance to hidden friction, mass, and motor mismatch
- meaningful push and command-change robustness
- credible stair and box traversal demonstrations

The demos are meant to show the policy's character, not to replace
quantitative evaluation.

## Why The Result Is Interesting

The useful result is not "one giant adaptive policy beats everything."

The useful result is:

- a clean blind-history runtime contract can still support strong rough-terrain
  behavior
- teacher guidance can help shape that student without leaking privilege at
  inference
- deployment-facing evaluation changes which models are actually worth keeping

## What To Conclude

Reasonable conclusion:

- a deployable blind-history controller can be made strong, inspectable, and
  exportable without a much larger adaptation stack

Unreasonable conclusion:

- rough-terrain blind locomotion is fully solved

That is not the claim of this repo.
