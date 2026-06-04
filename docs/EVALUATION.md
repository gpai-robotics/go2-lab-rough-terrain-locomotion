# Evaluation

This repo treats evaluation as more than reward tracking.

The line here is evaluated with a deployment-oriented filter:

- source/export parity
- nominal deployment rehearsal
- hidden-parameter mismatch checks
- disturbance checks
- cross-runtime sanity checks

## Core Evaluation Questions

1. Does the exported policy match the source checkpoint?
2. Does the policy stay stable on the deployment surface?
3. Does it survive realistic disturbances and hidden mismatch?
4. Where does it still break?

## Export Parity

The first gate is parity between the trained source policy and the exported
TorchScript bundle.

Representative metrics:

- mean absolute action difference: `6.0e-09`
- mean action MSE: `7.2e-16`
- max absolute action difference: `4.77e-07`

Interpretation:

- the export path is not merely "close enough"
- the export matched the source policy essentially exactly

## Isaac Deploy Rehearsal

The second gate is a deployment-surface rehearsal in Isaac.

Representative metrics:

- velocity error mean: `0.0679`
- base height mean: `0.3496`
- projected gravity xy mean: `0.0667`
- no terminations in the canonical rehearsal

Interpretation:

- the exported runtime contract is stable
- there was no immediate deployment-surface failure after export

## MuJoCo Runtime Validation

The third gate is a second runtime family used as a deployment-oriented stress
test rather than a claim of complete sim-to-real closure.

Representative nominal metrics:

- reward proxy mean: `0.4650`
- velocity error mean: `0.1363`
- yaw error mean: `0.0716`
- base height mean: `0.3222`
- control saturation fraction mean: `0.0`

Representative hidden-mismatch behavior:

- ultra-high friction remained healthy
- heavy payload remained healthy
- ultra-low friction remained functional
- weak-motor cases remained functional

Representative disturbance behavior:

- command steps were handled well
- moderate pushes were survivable
- moderate yaw pulses were survivable

## Main Remaining Weakness

The most important remaining weakness is not hidden.

The policy degraded most clearly on:

- lateral push behavior in the continuous corridor suite

That matters because it bounds the actual claim:

- the policy is viable and strong in several deployment-facing probes
- it is not presented here as "solved under all disturbances"

## Scripts

- `scripts/export_policy.py`
- `scripts/validate_bundle.py`
- `scripts/eval_deploy.py`

The goal of these scripts is to keep the evaluation path legible.

## Media Boundary

This repo includes IsaacLab demo media and one MuJoCo nominal runtime clip.

Additional MuJoCo-facing media should be labeled explicitly.
