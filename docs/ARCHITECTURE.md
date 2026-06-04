# Architecture

This repo contains one teacher path and one student path.

## System View

```text
deployable policy observations
        + history window
                |
                v
     blind history-conditioned student
                |
                v
             actions
```

During training only, the student is guided by a privileged teacher:

```text
deployable observations
    + privileged dynamics
    + privileged terrain
            |
            v
   privileged rough-terrain teacher
            |
            v
   supervision signal for student
```

## Repo Boundary

IsaacLab provides:

- simulator infrastructure
- environment plumbing
- RL integration surface

This repo provides:

- the extracted rough-terrain task configs
- the teacher and student model definitions
- training scripts
- export and validation scripts

## Teacher

The teacher is a privileged rough-terrain controller.

Its role is to use richer training-time information to learn a stronger control
policy and shape the student toward that behavior.

The teacher path matters because terrain and hidden-dynamics information are
available during training and not at inference.

## Student

The student is the main deployed policy.

Runtime contract:

- `policy_obs [48]`
- `policy_history [4800]`
- `action [12]`

That is equivalent to:

- current deployable observation
- 100-step history buffer at 50 Hz
- 12 joint actions

## Why The Student Is History-Conditioned

The history path is the point.

The student is built around the idea that temporal context helps with:

- disturbances
- hidden mismatch
- recovery after transients

without requiring privileged inputs at inference.

## Design Intent

The design goal was not to maximize architectural novelty.

The design goal was to keep this repo:

- deployable
- inspectable
- exportable
- small enough to understand quickly
