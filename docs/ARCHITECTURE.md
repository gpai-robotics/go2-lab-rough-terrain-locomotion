# Architecture

## Policy Type

This repo publishes a blind rough-terrain asymmetric PPO controller for Go2.

The actor is directly deployable. The critic gets extra information only to make
PPO training easier.

## Actor

The actor receives deployable signals only:

- base angular velocity
- projected gravity
- commanded velocity
- joint position relative to default pose
- joint velocity
- previous action
- flattened history of the same terms

The actor does not receive:

- base linear velocity
- height scan
- terrain class
- friction
- mass
- motor scale

## Critic

The critic receives the actor inputs plus privileged signals:

- base linear velocity
- terrain height scan
- tracked static and dynamic friction
- tracked base mass ratio
- joint stiffness and damping scale

These privileged signals are training-only.

## Why This Shape

The design keeps deployment simple and measurable:

- no latent estimator at runtime
- no online adaptation module
- no exteroceptive terrain sensing
- one actor path that can be exported and deployed directly

Robustness is expected to come primarily from terrain diversity, command
diversity, dynamics randomization, action history, and push disturbance training.
