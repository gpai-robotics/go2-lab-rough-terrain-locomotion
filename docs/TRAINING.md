# Training

## Stage 1: Flat Prior

Train a flat omnidirectional prior under the same deployable actor observation
contract used by the rough policy:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export GO2_USD_PATH=/path/to/go2.usd

$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_flat_prior.py --headless
```

The flat prior is used only as an actor warmstart. It is not deployed as the
rough-terrain controller.

## Stage 2: Rough AsymPPO

Train the rough policy:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_asymppo.py \
  --flat-prior-checkpoint /path/to/flat_prior_checkpoint.pt \
  --headless
```

If you do not want warmstart:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_asymppo.py --headless
```

## Important Ranges

Command curriculum target:

```text
vx:  [-0.8, 0.8] m/s
vy:  [-0.3, 0.3] m/s
yaw: [-0.6, 0.6] rad/s
```

Motor gain randomization:

```text
stiffness scale: [0.6, 1.4]
damping scale:   [0.6, 1.4]
```

Push disturbance:

```text
interval: 6-10 s
vx impulse command:  [-0.35, 0.35]
vy impulse command:  [-0.35, 0.35]
yaw impulse command: [-0.4, 0.4]
```

## Notes

The successful branch did not rely on extreme gain randomization. Wider gain
randomization hurt learning in our tests, so this public path keeps the narrower
deployment-proven envelope.
