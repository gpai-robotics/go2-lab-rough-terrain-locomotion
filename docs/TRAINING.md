# Training

## Stage 1: Flat Prior

Train a flat omnidirectional prior under the same deployable actor observation
contract used by the rough policy:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export GO2_USD_PATH=/path/to/go2.usd

$ISAACLAB_ROOT/_isaac_sim/python.sh -m pip install --user --no-deps -e .

bash scripts/isaaclab_user.sh -p scripts/doctor_isaaclab.py

bash scripts/isaaclab_user.sh -p scripts/train_flat_prior.py \
  --headless \
  --log-dir ~/isaaclab_logs/go2_flat_mjlab_prior_v1
```

The flat prior is used only as an actor warmstart. It is not deployed as the
rough-terrain controller.

## Stage 2: Rough AsymPPO

Train the rough policy:

```bash
bash scripts/isaaclab_user.sh -p scripts/train_asymppo.py \
  --flat-prior-checkpoint /path/to/flat_prior_checkpoint.pt \
  --headless \
  --log-dir ~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1
```

If you do not want warmstart:

```bash
bash scripts/isaaclab_user.sh -p scripts/train_asymppo.py \
  --headless \
  --log-dir ~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1
```

Do not add a standalone `/` between command arguments. It will be parsed as an
unknown argument by the training script.

For shared `/opt` IsaacLab installs, always launch through
`scripts/isaaclab_user.sh`. It redirects mutable Kit cache/log/temp files to the
current user's home directory and exposes Isaac Sim's bundled CUDA libraries to
PyTorch.

If the doctor reports a user-site `torch`, remove the conflicting user install
before training. The repo itself should be installed with `--no-deps` so it does
not replace IsaacLab's bundled PyTorch/CUDA stack.

## IsaacLab Dependency Boundary

This repository does not vendor or pin PyTorch. IsaacLab/Isaac Sim provides the
compatible `torch`, CUDA and `gymnasium` stack. Install this package into Isaac
Sim Python with:

```bash
$ISAACLAB_ROOT/_isaac_sim/python.sh -m pip install --user --no-deps -e .
```

Do not run plain `pip install -e .` if `pyproject.toml` has dependencies added
locally. Pulling a different PyTorch/NCCL build into `~/.local` can cause errors
such as `libtorch_cuda.so: undefined symbol: ncclCommResume`.

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
