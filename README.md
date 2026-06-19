# Trakr RL

Isaac Lab workspace for Trakr locomotion training and evaluation.

This repository is a **Trakr port** of the original Unitree RL Lab codebase.
The original authors and upstream project are:

- **Unitree Robotics**
- original project: `unitree_rl_lab`

This repo adapts that structure and training flow for the Trakr robot and its
locomotion tasks.


# Multi-Terrain Demonstration

<table>
<tr>
<td align="center" width="50%">
<img src="assets/videos/rl-video-step-0 (2).gif" width="100%"></video>
<br><br>
<b>Flat Terrain Omnidirectional Locomotion</b>
</td>

<td align="center" width="50%">
<img src="assets/videos/rl-video-step-0 (2).gif" width="100%"></video>
<br><br>
<b>Rough Terrain Omnidirectional Locomotion</b>
</td>
</tr>

<tr>
<td align="center" width="50%">
<img src="assets/videos/rl-video-step-0 (2).gif" width="100%"></video>
<br><br>
<b>Sloped Terrain Omnidirectional Locomotion</b>
</td>

<td align="center" width="50%">
<img src="assets/videos/rl-video-step-0 (2).gif" width="100%"></video>
<br><br>
<b>Stair Climbing Demonstration</b>
</td>
</tr>
</table>

## Installation

Prerequisite: install Isaac Lab separately and use its Python environment.

If the Trakr USD path changes, update it in:

- `source/trakr_rl/trakr_rl/assets/robots/trakr.py`

## Directory Structure

```text
.
├── scripts/
│   └── rsl_rl/
│       ├── train.py
│       └── play.py
├── source/
│   └── trakr_rl/
│       └── trakr_rl/
│           ├── assets/
│           │   └── robots/
│           │       └── trakr.py
│           └── tasks/
│               └── locomotion/
│                   └── robots/
│                       ├── trakr/
│                       └── trakr-rough/
├── logs/
└── trakr_rl.sh
```

Key pieces:

- `source/trakr_rl/trakr_rl/assets/robots/trakr.py`: Trakr USD and actuator configuration
- `source/trakr_rl/trakr_rl/tasks/locomotion/robots/trakr/`: flat-terrain Trakr task
- `source/trakr_rl/trakr_rl/tasks/locomotion/mdp/rewards.py`: Reward Functions definition
- `source/trakr_rl/trakr_rl/tasks/locomotion/robots/trakr-rough/`: rough-terrain Trakr task
- `source/trakr_rl/trakr_rl/tasks/locomotion/robots/[TASK-NAME]/velocity_env_cfg.py` : Task specific config file
- `scripts/rsl_rl/train.py`: RSL-RL training entrypoint
- `scripts/rsl_rl/play.py`: checkpoint visualization entrypoint
- `trakr_rl.sh`: wrapper for install, list, train, and play

## Training

Train flat-terrain locomotion:

```bash
./trakr_rl.sh -t --task Trakr-Velocity
```

Train rough-terrain locomotion:

```bash
./trakr_rl.sh -t --task Trakr-Velocity-Rough
```

Warm-start rough-terrain training from a flat-terrain checkpoint:

```bash
./trakr_rl.sh -t \
  --task Trakr-Velocity-Rough \
  --resume \
  --checkpoint /abs/path/to/model.pt
```

## Playing

Play a checkpoint:

```bash
./trakr_rl.sh -p \
  --task Trakr-Velocity \
  --checkpoint /abs/path/to/model.pt
```

Play a rough-terrain checkpoint with one environment in real time:

```bash
./trakr_rl.sh -p \
  --task Trakr-Velocity-Rough \
  --checkpoint /abs/path/to/model.pt \
  --num_envs 1 \
  --real-time
```
