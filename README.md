# Trakr RL

Isaac Lab workspace for Trakr locomotion training and evaluation.

This repository is a **Trakr port** of the original Unitree RL Lab codebase.
The original authors and upstream project are:

- **Unitree Robotics**
- original project: `unitree_rl_lab`

This repo adapts that structure and training flow for the Trakr robot and its
locomotion tasks.


## Multi-Terrain Demonstration

<table>
<tr>
<td align="center" width="50%">
<img src="assets/videos/flat.gif" width="100%"></video>
<br><br>
<b>Flat Terrain Omnidirectional Locomotion</b>
</td>

<td align="center" width="50%">
<img src="assets/videos/rough.gif" width="100%"></video>
<br><br>
<b>Rough Terrain Omnidirectional Locomotion</b>
</td>
</tr>

<tr>
<td align="center" width="50%">
<img src="assets/videos/slopes.gif" width="100%"></video>
<br><br>
<b>Sloped Terrain Omnidirectional Locomotion</b>
</td>

<td align="center" width="50%">
<img src="assets/videos/stairs.gif" width="100%"></video>
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
│                       ├── trakr-rough/
|                       ├── trakr-stairs/
|                       ├── trakr-blocks/
|                       └── trakr-eval/
├── logs/
└── trakr_rl.sh
```

Key pieces:

- `source/trakr_rl/trakr_rl/assets/robots/trakr.py`: Trakr USD and actuator configuration
- `source/trakr_rl/trakr_rl/tasks/locomotion/mdp/rewards.py`: Reward Functions definition
- `source/trakr_rl/trakr_rl/utils/ood_merics.py`: OOD Metrics definition
- `source/trakr_rl/trakr_rl/tasks/locomotion/robots/[TASK-NAME]/velocity_env_cfg.py` : Task specific config file
- `scripts/rsl_rl/train.py`: RSL-RL training entrypoint
- `scripts/rsl_rl/play.py`: checkpoint visualization entrypoint
- `scripts/rsl_rl/teleop.py` : teleoperation with trained checkpoint entrypoint
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

Train Stair Climbing/Descending:

```bash
./trakr_rl.sh -t --task Trakr-Velocity-Stairs
```

Warm-start training from a pre-trained checkpoint:

```bash
./trakr_rl.sh -t \
  --task [TASK-NAME] \
  --resume \
  --checkpoint /abs/path/to/model.pt
```

## Playing

Play a checkpoint:

```bash
./trakr_rl.sh -p \
  --task [TASK-NAME] \
  --checkpoint /abs/path/to/model.pt
```

Play a checkpoint with one environment in real time:

```bash
./trakr_rl.sh -p \
  --task [TASK_NAME] \
  --checkpoint /abs/path/to/model.pt \
  --num_envs 1 \
  --real-time
```

## Teleoperation 

Teleoperate Trakr with a checkpoint 

```bash
./trakr_rl.sh --teleop \
  --task [TASK-NAME] \
  --checkpoint /abs/path/to/model.pt
```

## Documentation
 
| Document | Description |
|---|---|
| [Articulation Setup](docs/articulation.md) | Configuring Trakr as an Isaac Lab articulation |
| [Task Setup](docs/task_setup.md) | Flat and rough terrain task registration and MDP design |
| [Policy & Training](docs/policy_training.md) | PPO architecture and hyperparameter reference |
| [Reward Functions](docs/rewards.md) | Reward terms, tuning rationale, and formulas |
