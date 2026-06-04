# Setup

This repo assumes you already have IsaacLab installed.

## Expected Layout

```text
workspace/
  IsaacLab/
  go2-lab-rough-terrain-locomotion/
```

## Install

Set your IsaacLab root:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
```

From the repo root:

```bash
$ISAACLAB_ROOT/_isaac_sim/python.sh -m pip install -e .
```

## Sanity Check

Run the task registration check:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/check_tasks.py
```

The tasks exposed here are:

- `RMA-Go2-PrivilegedTeacher-Rough-StageA`
- `RMA-Go2-BlindHistory-Rough-StageA`

## Entrypoints

- [`../scripts/train_teacher.py`](../scripts/train_teacher.py)
- [`../scripts/train_student.py`](../scripts/train_student.py)
- [`../scripts/export_policy.py`](../scripts/export_policy.py)
- [`../scripts/validate_bundle.py`](../scripts/validate_bundle.py)
- [`../scripts/eval_deploy.py`](../scripts/eval_deploy.py)

## Import Note

Task registration occurs when Python imports `go2_rough`, so scripts should
import the package before relying on Gym task ids.

## Reproducibility Boundary

This repo is not a standalone simulator distribution.

It assumes:

- a working IsaacLab installation
- IsaacLab-compatible `isaaclab_tasks`, `isaaclab_rl`, and `rsl_rl` packages in
  that environment
- the reader is running the repo scripts through the IsaacLab launcher
