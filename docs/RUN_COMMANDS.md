# Production Run Commands

```bash
export REPO=/path/to/go2-lab-rough-terrain-locomotion
export ISAACLAB_ROOT=/path/to/IsaacLab
export GO2_ETH_IF=eth0
export MUJOCO_PYTHON=python
export TERRAIN_STEPS_CKPT=~/isaaclab_logs/go2_terrain_locomotion_stairs_v1/model_2999.pt
export TERRAIN_BUNDLE=$REPO/artifacts/exported/go2_terrain_locomotion_steps_v1_candidate
export GO2_MUJOCO_MODEL=/path/to/unitree_go2/scene.xml
export UNITREE_SDK2PY_ROOT=$REPO/third_party/unitree_sdk2py
```

## Preflight

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/doctor_isaaclab.py
bash scripts/isaaclab_user.sh -p scripts/check_tasks.py
```

## Stage 1: Flat Prior

```bash
bash scripts/isaaclab_user.sh -p scripts/train_flat_prior.py \
  --task Go2-Terrain-Flat-Prior-V1 \
  --headless \
  --log-dir ~/isaaclab_logs/go2_terrain_flat_prior_v1
```

## Stage 2: Rough / Slopes

```bash
bash scripts/isaaclab_user.sh -p scripts/train_terrain_policy.py \
  --stage rough \
  --flat-prior-checkpoint ~/isaaclab_logs/go2_terrain_flat_prior_v1/model_1499.pt \
  --headless \
  --log-dir ~/isaaclab_logs/go2_terrain_locomotion_rough_v1
```

## Stage 3: Stairs / Inverted Stairs

```bash
bash scripts/isaaclab_user.sh -p scripts/train_terrain_policy.py \
  --stage stairs \
  --rough-checkpoint ~/isaaclab_logs/go2_terrain_locomotion_rough_v1/model_1999.pt \
  --headless \
  --log-dir ~/isaaclab_logs/go2_terrain_locomotion_stairs_v1
```

## Export Candidate

```bash
bash scripts/isaaclab_user.sh -p scripts/deploy/export_policy.py \
  --policy-name go2_terrain_locomotion_steps_v1_candidate \
  --checkpoint "$TERRAIN_STEPS_CKPT" \
  --task Go2-Terrain-Locomotion-Stairs-V1 \
  --phase terrain-locomotion-stairs-v1 \
  --bundle-dir "$TERRAIN_BUNDLE" \
  --policy-kind blind_history_policy \
  --observation-groups policy,policy_history \
  --policy-history-length 100 \
  --format torchscript \
  --format onnx
```

## Deployment Gate

```bash
python scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir "$TERRAIN_BUNDLE" \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --expected-action-dim 12 \
  --model-path "$GO2_MUJOCO_MODEL"
```

## MuJoCo Terrain Validation

```bash
bash scripts/deploy/run_terrain_locomotion_model5099_mujoco_validation.sh
```

## Unitree MJLAB C++ FSM

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Two-terminal simulation:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Hardware preflight and launch:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

Remote sequence:

```text
L2 + up  -> FixStand
R2 + A   -> Velocity policy
L2 + B   -> Passive / stop
```
