# Active AsymPPO Run Commands

This document contains the standalone command path for the active
AsymPPO lane.

## Environment

```bash
export REPO=/path/to/go2-lab-rough-terrain-locomotion
export ISAACLAB_ROOT=/path/to/IsaacLab
export GO2_ETH_IF=eth0
export ASYMPPO_CKPT=~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1/model_1999.pt
export ASYMPPO_BUNDLE=$REPO/artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate
export GO2_MUJOCO_MODEL=/path/to/unitree_go2/scene.xml
export UNITREE_SDK2PY_ROOT=/path/to/unitree_sdk2py
```

## Workstation Preflight

Run these from this repo:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/doctor_isaaclab.py
bash scripts/isaaclab_user.sh -p scripts/check_tasks.py
```

## Training

Flat prior:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/train_flat_prior.py \
  --headless \
  --log-dir ~/isaaclab_logs/go2_flat_mjlab_prior_v1
```

Rough AsymPPO:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/train_asymppo.py \
  --flat-prior-checkpoint ~/isaaclab_logs/go2_flat_mjlab_prior_v1/model_1499.pt \
  --headless \
  --log-dir ~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1
```

## Bundle Export

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/deploy/export_policy.py \
  --policy-name go2_blind_rough_asymppo_mjlab_v1_candidate \
  --checkpoint "$ASYMPPO_CKPT" \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --phase blind-rough-mjlab-asymppo-v1 \
  --policy-kind blind_history_policy \
  --observation-groups policy,policy_history \
  --format torchscript \
  --format onnx
```

## Parity And Rehearsal

```bash
cd "$REPO"
python scripts/deploy/validate_bundle.py \
  --bundle-dir "$ASYMPPO_BUNDLE"
```

```bash
cd "$REPO"
python scripts/deploy/validate_policy_inference_parity.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --output-dir artifacts/deployment_validation/golden_inference
```

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/deploy/play_deploy_policy.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --num-envs 16 \
  --max-steps 500 \
  --compare-source
```

## MuJoCo Gate

```bash
cd "$REPO"
python scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --model-path "$GO2_MUJOCO_MODEL"
```

Execute the local MuJoCo bridge:

```bash
cd "$REPO"
python scripts/deploy/run_sim2sim.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --model-path "$GO2_MUJOCO_MODEL" \
  --execute-runtime \
  --command-x 0.5 \
  --max-steps 900
```

## Hardware Bring-Up

Read-only DDS probe first:

```bash
cd "$REPO"
python scripts/deploy/probe_go2_readonly.py \
  --net-if "$GO2_ETH_IF" \
  --duration-s 5 \
  --subscribe-sport \
  --unitree-sdk-root "$UNITREE_SDK2PY_ROOT"
```

Read-only monitor:

```bash
cd "$REPO"
python scripts/deploy/monitor_go2_realtime.py \
  --net-if "$GO2_ETH_IF" \
  --subscribe-lowcmd \
  --jsonl-out artifacts/go2_realtime_monitor/asymppo_walk.jsonl \
  --unitree-sdk-root "$UNITREE_SDK2PY_ROOT"
```

Dry-run hardware contract:

```bash
cd "$REPO"
python scripts/deploy/run_go2_hardware.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --net-if "$GO2_ETH_IF" \
  --dry-run
```

Stance-only bring-up:

```bash
cd "$REPO"
python scripts/deploy/run_go2_hardware.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --net-if "$GO2_ETH_IF" \
  --unitree-sdk-root "$UNITREE_SDK2PY_ROOT" \
  --mode-switch-script /path/to/mode_switch.py \
  --stance-only \
  --duration-s 5
```

## Unitree RL MJLAB C++ FSM Runtime

This is the closest runtime to the validated deployment path. It requires the
external `unitree_rl_mjlab` C++ repository, but the patch/build/activation
flow is owned here.

Clone and patch the external runtime:

```bash
cd "$REPO"
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git \
  reference_repos/unitree_rl_mjlab

cd reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd "$REPO"
```

Build the controller and simulator:

```bash
cd "$REPO"
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
```

Stage the exported AsymPPO bundle into the C++ runtime and validate the FSM
contract:

```bash
cd "$REPO"
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Run sim/controller in two terminals:

```bash
# terminal 1
cd "$REPO"
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

```bash
# terminal 2
cd "$REPO"
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

C++ FSM hardware path:

```bash
cd "$REPO"
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

While hardware is running, monitor LowState/LowCmd in another terminal:

```bash
cd "$REPO"
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh monitor ethernet asymppo_walk
```

Detailed runtime recovery notes:

```text
docs/UNITREE_MJLAB_RUNTIME_BUILD.md
```
