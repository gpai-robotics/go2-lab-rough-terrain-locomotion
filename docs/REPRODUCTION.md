## Reproduction

This repo provides a standalone Go2 rough-terrain path:

1. train a flat MJLAB-contract prior,
2. train the rough blind AsymPPO policy,
3. export a deployment bundle,
4. validate parity locally,
5. run MuJoCo sim2sim with the repo-owned Python bridge,
6. optionally stage the Unitree RL MJLAB C++ FSM runtime,
7. run read-only DDS checks,
8. bring up hardware only after the earlier gates pass.

## Public Prerequisites

Expected layout:

```text
workspace/
  IsaacLab/
  go2-lab-rough-terrain-locomotion/
```

Environment variables used throughout:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export REPO=/path/to/go2-lab-rough-terrain-locomotion
```

Additional prerequisites by stage:

- MuJoCo sim2sim:
  - Python package `mujoco`
  - a Go2 MuJoCo scene XML, for example from `mujoco_menagerie`
  - `GO2_MUJOCO_MODEL=/path/to/unitree_go2/scene.xml` or `--model-path`
- Hardware:
  - a `unitree_sdk2py` checkout or install
  - a valid network path to the robot
  - an optional mode-switch helper script if you want the runner to switch into low-level mode for you
- Unitree RL MJLAB C++ FSM runtime:
  - external `unitree_rl_mjlab` clone under `reference_repos/unitree_rl_mjlab`
  - repo patch applied before build
  - `unitree_sdk2` C++ install, either system-wide or wrapper-built locally

Install this package into Isaac Sim Python:

```bash
cd "$REPO"
$ISAACLAB_ROOT/_isaac_sim/python.sh -m pip install --user --no-deps -e .
```

Do not use dependency resolution inside Isaac Sim Python. IsaacLab already
provides the compatible `torch`, CUDA, and `gymnasium` stack.

## Step 1: Preflight

Validate the IsaacLab side before training:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/doctor_isaaclab.py
bash scripts/isaaclab_user.sh -p scripts/check_tasks.py
```

If you train against a custom Go2 USD, set the asset contract first:

```bash
export GO2_USD_PATH=/path/to/custom/go2.usd
export GO2_BASE_BODY_NAME=base_link
export GO2_FOOT_BODY_REGEX='.*_calf'
export GO2_HEIGHT_SCANNER_PRIM='{ENV_REGEX_NS}/Robot/base_link'
```

Then rerun `doctor_isaaclab.py`.

## Step 2: Train the Flat Prior

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/train_flat_prior.py \
  --headless \
  --log-dir ~/isaaclab_logs/go2_flat_mjlab_prior_v1
```

RSL-RL writes checkpoints under the chosen log directory. The warmstart input
for Stage 2 is typically the final numbered model file, for example:

```text
~/isaaclab_logs/go2_flat_mjlab_prior_v1/model_1499.pt
```

## Step 3: Train the Rough AsymPPO Policy

Warmstarted run:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/train_asymppo.py \
  --flat-prior-checkpoint ~/isaaclab_logs/go2_flat_mjlab_prior_v1/model_1499.pt \
  --headless \
  --log-dir ~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1
```

Scratch run:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/train_asymppo.py \
  --headless \
  --log-dir ~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1
```

The expected export input after a successful run is the final rough checkpoint,
for example:

```text
~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1/model_1999.pt
```

## Step 4: Export the Deployment Bundle

This repo ships its own standalone export core at
`scripts/deploy/export_policy.py`.

The working AsymPPO export shape is:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/deploy/export_policy.py \
  --policy-name go2_blind_rough_asymppo_mjlab_v1_candidate \
  --checkpoint ~/isaaclab_logs/go2_blind_rough_asymppo_mjlab_v1/model_1999.pt \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --phase blind-rough-mjlab-asymppo-v1 \
  --policy-kind blind_history_policy \
  --observation-groups policy,policy_history \
  --format torchscript \
  --format onnx
```

Expected bundle artifacts include:

```text
bundle_manifest.json
*.torchscript.pt
*.onnx
*.export_metadata.json
*.deploy_config.json
*.deploy.yaml
export_request.json
```

## Step 5: Validate the Bundle

Structural validation:

```bash
cd "$REPO"
python scripts/deploy/validate_bundle.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate
```

Checkpoint-vs-export parity:

```bash
cd "$REPO"
python scripts/deploy/validate_policy_inference_parity.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate
```

Isaac deploy-side rehearsal:

```bash
cd "$REPO"
bash scripts/isaaclab_user.sh -p scripts/deploy/play_deploy_policy.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --num-envs 16 \
  --max-steps 500 \
  --compare-source
```

## Step 6: MuJoCo Sim2Sim

Point the runtime at a Go2 MuJoCo scene:

```bash
export GO2_MUJOCO_MODEL=/path/to/unitree_go2/scene.xml
```

Run the local deployment validation gate:

```bash
cd "$REPO"
python scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --model-path "$GO2_MUJOCO_MODEL"
```

Run an actual MuJoCo rollout:

```bash
cd "$REPO"
python scripts/deploy/run_sim2sim.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --model-path "$GO2_MUJOCO_MODEL" \
  --execute-runtime \
  --command-x 0.5 \
  --max-steps 900
```

## Step 7: Optional Unitree RL MJLAB C++ FSM Runtime

Use this path when you want the two-terminal sim/controller workflow and the
same FSM-style runtime used for the validated hardware deployment.

Clone and patch the external runtime:

```bash
cd "$REPO"
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git \
  reference_repos/unitree_rl_mjlab

cd reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd "$REPO"
```

Build, stage, and validate:

```bash
cd "$REPO"
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Run controller and simulator in separate terminals:

```bash
# terminal 1
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

```bash
# terminal 2
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Detailed notes are in `docs/UNITREE_MJLAB_RUNTIME_BUILD.md`.

## Step 8: Read-Only DDS Probe

Before any controller publishes `rt/lowcmd`, validate the transport with the
read-only probe:

```bash
cd "$REPO"
python scripts/deploy/probe_go2_readonly.py \
  --net-if "$GO2_ETH_IF" \
  --duration-s 5 \
  --subscribe-sport \
  --unitree-sdk-root /path/to/unitree_sdk2py
```

## Step 9: Hardware Bring-Up

Start the read-only monitor in one terminal:

```bash
cd "$REPO"
python scripts/deploy/monitor_go2_realtime.py \
  --net-if "$GO2_ETH_IF" \
  --subscribe-lowcmd \
  --jsonl-out artifacts/go2_realtime_monitor/asymppo_walk.jsonl \
  --unitree-sdk-root /path/to/unitree_sdk2py
```

Dry-run the hardware contract first:

```bash
cd "$REPO"
python scripts/deploy/run_go2_hardware.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --net-if "$GO2_ETH_IF" \
  --dry-run
```

Then stance-only:

```bash
cd "$REPO"
python scripts/deploy/run_go2_hardware.py \
  --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --net-if "$GO2_ETH_IF" \
  --unitree-sdk-root /path/to/unitree_sdk2py \
  --mode-switch-script /path/to/mode_switch.py \
  --stance-only \
  --duration-s 5
```

Only after that should you run the policy controller itself.
