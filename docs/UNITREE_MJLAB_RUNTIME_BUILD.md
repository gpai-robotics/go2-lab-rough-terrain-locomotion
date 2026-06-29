# Unitree RL MJLab Runtime Build And Recovery

This document covers the C++ binaries behind the active AsymPPO deployment
path:

- `go2_ctrl` — FSM controller (`Passive -> FixStand -> Velocity`)
- `unitree_mujoco` — MuJoCo simulator that speaks the same DDS interface

Both live under `reference_repos/unitree_rl_mjlab/` after you clone/build the
external runtime and are launched by
`scripts/deploy/run_unitree_mjlab_sim_deploy.sh`.

This repo does not vendor `unitree_rl_mjlab` or `unitree_sdk2`. It owns the
patch and wrapper flow needed to reproduce the validated controller behavior.

## What Depends On What

```text
Frozen bundle (repo)
  artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate/
       |
       v
Materializer bridge
  scripts/deploy/materialize_unitree_rl_lab_layout.py
  scripts/deploy/prepare_unitree_rl_mjlab_go2_runtime.py
       |
       v
Staged runtime (unitree_rl_mjlab tree)
  reference_repos/unitree_rl_mjlab/deploy/robots/go2/config/policy/velocity/<candidate>/
       |
       v
C++ binaries (this document)
  deploy/robots/go2/build/go2_ctrl
  simulate/build/unitree_mujoco
```

If sim or hardware fails immediately with "No such file or directory" for
either binary, start here.

## One-Command Build

From the repo root:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git \
  reference_repos/unitree_rl_mjlab

bash scripts/deploy/build_unitree_mjlab_runtime.sh all
```

This will:

1. clone `unitree_sdk2` into `reference_repos/unitree_sdk2` if missing
2. install it to `reference_repos/unitree_sdk2/install` (repo-local, no sudo)
3. build `go2_ctrl`
4. build `unitree_mujoco`
5. verify both binaries exist and resolve shared libraries

Before rebuilding a freshly restored `unitree_rl_mjlab` mirror, apply the
repo-owned runtime patch:

```bash
cd reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd ../..
```

This patch restores the local sim/controller contract that is not present in
upstream `unitree_rl_mjlab`:

- `UNITREE_MJLAB_AUTOSTART=1` scripted transition injection
- `UNITREE_MJLAB_VX/VY/YAW` fixed simulated command injection
- `UNITREE_MJLAB_TELEOP=1` keyboard teleop in the controller terminal
- `unitree_mujoco --use_joystick=0/1` CLI override support
- ONNX Runtime dynamic batch shape handling for exported policies with `-1`
  input/output batch dimensions

Without this patch, `controller` waits for physical/simulated joystick button
events and the two-terminal scripted simulation workflow does not work. Without
the ONNX shape fix, `Velocity` can abort at startup with:
`tried creating tensor with negative value in shape`.

Partial rebuilds:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh sdk
bash scripts/deploy/build_unitree_mjlab_runtime.sh controller
bash scripts/deploy/build_unitree_mjlab_runtime.sh sim
bash scripts/deploy/build_unitree_mjlab_runtime.sh verify
```

## Full Recovery Checklist

Use this order when the deploy path is broken after a clean checkout, machine
migration, or accidental deletion.

### 1. System packages

```bash
sudo apt install -y cmake g++ build-essential \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libfmt-dev
```

### 2. C++ binaries

```bash
cd /path/to/go2-lab-rough-terrain-locomotion
cd reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd ../..
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
```

### 3. Materialize and activate the policy

The materializer bridge is included in:

```text
scripts/deploy/materialize_unitree_rl_lab_layout.py
```

Then stage the frozen bundle:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Expected pass signals:

- `Velocity.policy_dir` points at
  `config/policy/velocity/go2_blind_rough_asymppo_mjlab_v1_candidate`
- staged `exported/policy.onnx` exists
- staged `params/deploy.yaml` has `policy_history` with `history_length: 100`

### 4. Run sim or hardware

```bash
# terminal 1
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller

# terminal 2
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Hardware:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

Canonical copy-paste commands remain in `docs/RUN_COMMANDS.md`.

## Why unitree_sdk2 Is Repo-Local

`unitree_rl_mjlab` expects `unitree_sdk2` at `/opt/unitree_robotics` on a
stock Unitree workstation. This project instead keeps a local install under:

```text
reference_repos/unitree_sdk2/install
```

The build script wires that prefix into both CMake projects. If you already
have a system install at `/opt/unitree_robotics`, the script will use it
automatically and skip the local SDK build.

To force a custom prefix:

```bash
export UNITREE_SDK2_INSTALL=/path/to/your/install
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
```

## Common Failures

### `go2_ctrl: No such file or directory`

The controller was never built or the build directory was deleted.

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh controller
```

### `unitree_mujoco: No such file or directory`

Same for the simulator:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh sim
```

### `unitree/dds_wrapper/robots/go2/go2.h: No such file or directory`

`unitree_sdk2` headers are missing. Rebuild the SDK:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh sdk
bash scripts/deploy/build_unitree_mjlab_runtime.sh controller
```

### `find_package(unitree_sdk2)` fails for `unitree_mujoco`

The simulator CMake project cannot see the SDK. Ensure
`reference_repos/unitree_sdk2/install/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake`
exists, then rebuild sim.

### `ImportError: cannot import name 'materialize_bundle'`

The active materializer was reduced to a copy-only stub. Restore
`materialize_bundle()` from the archive copy, then rerun `activate`.

### `activate` passes but policy behaves like the stock flat controller

`config.yaml` may still point at the generic `v0` runtime instead of the AsymPPO
candidate. Rerun:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Check that `Velocity.policy_dir` is
`config/policy/velocity/go2_blind_rough_asymppo_mjlab_v1_candidate`.

### Shared library errors at runtime (`libddsc.so not found`)

Rebuild with the wrapper script; it sets an rpath to the local SDK install.
Verify with:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh verify
ldd reference_repos/unitree_rl_mjlab/deploy/robots/go2/build/go2_ctrl
```

## Expected Binary Locations

```text
reference_repos/unitree_rl_mjlab/deploy/robots/go2/build/go2_ctrl
reference_repos/unitree_rl_mjlab/simulate/build/unitree_mujoco
```

These paths are hard-coded in `scripts/deploy/run_unitree_mjlab_sim_deploy.sh`.
