# Dependencies

This repo is intentionally smaller than the internal research codebase. It
publishes the validated Go2 blind rough-terrain AsymPPO lane, plus enough
training, export, validation, and bring-up code to reproduce that lane.

## Included In This Repo

```text
go2_rough/
  IsaacLab task registration, environment configs, model configs, and the
  blind-history AsymPPO actor-critic.

scripts/
  Training entry points, IsaacLab user wrapper, task doctor, and MuJoCo helper.

scripts/deploy/
  Local export, bundle validation, deploy-side Isaac rehearsal, MuJoCo bridge,
  read-only DDS probe, realtime monitor, and Python hardware bring-up helper.

assets/robots/go2/
  The Go2 USD asset contract used by this public baseline.

docs/
  Reproduction, training, validation, deployment, and limitation notes.
```

## External Dependencies

Install these outside the repo:

```text
Isaac Sim / IsaacLab
  Required for training and IsaacLab playback/rehearsal.

MuJoCo Python package
  Required for local sim2sim validation.

Go2 MuJoCo scene XML
  Required for MuJoCo validation. Set GO2_MUJOCO_MODEL or pass --model-path.

Unitree SDK2 Python bindings
  Required only for read-only DDS probes, realtime monitoring, and Python
  hardware bring-up.

Robot network access
  Required only for real hardware tests. Ethernet should be validated before
  Wi-Fi.
```

This repo does not vendor IsaacLab, MuJoCo Menagerie, Unitree SDKs, raw
training checkpoints, or generated exported bundles.

## IsaacLab Install Rule

Install this package into Isaac Sim Python with no dependency resolution:

```bash
$ISAACLAB_ROOT/_isaac_sim/python.sh -m pip install --user --no-deps -e .
```

Do not add `torch` to `pyproject.toml` and do not run normal dependency
resolution inside Isaac Sim Python. IsaacLab owns the compatible PyTorch,
CUDA, and NCCL stack.

## Asset Contract

The bundled/default Go2 asset is:

```text
assets/robots/go2/go2.usd
```

Default IsaacLab body naming:

```text
base body: base
foot/contact bodies: .*_foot
height scanner prim: {ENV_REGEX_NS}/Robot/base
```

If you intentionally use a different USD, set:

```bash
export GO2_USD_PATH=/path/to/custom/go2.usd
export GO2_BASE_BODY_NAME=base_link
export GO2_FOOT_BODY_REGEX='.*_calf'
export GO2_HEIGHT_SCANNER_PRIM='{ENV_REGEX_NS}/Robot/base_link'
```

Then rerun:

```bash
bash scripts/isaaclab_user.sh -p scripts/doctor_isaaclab.py
```

## Deployment Artifact Boundary

Training checkpoints and exported deployment bundles are generated artifacts.
They are not committed by default.

The expected generated bundle location in the docs is:

```text
artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate/
```

It should contain:

```text
bundle_manifest.json
*.torchscript.pt
*.onnx
*.export_metadata.json
*.deploy_config.json
*.deploy.yaml
export_request.json
```
