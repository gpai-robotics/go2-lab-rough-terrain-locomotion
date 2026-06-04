# Go2 Lab Rough Terrain Locomotion

Blind history-conditioned rough-terrain locomotion for Unitree Go2, focused on
one deployment-facing line.

This repo is intentionally narrow:

- one privileged teacher path
- one blind deployable student path
- one export and parity-validation path
- one deployment-oriented evaluation path

## What This Repo Shows

The main question here is:

> How strong can a deployable blind-history locomotion policy get on rough
> terrain before introducing heavier online adaptation machinery?

The main line here is a `C1`-style student:

- blind at inference
- history-conditioned
- teacher-guided during training
- validated through export parity and deployment-side rehearsals

## Demo

<table>
  <tr>
    <td align="center"><strong>Boxes</strong></td>
    <td align="center"><strong>Low Friction</strong></td>
  </tr>
  <tr>
    <td><a href="assets/clips/blind_history_boxes.mp4"><img src="assets/thumbs/blind_history_boxes.gif" alt="Blind-history baseline on box terrain" width="100%"></a></td>
    <td><a href="assets/clips/blind_history_low_friction.mp4"><img src="assets/thumbs/blind_history_low_friction.gif" alt="Blind-history baseline on low-friction terrain" width="100%"></a></td>
  </tr>
  <tr>
    <td align="center"><strong>Push Disturbances</strong></td>
    <td align="center"><strong>Downstairs</strong></td>
  </tr>
  <tr>
    <td><a href="assets/clips/blind_history_push_disturbance.mp4"><img src="assets/thumbs/blind_history_push_disturbance.gif" alt="Blind-history baseline under push disturbances" width="100%"></a></td>
    <td><a href="assets/clips/blind_history_downstairs.mp4"><img src="assets/thumbs/blind_history_downstairs.gif" alt="Blind-history baseline descending stairs" width="100%"></a></td>
  </tr>
</table>

### MuJoCo Runtime

<table>
  <tr>
    <td align="center"><strong>Nominal Runtime</strong></td>
  </tr>
  <tr>
    <td><a href="assets/clips/mujoco_nominal_runtime.mp4"><img src="assets/thumbs/mujoco_nominal_runtime.gif" alt="MuJoCo nominal runtime rehearsal for the blind-history policy" width="100%"></a></td>
  </tr>
</table>

## Key Takeaways

| Axis | Result | Why it matters |
| --- | --- | --- |
| Deploy-time inputs | `policy [48]` + `policy_history [4800]` | The runtime contract stays blind and explicit |
| Export parity | mean abs diff `6.0e-09` | TorchScript export matched the source policy essentially exactly |
| Isaac deploy rehearsal | no terminations in the canonical rehearsal | The exported policy remained stable in the Isaac-side deployment surface |
| MuJoCo nominal runtime | velocity error mean `0.1363` | Nominal cross-runtime behavior remained viable |
| Main deployment weakness | lateral push in the continuous corridor suite | The repo shows both strengths and the remaining hole |

These figures come from the main checkpoint and export path used in this repo.

## Deployment Contract

The deployed student consumes only:

```text
policy_obs       [48]
policy_history   [4800]
```

and outputs:

```text
action           [12]
```

Important runtime assumptions:

- history length: `100`
- control rate: `50 Hz`
- no terrain privilege at inference
- no dynamics privilege at inference

## Quickstart

Expected layout:

```text
workspace/
  IsaacLab/
  go2-lab-rough-terrain-locomotion/
```

Set your IsaacLab root once per shell:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
```

Install this package into the IsaacLab Python environment:

```bash
cd go2-lab-rough-terrain-locomotion
$ISAACLAB_ROOT/_isaac_sim/python.sh -m pip install -e .
```

Sanity check the task registration:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/check_tasks.py
```

Train the teacher and student:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_teacher.py --headless

$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_student.py \
  --teacher-checkpoint artifacts/checkpoints/teacher_stagea.pt \
  --headless
```

Export, validate, and evaluate the deployable student:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/export_policy.py \
  --policy-name go2_blind_history_stagea \
  --checkpoint artifacts/checkpoints/student_stagea.pt \
  --bundle-dir artifacts/go2_blind_history_stagea_bundle

$ISAACLAB_ROOT/isaaclab.sh -p scripts/validate_bundle.py \
  --bundle-dir artifacts/go2_blind_history_stagea_bundle

$ISAACLAB_ROOT/isaaclab.sh -p scripts/eval_deploy.py \
  --bundle-dir artifacts/go2_blind_history_stagea_bundle \
  --headless
```

## Checkpoints

This repo contains code, configs, docs, and demo media.

It does not version frozen checkpoints or exported bundles inside the repo
itself.

That means:

- train commands are runnable from source
- export and evaluation commands require valid checkpoint paths

## Important Files

If you want the shortest code-reading path, start here:

- `go2_rough/__init__.py`
  - task registration
- `go2_rough/envs/privileged_teacher_rough_cfg.py`
  - teacher environment entrypoint
- `go2_rough/configs/privileged_teacher_ppo_cfg.py`
  - teacher runner config
- `go2_rough/envs/blind_history_rough_cfg.py`
  - student environment entrypoint
- `go2_rough/configs/blind_history_ppo_cfg.py`
  - student runner config
- `go2_rough/models/privileged_teacher_actor_critic.py`
  - teacher actor-critic
- `go2_rough/models/history_actor_critic.py`
  - blind history-conditioned student actor-critic
- `go2_rough/models/teacher_guided_blind_history_ppo.py`
  - student training algorithm with teacher guidance
- `scripts/train_teacher.py`
  - teacher training entrypoint
- `scripts/train_student.py`
  - student training entrypoint
- `scripts/export_policy.py`
  - deployment bundle export
- `scripts/validate_bundle.py`
  - bundle structural validation
- `scripts/eval_deploy.py`
  - Isaac-side deploy rehearsal

## Media Status

The README media set now includes:

- IsaacLab behavior clips
- one MuJoCo nominal runtime clip

## Registered Tasks

- `RMA-Go2-PrivilegedTeacher-Rough-StageA`
- `RMA-Go2-BlindHistory-Rough-StageA`

Task registration happens when Python imports `go2_rough`.

## Reading Guide

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/TRAINING.md](docs/TRAINING.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/RESULTS.md](docs/RESULTS.md)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
- [docs/SETUP.md](docs/SETUP.md)

## Repo Layout

- `go2_rough/`
  - extracted task registration, env configs, models, and runner configs
- `scripts/`
  - training, export, parity validation, and deployment-eval entrypoints
- `assets/`
  - stable demo media used by the README
- `docs/`
  - architecture, training, evaluation, results, and limitations notes

## Scope Notes

- This repo keeps the focused C1 deployment-facing path, not the full lab notebook.
- The main claim here is the blind-history deployment line.
- Explicit online adaptation is outside the scope of this repo.
