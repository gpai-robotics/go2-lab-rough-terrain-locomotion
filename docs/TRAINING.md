# Training

This repo keeps the training story intentionally small.

## Published Paths

Two tasks matter in this repo:

- `RMA-Go2-PrivilegedTeacher-Rough-StageA`
- `RMA-Go2-BlindHistory-Rough-StageA`

The teacher is trained first. The student is then trained with that teacher as
its privileged reference during training.

## Teacher

The teacher is the privileged rough-terrain controller.

Its role is:

- consume deployable policy observations
- consume privileged dynamics and terrain information during training
- learn a strong rough-terrain control prior
- provide supervision pressure for the blind student

The teacher matters because it exposes terrain and hidden-dynamics information
during training while the student remains blind at inference.

## Student

The student is the main deployed artifact in this repo.

Its runtime contract is blind:

- current deployable observations
- a fixed window of deployable history

It does not receive terrain or dynamics privilege at inference.

During training, the student is shaped by:

- teacher guidance
- explicit history-path usage
- a deployment-facing observation contract

## Why History Matters

The point of the student is not just "blind locomotion."

The stronger claim is:

- the history pathway is behaviorally load-bearing
- especially under temporal robustness probes such as pushes and hidden changes

That is why this repo keeps a history-conditioned student instead of collapsing
to a stateless blind policy.

## Training Entry Points

- `scripts/train_teacher.py`
- `scripts/train_student.py`

Typical flow:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_teacher.py --headless

$ISAACLAB_ROOT/isaaclab.sh -p scripts/train_student.py \
  --teacher-checkpoint artifacts/checkpoints/teacher_stagea.pt \
  --headless
```

Optional student knobs exposed by the script:

- `--num-envs`
- `--max-iterations`
- `--seed`
- `--log-dir`
- `--teacher-checkpoint`
- `--blind-warmstart`

## Checkpoint Policy

The repo does not currently ship frozen weights inside version control.

So there are two valid workflows here:

1. train teacher and student from scratch
2. supply checkpoints from your own workspace

If you are following the export or evaluation flow, make sure the checkpoint
paths you pass actually exist in your local workspace.

The repo surface is meant to stay understandable by a new reader in one pass.
