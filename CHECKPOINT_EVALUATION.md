# Checkpoint Evaluation

The public repository does not include large checkpoint files. Paper-protocol
evaluation can run in either of two ways:

1. Train score sources with the public scripts, then generate a manifest with
   `tools/build_eval_manifest.py`.
2. Download a released checkpoint package and pass its manifest explicitly to
   `scripts/run_supplement_group_eval.sh`.

Checkpoint evaluation follows the same paper protocol as fresh training. Each
checkpoint is treated as one score source and is evaluated under the formal
fixed `0.5` (`-F`) and validation-selected H/M/T group (`-G`) operating rules.
Fresh score-source training runs for up to 30 epochs with validation-mAP early
stopping, and the best checkpoint is selected by validation mAP.

The formal group-boundary protocol is validation-only H/M/T group search on:

```text
0.10, 0.15, ..., 0.70
```

with step `0.05`. Final `-G` test metrics always use validation-selected
thresholds. Fixed `0.5` (`-F`) is also a formal paper operating rule. Global
and classwise thresholds are diagnostic only and are not included in
paper-facing tables.

## From Fresh Training

```bash
python tools/build_eval_manifest.py \
  --runs outputs/runs \
  --out artifacts/eval_manifest_seed2026.json

GPU_LIST=0 bash scripts/run_supplement_group_eval.sh \
  --manifest artifacts/eval_manifest_seed2026.json
```

## From Released Checkpoints

Place checkpoint files exactly where the released manifest points, then run:

```bash
GPU_LIST=0 bash scripts/run_supplement_group_eval.sh \
  --manifest path/to/released_manifest.json
```

`--allow-legacy-metadata-mismatch` is non-default. Use it only for archived
internal checkpoints after verifying that class mapping and split metadata match
the paper protocol.
