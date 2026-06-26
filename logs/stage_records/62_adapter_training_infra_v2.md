# Stage62 Adapter Training Infra v2

Date: 2026-06-27

## Goal

Add adapter training infrastructure for the new DAVIS all-gap anchor manifest on `/data`: train/val split, best checkpoint selection, resume state, and storage-safe external checkpoints.

## Code

```text
scripts/run_stage62_adapter_training_infra_v2.py
```

## Inputs

```text
experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full/
```

## Verification Commands

GPU status was checked with `nvidia-smi` before each run.

Smoke:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage62_adapter_training_infra_v2.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage62_adapter_training_infra_v2.py --device cuda --steps 4 --eval_interval 2 --frame_gaps 2 4 --max_train_rows_per_gap 1 --max_eval_rows_per_gap 1 --targets_per_row 1
```

Resume validation:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage62_adapter_training_infra_v2.py --device cuda --steps 5 --eval_interval 1 --frame_gaps 2 4 --max_train_rows_per_gap 1 --max_eval_rows_per_gap 1 --targets_per_row 1 --resume_state /data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_latest_training_state.pt
```

## Outputs

Tracked:

```text
experiments/stage62_adapter_training_infra_v2/stage62_adapter_training_infra_v2_summary.json
experiments/stage62_adapter_training_infra_v2/stage62_train_rgb_losses.csv
experiments/stage62_adapter_training_infra_v2/stage62_validation_log.csv
experiments/stage62_adapter_training_infra_v2/stage62_selected_rows.csv
experiments/stage62_adapter_training_infra_v2/stage62_initial_eval.csv
experiments/stage62_adapter_training_infra_v2/stage62_final_eval.csv
experiments/stage62_adapter_training_infra_v2/stage62_best_eval.csv
experiments/stage62_adapter_training_infra_v2/stage62_best_gap_eval_summary.csv
```

External:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_best_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_final_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_latest_training_state.pt
```

## Results

| Metric | Value |
|---|---:|
| available train rows, gaps 2/4 | 3103 |
| available eval rows, gaps 2/4 | 1469 |
| selected train rows | 2 |
| selected eval rows | 2 |
| train tasks | 2 |
| eval tasks | 2 |
| resume start step | 4 |
| final/best step | 5 |
| best eval model PSNR avg | 23.256934739494863 |
| linear PSNR avg | 23.252508732675636 |
| best margin over linear | 0.004426006819226558 |

## Conclusion

Stage62 infra is functional and resume-safe. The result is a smoke check only; Stage63 should run medium adapter training on more DAVIS rows and gaps.
