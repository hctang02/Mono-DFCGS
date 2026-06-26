# Stage63 Medium Adapter Training Pilot

Date: 2026-06-27

## Goal

Run a medium-training pilot on DAVIS all-gap anchors using the Stage62 adapter training infra before committing to a longer 5k+ step run.

## Planned Command

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage62_adapter_training_infra_v2.py --device cuda --summary_root experiments/stage63_medium_adapter_training_pilot --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot --steps 128 --eval_interval 32 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 4 --max_eval_rows_per_gap 2 --targets_per_row 1
```

## Expected Scope

- Train split: DAVIS train.
- Eval split: DAVIS val.
- Gaps: `2, 4, 8, 16`.
- Selected train rows: up to `16`.
- Selected eval rows: up to `8`.
- Steps: `128`.
- This is a pilot, not the final medium/long adapter claim.

## Outputs

Tracked:

```text
experiments/stage63_medium_adapter_training_pilot/
```

External:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot/stage62_best_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot/stage62_final_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot/stage62_latest_training_state.pt
```

## Results

| Metric | Value |
|---|---:|
| available train rows | 3926 |
| available eval rows | 1853 |
| selected train rows | 16 |
| selected eval rows | 8 |
| train tasks | 16 |
| eval tasks | 8 |
| steps | 128 |
| best step | 128 |
| initial eval PSNR | 20.90521679961685 |
| final/best eval PSNR | 20.93635879151969 |
| best margin over linear | 0.031141991902842392 |

Validation curve:

| step | model PSNR | linear PSNR | margin |
|---:|---:|---:|---:|
| 0 | 20.90521679961685 | 20.90521679961685 | 0.0 |
| 32 | 20.913996462661025 | 20.90521679961685 | 0.008779663044176544 |
| 64 | 20.92152115695736 | 20.90521679961685 | 0.01630435734051261 |
| 96 | 20.929940697543877 | 20.90521679961685 | 0.02472389792702856 |
| 128 | 20.93635879151969 | 20.90521679961685 | 0.031141991902842392 |

Gap-wise best margins:

| gap | margin |
|---:|---:|
| 2 | 0.04881351193215622 |
| 4 | 0.03418816116420231 |
| 8 | 0.024005080795515 |
| 16 | 0.017561213719488933 |

## Conclusion

The pilot shows monotonic validation gains across all tested gaps. It justifies a longer Stage63/65 run, but it is not yet the planned 5k-20k step medium-training result.
