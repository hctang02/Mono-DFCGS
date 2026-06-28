# Stage81 Adapter Training Pilot

Date: 2026-06-28

## Goal

Run a controlled adapter pilot fine-tuning pass using the Stage79 task manifest and the Stage65 `rgb_h256` adapter as initialization.

## Scope

- Input manifest: `experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`.
- Codecs: `q10`, `q12`.
- Reference gaps: `4`, `8`, `16`.
- Train tasks: `48`.
- Eval tasks: `18`.
- Steps: `48`.
- Eval interval: `12`.

## Implementation

Updated:

```text
scripts/run_stage80_adapter_training_smoke.py
```

Added support for:

- `--init_checkpoint` to initialize the training adapter from an existing checkpoint.
- `--stage`, `--stage_label`, `--output_prefix`, and `--summary_name` for Stage81 outputs.
- `--run_note` for smoke/pilot/long-run report wording.

## Run

GPU check was performed before execution. GPU4 was idle, so the pilot used:

```text
CUDA_VISIBLE_DEVICES=4
```

Initialization checkpoint:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors
```

## Outputs

Repository outputs:

```text
experiments/stage81_adapter_training_pilot/stage81_adapter_training_pilot_summary.json
experiments/stage81_adapter_training_pilot/stage81_adapter_training_pilot_report.md
experiments/stage81_adapter_training_pilot/stage81_train_log.csv
experiments/stage81_adapter_training_pilot/stage81_validation_log.csv
experiments/stage81_adapter_training_pilot/stage81_best_eval_rows.csv
experiments/stage81_adapter_training_pilot/stage81_final_eval_rows.csv
experiments/stage81_adapter_training_pilot/stage81_reference_eval_rows.csv
```

External checkpoints:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage81_adapter_training_pilot/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage81_adapter_training_pilot/final_adapter.safetensors
```

## Result

| checkpoint | step | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|---:|
| initial/reference Stage65 | 0 | 18.43683802603116 | 18.37495762954318 | 0.061880396487980015 |
| best pilot | 24 | 18.481773239223003 | 18.37495762954318 | 0.10681560967982436 |
| final pilot | 48 | 18.48140534613068 | 18.37495762954318 | 0.10644771658749658 |

Best margin by gap:

| gap | margin |
|---:|---:|
| 4 | -0.18006936459308479 |
| 8 | 0.14113360277949102 |
| 16 | 0.403032388467782 |

## Conclusion

- The Stage65 adapter can be used as a stable initialization for q10/q12 Stage79 task fine-tuning.
- The pilot improved mean margin on this small held-out slice from `+0.061880396487980015 dB` to `+0.10681560967982436 dB`.
- The gap4 margin remains negative, so the next step should use gap-aware sampling/loss or a broader validation slice before claiming a robust adapter improvement.
