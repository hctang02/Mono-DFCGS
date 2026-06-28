# Stage82 Adapter Pilot Broader Eval

Date: 2026-06-28

## Goal

Evaluate whether the Stage81 pilot adapter improvement holds on a broader held-out validation slice.

## Scope

- Eval-only run with `steps=0`.
- Init checkpoint: Stage81 pilot best adapter.
- Reference checkpoint: Stage65 `rgb_h256` best adapter.
- Codecs: `q10`, `q12`.
- Reference gaps: `4`, `8`, `16`.
- Eval tasks: `60`.

## Run

GPU check was performed before execution. GPU4 was idle, so the eval used:

```text
CUDA_VISIBLE_DEVICES=4
```

Stage81 best checkpoint:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage81_adapter_training_pilot/best_adapter.safetensors
```

Stage65 reference checkpoint:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors
```

## Outputs

Repository outputs:

```text
experiments/stage82_adapter_pilot_broader_eval/stage82_adapter_pilot_broader_eval_summary.json
experiments/stage82_adapter_pilot_broader_eval/stage82_adapter_pilot_broader_eval_report.md
experiments/stage82_adapter_pilot_broader_eval/stage82_train_log.csv
experiments/stage82_adapter_pilot_broader_eval/stage82_validation_log.csv
experiments/stage82_adapter_pilot_broader_eval/stage82_best_eval_rows.csv
experiments/stage82_adapter_pilot_broader_eval/stage82_final_eval_rows.csv
experiments/stage82_adapter_pilot_broader_eval/stage82_reference_eval_rows.csv
```

## Result

| model | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|
| Stage81 best | 19.360426943660283 | 19.259310564194283 | 0.10111637946599329 |
| Stage65 reference | 19.369958827711727 | 19.259310564194283 | 0.11064826351743612 |

Margin by gap:

| model | gap4 | gap8 | gap16 |
|---|---:|---:|---:|
| Stage81 best | -0.0773488024139178 | 0.06382222406404825 | 0.3698363934848712 |
| Stage65 reference | -0.010726898647836793 | 0.05364375708512303 | 0.3343528251045902 |

## Conclusion

- Stage81 pilot does not beat Stage65 reference on the broader 60-task eval slice.
- Stage81 improves gap8/gap16 margins but worsens gap4 more than Stage65.
- Keep Stage65 as the current best adapter for now.
- Next adapter training should be gap-aware and should explicitly protect gap4 quality.
