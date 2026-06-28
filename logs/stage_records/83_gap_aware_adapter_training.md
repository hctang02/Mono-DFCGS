# Stage83 Gap-Aware Adapter Training

Date: 2026-06-28

## Goal

Test whether gap-aware RGB-loss fine-tuning can improve adapter performance while protecting gap4 quality.

## Scope

- Initialize from Stage65 `rgb_h256` best adapter.
- Codecs: `q10`, `q12`.
- Reference gaps: `4`, `8`, `16`.
- Train tasks: `72`.
- Eval tasks: `60`.
- Steps: `72`.
- Gap loss weights: gap4 `3.0`, gap8 `1.0`, gap16 `1.0`.
- Best metric: `protected_gap4_margin` with gap4 penalty `2.0`.

## Implementation

Updated:

```text
scripts/run_stage80_adapter_training_smoke.py
```

Added support for:

- `--gap_loss_weights` for reference-gap-specific RGB loss weighting.
- `--best_metric protected_gap4_margin` for gap4-protected checkpoint selection.
- `--gap4_penalty` for tuning the negative gap4 penalty.

## Run

GPU check was performed before execution. GPU1 was idle, so the pilot used:

```text
CUDA_VISIBLE_DEVICES=1
```

## Outputs

Repository outputs:

```text
experiments/stage83_gap_aware_adapter_training/stage83_gap_aware_adapter_training_summary.json
experiments/stage83_gap_aware_adapter_training/stage83_gap_aware_adapter_training_report.md
experiments/stage83_gap_aware_adapter_training/stage83_train_log.csv
experiments/stage83_gap_aware_adapter_training/stage83_validation_log.csv
experiments/stage83_gap_aware_adapter_training/stage83_best_eval_rows.csv
experiments/stage83_gap_aware_adapter_training/stage83_final_eval_rows.csv
experiments/stage83_gap_aware_adapter_training/stage83_reference_eval_rows.csv
```

External checkpoints:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage83_gap_aware_adapter_training/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage83_gap_aware_adapter_training/final_adapter.safetensors
```

## Result

| checkpoint | step | mean margin | gap4 margin | gap8 margin | gap16 margin |
|---|---:|---:|---:|---:|---:|
| initial/reference Stage65 | 0 | 0.11064826351743612 | -0.010726898647836793 | 0.05364375708512303 | 0.3343528251045902 |
| protected best | 0 | 0.11064826351743612 | -0.010726898647836793 | 0.05364375708512303 | 0.3343528251045902 |
| final | 72 | 0.11286058458557878 | -0.044278793981359484 | 0.06219831435034278 | 0.3725368725315138 |

## Conclusion

- The protected best metric selected the initial Stage65 checkpoint, not the fine-tuned final checkpoint.
- The final checkpoint slightly improved mean margin and long-gap margins but degraded gap4 more.
- Stage65 remains the current best adapter.
- Further simple RGB-loss fine-tuning is unlikely to solve the bottleneck alone; next stages should focus on rendered-label selector or dynamic side-info.
