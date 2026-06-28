# Stage98 Residual Importance Predictor Smoke

Date: 2026-06-28

## Goal

Train a small feed-forward importance predictor to localize teacher top10 residual-energy Gaussians using only decoder-available anchor features.

## Implementation

Added:

```text
scripts/run_stage98_residual_importance_predictor_smoke.py
```

The model input contains base anchor attrs, left/right keyframe attrs, endpoint diff, absolute endpoint diff, normalized time, and base method id. Labels are generated offline from target dense anchors.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle, so the run used:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage98_residual_importance_predictor_smoke.py
```

## Outputs

```text
experiments/stage98_residual_importance_predictor_smoke/stage98_residual_importance_predictor_rows.csv
experiments/stage98_residual_importance_predictor_smoke/stage98_residual_importance_predictor_summary.csv
experiments/stage98_residual_importance_predictor_smoke/stage98_residual_importance_predictor_train_log.csv
experiments/stage98_residual_importance_predictor_smoke/stage98_residual_importance_predictor_summary.json
experiments/stage98_residual_importance_predictor_smoke/stage98_residual_importance_predictor_report.md
```

## Results

| candidate | base | gap | precision@keep | energy recall | relative recall vs oracle |
|---|---|---:|---:|---:|---:|
| endpoint_diff_baseline | linear | 4 | 0.3122626145680745 | 0.2655188664793968 | 0.42121463517347973 |
| endpoint_diff_baseline | linear | 8 | 0.24724181989828745 | 0.23293344179789224 | 0.34736502170562744 |
| endpoint_diff_baseline | linear | 16 | 0.2551998645067215 | 0.29756257434686023 | 0.38490988314151764 |
| mlp_importance | linear | 4 | 0.310047022998333 | 0.2947804356614749 | 0.46986613670984906 |
| mlp_importance | linear | 8 | 0.292729248603185 | 0.2800879975159963 | 0.4127636154492696 |
| mlp_importance | linear | 16 | 0.26433352132638294 | 0.3037059058745702 | 0.4129539728164673 |
| endpoint_diff_baseline | stage65_adapter | 4 | 0.27450714260339737 | 0.12835493932167688 | 0.6087801853815714 |
| endpoint_diff_baseline | stage65_adapter | 8 | 0.2631578892469406 | 0.13744011024634042 | 0.5076933900515238 |
| endpoint_diff_baseline | stage65_adapter | 16 | 0.18972689906756082 | 0.11895131816466649 | 0.5244861443837484 |
| mlp_importance | stage65_adapter | 4 | 0.3889039605855942 | 0.14692467202742895 | 0.6976491312185923 |
| mlp_importance | stage65_adapter | 8 | 0.37384698788324994 | 0.1679172863562902 | 0.6167584160963694 |
| mlp_importance | stage65_adapter | 16 | 0.28431904315948486 | 0.14672287801901499 | 0.6156713565190634 |

## Conclusion

- The MLP importance predictor improves energy recall over endpoint-difference baseline for every base/gap group.
- It remains far below oracle top10 residual-energy selection, so it is not yet a deployable replacement for teacher residual side-info.
- Stage99 should test predictor-selected indices with teacher residual values to isolate selection error from residual-value prediction error.
