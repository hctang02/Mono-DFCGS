# Stage67 DAVIS Selector Predictor Training

Date: 2026-06-27

## Goal

Train and validate feed-forward segment-cost predictors on the Stage66 DAVIS selector dataset.

This stage is still predictor validation, not final selector RD. The target label is Stage66's offline anchor-space adapter error proxy:

```text
label_log_adapter_mse_mean
```

## Inputs

| item | path |
|---|---|
| Stage66 dataset | `experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset.csv` |
| Output summary root | `experiments/stage67_davis_selector_predictor_training` |

## Predictor Variants

| model | features |
|---|---|
| `length_only_ridge` | `segment_length`, `middle_count` |
| `rgb_motion_ridge` | length features + RGB motion features |
| `anchor_endpoint_ridge` | length features + endpoint anchor statistics |
| `full_feature_ridge` | all Stage66 encoder-side features |

## Expected Outputs

```text
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_metrics.csv
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_predictions.csv
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_model_params.json
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_training_summary.json
```

## Notes

- Features are feed-forward/deployable; labels are offline supervision only.
- A good Stage67 proxy predictor is not sufficient for final selector claims because Stage66 labels are anchor-space, not rendered all-frame PSNR.
- Stage68 should add deterministic DP selection and rendered/full-video validation before claiming selector gains.

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage67 是 CPU ridge training，但仍完成 GPU 状态检查。

Command:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage67_davis_selector_predictor_training.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage67_davis_selector_predictor_training.py
```

## Results

| model | features | eval RMSE log | eval Pearson | eval Spearman |
|---|---:|---:|---:|---:|
| `length_only_ridge` | `2` | `0.03748165925098028` | `0.3094398002609641` | `0.3391438570632329` |
| `rgb_motion_ridge` | `7` | `0.032271284148152724` | `0.6528770726740828` | `0.6576988074032709` |
| `anchor_endpoint_ridge` | `15` | `0.020229501422918364` | `0.8664301098563049` | `0.8671254647839715` |
| `full_feature_ridge` | `18` | `0.01746532908957139` | `0.9103771172408511` | `0.9017146144293104` |

Best model:

```text
full_feature_ridge
```

Dataset size:

| split | rows |
|---|---:|
| train | `3072` |
| eval | `1536` |

Tracked outputs:

```text
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_metrics.csv
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_predictions.csv
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_model_params.json
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_training_summary.json
```

Output size:

```text
3.2M experiments/stage67_davis_selector_predictor_training
```

## Conclusion

- Feed-forward encoder-side features can predict the Stage66 anchor-space proxy label well on the scoped DAVIS train/val split.
- Endpoint-anchor features are much stronger than length-only or RGB-motion-only features.
- Full features are best by eval Spearman and RMSE.
- This is not yet a selector RD result. Stage68 needs deterministic DP selection from predicted segment costs and rendered/full-video validation, preferably with a rendered-distortion label subset.
