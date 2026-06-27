# Stage69 Selector Fallback Calibration Analysis

Date: 2026-06-27

## Goal

Analyze whether a simple fallback-to-uniform policy can make the Stage68 feed-forward selector more robust.

Stage68 is mixed-positive: average all-frame PSNR is positive, but several points are non-positive. Stage69 reuses Stage68 rendered outcomes as offline calibration labels and does not rerender.

## Policies

| policy | description | deployability status |
|---|---|---|
| `uniform` | always use uniform keyframes | deployable baseline |
| `fixed_predicted` | always use Stage68 predicted DP layout | deployable but unstable |
| `oracle_positive_fallback` | use predicted only when Stage68 rendered delta is positive | oracle upper bound, not deployable |
| `loocv_threshold_fallback` | train simple layout/cost thresholds leave-one-sequence-out | deployable-style analysis, labels from rendered outcomes |

## Inputs

```text
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_selections.csv
```

## Expected Outputs

```text
experiments/stage69_selector_fallback_calibration/stage69_selector_decision_records.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_choices.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_summary.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_fallback_calibration_summary.json
```

## Notes

- This stage is analysis-only and does not run new rendered evaluation.
- `oracle_positive_fallback` is only an upper bound because it uses actual rendered outcomes.
- `loocv_threshold_fallback` is deployable-style but still trained from the small Stage68 rendered outcome set.

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage69 是 CPU analysis，但仍完成 GPU 状态检查。

Command:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage69_selector_fallback_calibration.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage69_selector_fallback_calibration.py
```

## Outputs

Tracked outputs:

```text
experiments/stage69_selector_fallback_calibration/stage69_selector_decision_records.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_choices.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_summary.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_fallback_calibration_summary.json
```

Output size:

```text
28K experiments/stage69_selector_fallback_calibration
```

## Results

Policy summary:

| policy | accepted predicted | positive points | mean all PSNR delta | min all PSNR delta | status |
|---|---:|---:|---:|---:|---|
| `uniform` | `0 / 12` | `0 / 12` | `0.0` | `0.0` | deployable baseline |
| `fixed_predicted` | `12 / 12` | `7 / 12` | `+0.030738190041048163` | `-0.10978492809701024` | deployable but unstable |
| `oracle_positive_fallback` | `7 / 12` | `7 / 12` | `+0.04350771650468873` | `0.0` | oracle upper bound, not deployable |
| `same_data_threshold_fallback` | `5 / 12` | `5 / 12` | `+0.03745316604097404` | `0.0` | same-data upper bound, not deployable |
| `loocv_threshold_fallback` | `4 / 12` | `1 / 12` | `-0.01170162890067535` | `-0.10978492809701024` | deployable-style small-sample calibration |

## Conclusion

- Oracle and same-data fallback show that avoiding negative predicted layouts could improve Stage68, but those policies are not deployable claims.
- Leave-one-sequence-out threshold fallback is worse than fixed predicted and worse than uniform on this small set.
- The best deployable-style policy remains fixed predicted, but it is unstable because it keeps the `goat gap8` negative point.
- Next work should build rendered-distortion labels on more sequences or train a decision-aware fallback classifier with more data, rather than relying on simple layout/cost thresholds.
