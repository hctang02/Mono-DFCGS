# Stage84 Rendered Selector Label Package

Date: 2026-06-28

## Goal

Package rendered selector labels and policy guardrails from Stage68 and Stage69 without rerendering.

## Scope

- Input rendered comparison: `experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv`.
- Input decisions and policies: `experiments/stage69_selector_fallback_calibration/`.
- Output per-point rendered labels and policy guardrail summaries.

## Implementation

Added:

```text
scripts/run_stage84_rendered_selector_label_package.py
```

The package records rendered predicted-vs-uniform adapter PSNR deltas as offline labels. These labels are not available at test time and can only be used for training or analysis.

## Run

GPU check was performed before execution. Stage84 is CPU-only package generation.

## Outputs

```text
experiments/stage84_rendered_selector_label_package/stage84_rendered_selector_label_package_summary.json
experiments/stage84_rendered_selector_label_package/stage84_rendered_selector_label_package_report.md
experiments/stage84_rendered_selector_label_package/stage84_rendered_selector_labels.csv
experiments/stage84_rendered_selector_label_package/stage84_rendered_selector_gap_summary.csv
experiments/stage84_rendered_selector_label_package/stage84_selector_policy_guardrails.csv
experiments/stage84_rendered_selector_label_package/stage84_selector_policy_choices.csv
```

## Result

Rendered labels:

| labels | positive | mean delta | min delta | max delta |
|---:|---:|---:|---:|---:|
| 12 | 7 | 0.030738190041048163 | -0.10978492809701024 | 0.26277712715562274 |

Gap summary:

| gap | count | positives | mean delta | min delta |
|---:|---:|---:|---:|---:|
| 4 | 4 | 3 | 0.025675568904931723 | -0.000555582328463089 |
| 8 | 4 | 1 | -0.01731893196183698 | -0.10978492809701024 |
| 16 | 4 | 3 | 0.08385793318004975 | 0.0 |

Policy guardrails:

| policy | category | mean delta | min delta |
|---|---|---:|---:|
| uniform | safe_deployable_baseline | 0.0 | 0.0 |
| fixed_predicted | deployable_candidate_unstable | 0.030738190041048163 | -0.10978492809701024 |
| oracle_positive_fallback | analysis_oracle_not_deployable | 0.04350771650468873 | 0.0 |
| same_data_threshold_fallback | analysis_oracle_not_deployable | 0.03745316604097404 | 0.0 |
| loocv_threshold_fallback | deployable_style_small_sample | -0.01170162890067535 | -0.10978492809701024 |

## Conclusion

- Fixed-predicted selection has positive mean gain but is unstable.
- The best safe deployable policy under nonnegative min-delta guardrail is still uniform.
- Oracle-positive and same-data threshold policies are analysis upper bounds and not deployable.
- Future selector work needs more rendered labels and decision-aware feed-forward calibration.
