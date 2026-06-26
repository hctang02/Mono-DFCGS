# Stage54 Decision-Aware Selector Analysis

## Outputs
- Decision records CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage54_decision_aware_selector_analysis/stage54_decision_records.csv`
- Policy choices CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage54_decision_aware_selector_analysis/stage54_policy_choices.csv`
- Policy summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage54_decision_aware_selector_analysis/stage54_policy_summary.csv`
- Summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage54_decision_aware_selector_analysis/stage54_decision_aware_selector_analysis_summary.json`

## Policy Summary
| Policy | Mean all delta | Positive all | Min all delta | Accepted adaptive | Notes |
|---|---:|---:|---:|---:|---|
| uniform | 0.000000 | 0/12 | 0.000000 | 0 | baseline fallback |
| oracle_best_candidate_pool | 0.006302 | 3/12 | 0.000000 | 3 | upper bound over Stage48 candidate layouts plus uniform |
| oracle_layout_imitation | 0.000000 | 0/12 | 0.000000 | 0 | upper bound using Stage49 oracle layout similarity |
| loocv_layout_threshold | -0.004029 | 1/12 | -0.059917 | 2 | leave-one-sample-out layout-threshold fallback |
| fixed_length_raw_log_prior_0p1 | -0.040929 | 0/12 | -0.385768 | 12 | fixed Stage48 predicted method |
| fixed_full_raw_log_prior_0p1 | -0.210704 | 0/12 | -0.415297 | 12 | fixed Stage48 predicted method |
| fixed_length_sample_z_rank_prior_0p1 | -0.038395 | 0/12 | -0.385768 | 12 | fixed Stage48 predicted method |
| fixed_full_sample_z_rank | -1.051881 | 0/12 | -1.777892 | 12 | fixed Stage48 predicted method |
| fixed_full_sample_z_rank_prior_0p1 | -0.346864 | 1/12 | -1.775271 | 12 | fixed Stage48 predicted method |
| fixed_full_sample_z_rank_prior_0p3 | -0.213574 | 2/12 | -1.690961 | 12 | fixed Stage48 predicted method |

## Interpretation
- `oracle_best_candidate_pool` is an upper bound that uses actual RD outcomes to choose among Stage48 predicted layouts and uniform.
- `oracle_layout_imitation` uses the Stage49 rendered-oracle layout only for analysis; it is not deployable.
- `loocv_layout_threshold` is a deployable-style fallback rule trained leave-one-sample-out from layout features only, but it reuses Stage48 rendered outcomes as labels for this analysis.
- If the oracle candidate pool is much better than fixed methods, the next step is a decision-aware selector objective/fallback classifier rather than only segment-cost correlation.
