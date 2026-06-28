# Stage84 Rendered Selector Label Package

## Rendered Labels

- label count: `12`
- positive predicted selections: `7`
- mean rendered delta: `0.030738190041048163`
- minimum rendered delta: `-0.10978492809701024`

| gap | count | positives | mean delta | min delta | max delta | mean rate |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 3 | 0.025675568904931723 | -0.000555582328463089 | 0.06357661189149155 | 0.12188942649147727 |
| 8 | 4 | 1 | -0.01731893196183698 | -0.10978492809701024 | 0.08340300738787576 | 0.06551069779829546 |
| 16 | 4 | 3 | 0.08385793318004975 | 0.0 | 0.26277712715562274 | 0.03811479048295455 |

## Policy Guardrails

| policy | category | accepted | mean delta | min delta | notes |
|---|---|---:|---:|---:|---|
| uniform | safe_deployable_baseline | 0 | 0.0 | 0.0 | deployable baseline |
| fixed_predicted | deployable_candidate_unstable | 12 | 0.030738190041048163 | -0.10978492809701024 | deployable but unstable Stage68 selector |
| oracle_positive_fallback | analysis_oracle_not_deployable | 7 | 0.04350771650468873 | 0.0 | oracle upper bound; not deployable |
| same_data_threshold_fallback | analysis_oracle_not_deployable | 5 | 0.03745316604097404 | 0.0 | analysis upper bound; not deployable |
| loocv_threshold_fallback | deployable_style_small_sample | 4 | -0.01170162890067535 | -0.10978492809701024 | deployable-style small-sample calibration |

## Conclusion

- best deployable policy: `fixed_predicted`
- best deployable mean delta: `0.030738190041048163`
- best safe deployable policy: `uniform`
- best safe deployable min delta: `0.0`
- Rendered labels are offline supervision only; a final selector must use feed-forward features plus deterministic DP at test time.
- Current rendered label set is small, so oracle-positive and same-data threshold rows are analysis upper bounds, not deployable claims.
