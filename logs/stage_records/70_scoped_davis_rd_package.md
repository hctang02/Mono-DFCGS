# Stage70 Scoped DAVIS RD Package

Date: 2026-06-27

## Goal

Package the current DAVIS eval-subset RD evidence into tables and plots.

This is a scoped package over Stage68 outputs, not the final paper-level RD package. It does not add FCGS/D-FCGS apples-to-apples runs.

## Inputs

```text
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_summary.csv
```

## Expected Outputs

```text
experiments/stage70_scoped_davis_rd_package/stage70_rate_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_all_psnr_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_selector_delta_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_method_summary.csv
experiments/stage70_scoped_davis_rd_package/stage70_baseline_status.csv
experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_curve.png
experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_package_summary.json
```

## Notes

- Default metric is all-frame PSNR.
- Rate is q8 static keyframe-anchor MiB/frame from the existing anchor-count estimate.
- Decoder/model weights are not counted per-video.
- Baseline status should clearly mark FCGS/D-FCGS as not yet locally evaluated apples-to-apples.

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage70 是 report/plot generation，不重渲染。

Command:

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage70_scoped_davis_rd_package.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage70_scoped_davis_rd_package.py
```

## Outputs

Tracked outputs:

```text
experiments/stage70_scoped_davis_rd_package/stage70_rate_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_all_psnr_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_selector_delta_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_method_summary.csv
experiments/stage70_scoped_davis_rd_package/stage70_baseline_status.csv
experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_curve.png
experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_package_summary.json
```

Output size:

```text
408K experiments/stage70_scoped_davis_rd_package
```

## Results

Mean all-frame PSNR by method/selector/gap:

| method | selector | gap | mean rate MiB/frame | mean all PSNR |
|---|---|---:|---:|---:|
| `linear_anchor` | `uniform` | `4` | `0.12188942649147727` | `20.137413494810616` |
| `linear_anchor` | `uniform` | `8` | `0.06551069779829546` | `18.063459460774446` |
| `linear_anchor` | `uniform` | `16` | `0.03811479048295455` | `16.59656669447801` |
| `stage65_rgb_h256_adapter` | `uniform` | `4` | `0.12188942649147727` | `20.608531857721726` |
| `stage65_rgb_h256_adapter` | `uniform` | `8` | `0.06551069779829546` | `18.54248864942665` |
| `stage65_rgb_h256_adapter` | `uniform` | `16` | `0.03811479048295455` | `17.01303254555753` |
| `stage65_rgb_h256_adapter` | `predicted_full_feature_dp` | `4` | `0.12188942649147727` | `20.634207426626656` |
| `stage65_rgb_h256_adapter` | `predicted_full_feature_dp` | `8` | `0.06551069779829546` | `18.525169717464813` |
| `stage65_rgb_h256_adapter` | `predicted_full_feature_dp` | `16` | `0.03811479048295455` | `17.096890478737578` |

Selector aggregate from Stage68:

| item | value |
|---|---:|
| selector positive points | `7 / 12` |
| selector mean adapter all-frame PSNR delta | `+0.030738190041048163 dB` |

Baseline status:

| method | status |
|---|---|
| `FCGS` | not run locally apples-to-apples |
| `D-FCGS` | not run locally apples-to-apples |
| `CWGS` | optional, not run locally |

## Conclusion

- Stage70 packages the current DAVIS eval-subset results into rate tables, all-frame PSNR tables, selector delta table, method summary, and an RD curve.
- This package is useful for internal tracking and presentation of current progress.
- It is not the final benchmark because FCGS/D-FCGS local fair baselines are missing and selector robustness remains unresolved.
