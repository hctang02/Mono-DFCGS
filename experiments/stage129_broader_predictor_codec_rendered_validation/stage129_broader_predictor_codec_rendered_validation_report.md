# Stage129 Broader Predictor Codec Rendered Validation

## Scope

- Render-validates Stage128 predictor-only selected residual codec on 60 eval tasks.
- Uses no residual payload and no selected-index payload.
- Target dense anchors are not loaded; target RGB is used only for offline rendered metrics.

## Summary

| setting | role | keep | tasks | rate | base PSNR | full adapter PSNR | predictor PSNR | delta base | delta full | positives | near full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.865778 | -0.085025 | -0.321866 | 32/60 | 13/60 |
| q4_top20 | primary | 0.2 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.765203 | -0.185600 | -0.422440 | 26/60 | 12/60 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_report.md`
