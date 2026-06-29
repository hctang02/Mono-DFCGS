# Stage135 Render-Aware Predictor Protocol

## Protocol

- Predictor family: adapter-delta selected residual predictor.
- Residual rule: `scaled_residual = adapter_delta_scale * (adapter_attrs - linear_attrs)` at deterministic endpoint-diff indices.
- Scale candidates: `0.0, 0.25, 0.5, 0.75, 1.0, 1.25`.
- Settings: q4/top20 primary and q4/top10 low-rate.
- Target RGB is used only for offline protocol selection and validation.
- Target dense anchors and teacher residuals are not used.

## Acceptance Criteria

- Choose the scale/setting with the highest rendered PSNR on Stage136 smoke.
- Keep only candidates with non-negative mean delta vs linear base on Stage137 broader validation.
- If no scale beats Stage132, retain Stage132 adapter-delta q4/top20 policy.

## Outputs

- protocol CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol.csv`
- protocol JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol_report.md`
