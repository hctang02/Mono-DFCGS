# Stage123 Compressed Deterministic Codec Policy Package

Date: 2026-06-29

## Goal

Package a reusable codec policy manifest for compressed deterministic value-only residual side-info.

## Implementation

Added:

```text
scripts/run_stage123_compressed_deterministic_codec_policy_package.py
```

The script consumes the Stage114 strict-safe selector policy and the Stage122 RD package. It emits a policy JSON, settings CSV, package JSON, and Markdown report.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`. After fixing source path handling in the script, GPU was checked again and the package was regenerated.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage123_compressed_deterministic_codec_policy_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage123_compressed_deterministic_codec_policy_package.py
```

## Outputs

```text
experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy.json
experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy_settings.csv
experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy_package.json
experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy_report.md
```

## Configuration

| item | value |
|---|---:|
| policy | compressed_deterministic_value_only_residual_codec_v1 |
| status | package_not_full_residual_predictor |
| setting count | 4 |
| selector | strict_safe_endpoint_selector_v1 |
| selected candidate | endpoint_diff_baseline |
| index rule | endpoint_diff_topk_v1 |
| payload magic | RSDZ |
| header bytes | 26 |
| zlib level | 9 |

## Decoder Contract

- `keep_count = min(max(round(N * keep_fraction), 0), N)`.
- Score rule: `sum((right_attrs[0].float() - left_attrs[0].float()) ** 2, dim=-1)`.
- Select top `keep_count` largest scores and sort selected indices ascending.
- Selected indices are not transmitted.
- Decoder forbidden inputs: target dense anchor, target residual, target RGB, oracle task label, transmitted selected indices.

## Settings

| role | setting | keep | bits | payload bytes | direct rate | amortized rate | PSNR | delta q6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 4 | 28320.791666666668 | 0.1337680887378662 | 0.13005386354500142 | 20.689270746602087 | 0.9223020959187475 |
| low-rate | q4_top10 | 0.1 | 4 | 15190.475 | 0.12124604296658527 | 0.11924717736218463 | 19.73848817438193 | -0.028480476301425663 |
| near-anchor | q5_top10 | 0.1 | 5 | 24809.95 | 0.13041988921139727 | 0.127149533351005 | 19.761047533309117 | -0.005921117374238853 |
| anchor | q6_top10 | 0.1 | 6 | 29442.208333333332 | 0.1348375550108561 | 0.13095624756787308 | 19.766968650683353 | 0.0 |

## Conclusion

- Stage123 freezes the compressed deterministic codec policy manifest.
- q4/top20 remains the primary policy setting and q4/top10 is the low-rate setting.
- Decoder-side selected indices are reproducible from left/right anchors and are not transmitted.
- Residual values remain teacher-derived at encoder side, so this is not a full residual value predictor.
- Stage124 should start residual value predictor packaging or smoke validation.
