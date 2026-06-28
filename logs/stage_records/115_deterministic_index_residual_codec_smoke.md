# Stage115 Deterministic-Index Residual Codec Smoke

Date: 2026-06-29

## Goal

Validate a deterministic-index residual side-info codec where the decoder reproduces endpoint-diff selected indices from left/right anchors and the bitstream carries only residual values and min/max metadata.

## Implementation

Updated:

```text
mono_dfcgs/residual_sideinfo_codec.py
```

Added helpers:

```text
encode_selected_residual_sideinfo
encode_selected_residual_values_sideinfo
decode_selected_residual_values_sideinfo
deterministic_sideinfo_bits_without_header
deterministic_sideinfo_mib_without_header
```

Added:

```text
scripts/run_stage115_deterministic_index_residual_codec_smoke.py
```

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/residual_sideinfo_codec.py scripts/run_stage115_deterministic_index_residual_codec_smoke.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage115_deterministic_index_residual_codec_smoke.py
```

## Outputs

```text
experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_rows.csv
experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_summary.csv
experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_summary.json
experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_report.md
```

## Configuration

| item | value |
|---|---:|
| selector policy | `strict_safe_endpoint_selector_v1` |
| selected candidate | `endpoint_diff_baseline` |
| task count | 12 |
| keep fraction | 0.1 |
| side bits | 6 |

## Results

| base | gap | tasks | fixed bytes | deterministic bytes | saved bytes | ratio | deterministic MiB/intermediate | max decode diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 3 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |
| linear | 8 | 5 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388002 | 0.034340858459472656 | 0.0 |
| linear | 16 | 4 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |
| stage65_adapter | 4 | 3 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |
| stage65_adapter | 8 | 5 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388002 | 0.034340858459472656 | 0.0 |
| stage65_adapter | 16 | 4 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |

## Conclusion

- Deterministic value-only side-info removes selected-index bytes when the decoder reproduces endpoint-diff indices.
- The q6 top10 smoke payload drops from `43381` to `36009` bytes per intermediate frame.
- Decode is equivalent to fixed index+value payload under the same selected indices, with max diff `0.0`.
- Residual values remain teacher-derived; this is not residual value prediction.
- Stage116 should compare deterministic value-only side-info against entropy-coded index+value side-info in RD accounting.
