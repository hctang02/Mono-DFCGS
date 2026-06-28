# Stage118 Compressed Deterministic Codec Smoke

Date: 2026-06-29

## Goal

Validate a compressed deterministic-index value-only residual side-info codec. Decoder-side selected indices are reproduced deterministically, while the bitstream stores zlib-compressed metadata and q residual values only.

## Implementation

Updated:

```text
mono_dfcgs/residual_sideinfo_codec.py
```

Added helpers:

```text
encode_selected_residual_values_sideinfo_entropy
decode_selected_residual_values_sideinfo_entropy
```

Added:

```text
scripts/run_stage118_compressed_deterministic_codec_smoke.py
```

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/residual_sideinfo_codec.py scripts/run_stage118_compressed_deterministic_codec_smoke.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage118_compressed_deterministic_codec_smoke.py
```

## Outputs

```text
experiments/stage118_compressed_deterministic_codec_smoke/stage118_compressed_deterministic_codec_rows.csv
experiments/stage118_compressed_deterministic_codec_smoke/stage118_compressed_deterministic_codec_summary.csv
experiments/stage118_compressed_deterministic_codec_smoke/stage118_compressed_deterministic_codec_summary.json
experiments/stage118_compressed_deterministic_codec_smoke/stage118_compressed_deterministic_codec_report.md
```

## Configuration

| item | value |
|---|---:|
| selector policy | `strict_safe_endpoint_selector_v1` |
| selected candidate | `endpoint_diff_baseline` |
| task count | 12 |
| keep fraction | 0.1 |
| side bits | 6 |
| zlib level | 9 |

## Results

| base | gap | tasks | raw det bytes | compressed det bytes | comp/raw | Stage96 entropy bytes | comp - entropy | max raw diff | max fixed diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 3 | 36009.0 | 25652.333333333332 | 0.712386718135281 | 29869.833333333332 | -4217.499999999999 | 0.0 | 0.0 |
| linear | 8 | 5 | 36009.0 | 27322.2 | 0.7587603099225195 | 30440.894736842107 | -3118.694736842107 | 0.0 | 0.0 |
| linear | 16 | 4 | 36009.0 | 26884.0 | 0.746591129995279 | 30239.565217391304 | -3355.565217391304 | 0.0 | 0.0 |
| stage65_adapter | 4 | 3 | 36009.0 | 30934.666666666668 | 0.8590815259148177 | 34757.88888888889 | -3823.222222222224 | 0.0 | 0.0 |
| stage65_adapter | 8 | 5 | 36009.0 | 32457.6 | 0.9013746563359162 | 35513.15789473684 | -3055.55789473684 | 0.0 | 0.0 |
| stage65_adapter | 16 | 4 | 36009.0 | 32162.5 | 0.893179482907051 | 35406.434782608696 | -3243.934782608696 | 0.0 | 0.0 |

## Conclusion

- Compressed deterministic value-only side-info removes selected-index bytes and compresses metadata/residual values.
- Decode is identical to raw deterministic and fixed index+value payloads under the same selected indices.
- q6/top10 compressed deterministic payload is below Stage96 q6/top10 entropy-coded index+value reference in all groups.
- Residual values remain teacher-derived; this is not residual value prediction.
- Stage119 should run an actual compressed q-bit / keep-fraction sweep.
