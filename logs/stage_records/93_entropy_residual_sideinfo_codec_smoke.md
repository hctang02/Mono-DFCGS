# Stage93 Entropy-Coded Residual Side-Info Codec Smoke

Date: 2026-06-28

## Goal

Turn Stage92 `delta_index_zlib` preflight into a real decode-capable entropy-coded residual side-info codec.

## Implementation

Updated:

```text
mono_dfcgs/residual_sideinfo_codec.py
```

Added functions:

```text
encode_topk_residual_sideinfo_entropy
decode_residual_sideinfo_entropy
```

Added script:

```text
scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py
```

Codec v2 payload:

- header
- zlib-compressed float16 min/max metadata
- zlib-compressed sorted-index deltas
- zlib-compressed q6 residual values

## Run

GPU check was performed before execution. GPU1 was idle, so the run used:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py
```

## Outputs

```text
experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_rows.csv
experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_summary.csv
experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_summary.json
experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_report.md
```

## Results

| base | gap | entropy bytes | ratio vs fixed | entropy MiB/intermediate | max decode diff | delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| linear | 4 | 29763.666666666668 | 0.6860991371030328 | 0.028384844462076824 | 0.0 | 3.3372252270957077 |
| linear | 8 | 30815.4 | 0.71034323782301 | 0.029387855529785158 | 0.0 | 2.8242945281834873 |
| linear | 16 | 29559.5 | 0.6813927756391047 | 0.028190135955810547 | 0.0 | 4.476530074672733 |
| stage65_adapter | 4 | 35026.0 | 0.8074041631128835 | 0.03340339660644531 | 0.0 | 2.870092812012075 |
| stage65_adapter | 8 | 36109.8 | 0.8323874507272769 | 0.03443698883056641 | 0.0 | 2.41134126509652 |
| stage65_adapter | 16 | 34910.25 | 0.8047359443074157 | 0.03329300880432129 | 0.0 | 3.407310328486104 |

## Conclusion

- Entropy codec v2 decodes exactly to the fixed-codec reconstruction in this smoke.
- It preserves rendered gains while reducing payload bytes.
- Residuals are still teacher-derived; this is a codec stage, not a deployable residual predictor.
