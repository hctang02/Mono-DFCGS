# Stage91 Fixed Residual Side-Info Bitstream Smoke

Date: 2026-06-28

## Goal

Validate q6 top10 residual side-info with an actual fixed-length byte payload instead of only theoretical bit accounting.

## Implementation

Added:

```text
mono_dfcgs/residual_sideinfo_codec.py
scripts/run_stage91_residual_sideinfo_bitstream_smoke.py
```

The codec packs:

- fixed header
- float16 per-attribute min/max metadata
- bit-packed Gaussian indices
- bit-packed q6 residual values

## Run

GPU check was performed before execution. GPU0/1 were idle; the run used GPU1 to avoid default GPU interference:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage91_residual_sideinfo_bitstream_smoke.py
```

## Outputs

```text
experiments/stage91_residual_sideinfo_bitstream_smoke/stage91_residual_sideinfo_bitstream_rows.csv
experiments/stage91_residual_sideinfo_bitstream_smoke/stage91_residual_sideinfo_bitstream_summary.csv
experiments/stage91_residual_sideinfo_bitstream_smoke/stage91_residual_sideinfo_bitstream_summary.json
experiments/stage91_residual_sideinfo_bitstream_smoke/stage91_residual_sideinfo_bitstream_report.md
```

## Payload Accounting

| component | bytes |
|---|---:|
| header | 18 |
| float16 min/max metadata | 52 |
| bit-packed Gaussian indices | 7372 |
| bit-packed q6 residual values | 35939 |
| total payload | 43381 |

Theoretical Stage87-90 rate without header/byte padding: `43362.5 bytes` / `0.041353702545166016 MiB/intermediate-frame`.

Actual fixed payload: `43381 bytes` / `0.04137134552001953 MiB/intermediate-frame`.

Overhead: `18.5 bytes`.

## Results

| base | gap | tasks | bitstream PSNR | delta PSNR | positives |
|---|---:|---:|---:|---:|---:|
| linear | 4 | 3 | 23.327021755289916 | 3.3372252270957077 | 3 |
| linear | 8 | 5 | 20.94541228168298 | 2.8242945281834873 | 5 |
| linear | 16 | 4 | 23.589545045225233 | 4.476530074672733 | 4 |
| stage65_adapter | 4 | 3 | 23.352831018389125 | 2.870092812012075 | 3 |
| stage65_adapter | 8 | 5 | 20.943699759081163 | 2.41134126509652 | 5 |
| stage65_adapter | 16 | 4 | 22.48232041707169 | 3.407310328486104 | 4 |

## Conclusion

- Fixed bitstream payload accounting closely matches the theoretical q6 top10 side-info rate.
- Rendered PSNR remains positive after decode for every evaluated task.
- This is still teacher residual side-info, not entropy coding or a deployable learned residual predictor.
