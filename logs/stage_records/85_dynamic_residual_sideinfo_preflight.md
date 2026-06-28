# Stage85 Dynamic Residual Side-Info Preflight

Date: 2026-06-28

## Goal

Estimate anchor-space potential and rate cost for dynamic residual side-info.

## Scope

- Tasks: `60` Stage79 eval tasks.
- Codecs: `q10`, `q12` endpoint anchors.
- Gaps: `4`, `8`, `16`.
- Methods: linear anchor interpolation and Stage65 adapter.
- Side bits: `6`, `8`.
- Keep fractions: `0`, `0.01`, `0.05`, `0.1`, `0.25`, `1.0`.

## Implementation

Added:

```text
scripts/run_stage85_dynamic_residual_sideinfo_preflight.py
```

The script uses Stage61 dense gap1 anchors as offline teacher targets, computes anchor-space residual energy, and estimates top-k residual side-info rate with Gaussian index bits plus quantized residual attribute bits.

## Run

GPU check was performed before execution. GPU1 was idle, so the run used:

```text
CUDA_VISIBLE_DEVICES=1
```

## Outputs

```text
experiments/stage85_dynamic_residual_sideinfo_preflight/stage85_dynamic_residual_sideinfo_preflight_summary.json
experiments/stage85_dynamic_residual_sideinfo_preflight/stage85_dynamic_residual_sideinfo_preflight_report.md
experiments/stage85_dynamic_residual_sideinfo_preflight/stage85_dynamic_residual_sideinfo_rows.csv
experiments/stage85_dynamic_residual_sideinfo_preflight/stage85_dynamic_residual_sideinfo_summary.csv
```

## Result

q12 linear residual side-info:

| gap | keep | 8-bit MiB/intermediate-frame | captured energy |
|---:|---:|---:|---:|
| 4 | 0.10 | 0.05272865295410156 | 0.6736598664316831 |
| 4 | 0.25 | 0.1318359375 | 0.8786127617934746 |
| 8 | 0.10 | 0.05272865295410156 | 0.6167360978722798 |
| 8 | 0.25 | 0.1318359375 | 0.8502699399525562 |
| 16 | 0.10 | 0.05272865295410156 | 0.6257691648725925 |
| 16 | 0.25 | 0.1318359375 | 0.8485600745450567 |

q12 Stage65 adapter residual side-info:

| gap | keep | 8-bit MiB/intermediate-frame | captured energy |
|---:|---:|---:|---:|
| 4 | 0.10 | 0.05272865295410156 | 0.25083170952874323 |
| 4 | 0.25 | 0.1318359375 | 0.4357608051236221 |
| 8 | 0.10 | 0.05272865295410156 | 0.26118229986966074 |
| 8 | 0.25 | 0.1318359375 | 0.45968361079173065 |
| 16 | 0.10 | 0.05272865295410156 | 0.24572836645426546 |
| 16 | 0.25 | 0.1318359375 | 0.4386825171223625 |

## Conclusion

- Linear residual energy is concentrated enough that top-k residual side-info could be promising.
- Stage65 adapter residual is more diffuse in anchor space, so top-k residual side-info captures less energy.
- This is an optimistic anchor-space preflight, not rendered RD.
- Any transmitted dynamic residual side-info must be added to side-info rate and total rate.
