# Stage118 Compressed Deterministic Codec Smoke

## Configuration

- task count: `12`
- selector policy: `strict_safe_endpoint_selector_v1`
- selected candidate: `endpoint_diff_baseline`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`
- fixed payload: indices + q residual values
- raw deterministic payload: q residual values only; decoder supplies endpoint-diff indices
- compressed deterministic payload: zlib(metadata) + zlib(q residual values); no indices
- no rendering, no training, no checkpoint, no heavy tensor output

## Summary

| base | codec | gap | tasks | raw det bytes | compressed det bytes | comp/raw | Stage96 entropy bytes | comp - entropy | max raw diff | max fixed diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 3 | 36009.0 | 25652.333333333332 | 0.712386718135281 | 29869.833333333332 | -4217.499999999999 | 0.0 | 0.0 |
| linear | q12 | 8 | 5 | 36009.0 | 27322.2 | 0.7587603099225195 | 30440.894736842107 | -3118.694736842107 | 0.0 | 0.0 |
| linear | q12 | 16 | 4 | 36009.0 | 26884.0 | 0.746591129995279 | 30239.565217391304 | -3355.565217391304 | 0.0 | 0.0 |
| stage65_adapter | q12 | 4 | 3 | 36009.0 | 30934.666666666668 | 0.8590815259148177 | 34757.88888888889 | -3823.222222222224 | 0.0 | 0.0 |
| stage65_adapter | q12 | 8 | 5 | 36009.0 | 32457.6 | 0.9013746563359162 | 35513.15789473684 | -3055.55789473684 | 0.0 | 0.0 |
| stage65_adapter | q12 | 16 | 4 | 36009.0 | 32162.5 | 0.893179482907051 | 35406.434782608696 | -3243.934782608696 | 0.0 | 0.0 |

## Notes

- Compressed deterministic decode is compared against raw deterministic decode and fixed index+value decode.
- Stage96 entropy reference is a broader q6/top10 index+value side-info rate reference, not a task-matched quality comparison.
- Residual values remain teacher-derived; this is still not residual value prediction.
