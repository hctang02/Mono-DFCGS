# Stage115 Deterministic-Index Residual Codec Smoke

## Configuration

- task count: `12`
- selector policy: `strict_safe_endpoint_selector_v1`
- selected candidate: `endpoint_diff_baseline`
- keep fraction: `0.1`
- side bits: `6`
- fixed payload: indices + q residual values
- deterministic payload: q residual values only; decoder supplies deterministic endpoint-diff indices
- no rendering, no training, no checkpoint, no heavy tensor output

## Summary

| base | codec | gap | tasks | fixed bytes | deterministic bytes | saved bytes | ratio | det MiB/intermediate | max decode diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 3 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |
| linear | q12 | 8 | 5 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388002 | 0.034340858459472656 | 0.0 |
| linear | q12 | 16 | 4 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |
| stage65_adapter | q12 | 4 | 3 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |
| stage65_adapter | q12 | 8 | 5 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388002 | 0.034340858459472656 | 0.0 |
| stage65_adapter | q12 | 16 | 4 | 43381.0 | 36009.0 | 7372.0 | 0.8300638528388004 | 0.034340858459472656 | 0.0 |

## Notes

- The deterministic payload still transmits teacher residual values; this is not residual value prediction.
- Index bytes are removed only because the decoder can deterministically reproduce endpoint-diff selected indices from left/right anchors.
- The side-info payload remains transmitted information and must be counted in total RD rate.
