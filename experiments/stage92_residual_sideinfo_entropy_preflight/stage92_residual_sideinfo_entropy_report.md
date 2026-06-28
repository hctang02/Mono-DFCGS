# Stage92 Residual Side-Info Entropy Preflight

## Configuration

- task count: `60`
- codecs: `['q12']`
- gaps: `[4, 8, 16]`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`
- no rendering is run in this stage

## Candidate Summary

| candidate | mean bytes | mean MiB/intermediate | ratio vs raw | savings bytes |
|---|---:|---:|---:|---:|
| delta_index_zlib | 32623.583333333332 | 0.031112273534138996 | 0.7520246959114205 | 10757.416666666668 |
| component_zlib | 37162.98333333333 | 0.035441382726033525 | 0.8566649762184673 | 6218.01666666667 |
| residual_zlib | 37422.4 | 0.03568878173828125 | 0.8626449367234504 | 5958.5999999999985 |
| zlib_whole | 37426.28333333333 | 0.035692485173543294 | 0.8627344536394581 | 5954.716666666667 |
| raw_fixed | 43381.0 | 0.04137134552001953 | 1.0 | 0.0 |

## Gap/Method Summary

| base | gap | tasks | raw bytes | zlib whole | component zlib | delta-index zlib | best bytes | best ratio | best counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| linear | 4 | 18 | 43381.0 | 34559.0 | 34314.333333333336 | 29669.38888888889 | 29669.38888888889 | 0.683925886652887 | `{"delta_index_zlib": 18}` |
| linear | 8 | 19 | 43381.0 | 34998.78947368421 | 34767.10526315789 | 30261.052631578947 | 30261.052631578947 | 0.6975646626767236 | `{"delta_index_zlib": 19}` |
| linear | 16 | 23 | 43381.0 | 34986.17391304348 | 34725.434782608696 | 30109.91304347826 | 30109.91304347826 | 0.6940806584329144 | `{"delta_index_zlib": 23}` |
| stage65_adapter | 4 | 18 | 43381.0 | 39556.166666666664 | 39268.72222222222 | 34731.444444444445 | 34731.444444444445 | 0.8006141961790748 | `{"delta_index_zlib": 18}` |
| stage65_adapter | 8 | 19 | 43381.0 | 40151.42105263158 | 39894.89473684211 | 35462.84210526316 | 35462.84210526316 | 0.8174740578885493 | `{"delta_index_zlib": 19}` |
| stage65_adapter | 16 | 23 | 43381.0 | 40197.608695652176 | 39904.34782608696 | 35405.782608695656 | 35405.782608695656 | 0.8161587471173013 | `{"delta_index_zlib": 23}` |

## Notes

- `raw_fixed` is the Stage91 fixed-length byte payload.
- `zlib_whole` compresses the whole fixed payload with zlib.
- `component_zlib` keeps the header raw but zlib-compresses metadata, indices, and residual bytes separately.
- `delta_index_zlib` sorts selected indices, stores uint16 deltas, and zlib-compresses those deltas plus metadata and residual bytes.
- These are preflight estimates for lossless byte coding; no rendered metrics are recomputed here.
