# Stage95 Broader Entropy-Coded Residual Side-Info Eval

## Configuration

- task count: `60`
- codecs: `['q12']`
- gaps: `[4, 8, 16]`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`

## Summary

| base | codec | gap | tasks | fixed bytes | entropy bytes | ratio | entropy MiB/intermediate | max decode diff | entropy PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 18 | 43381.0 | 29869.833333333332 | 0.6885464450642754 | 0.028486092885335285 | 0.0 | 23.403532118481326 | 3.4187356956292225 | 18 |
| linear | q12 | 8 | 19 | 43381.0 | 30440.894736842107 | 0.7017103048994284 | 0.02903069947895251 | 0.0 | 21.51362735577916 | 3.05603754877287 | 19 |
| linear | q12 | 16 | 23 | 43381.0 | 30239.565217391304 | 0.6970693441228029 | 0.028838696687117867 | 0.0 | 20.344147709407626 | 3.2599322732336695 | 23 |
| stage65_adapter | q12 | 4 | 18 | 43381.0 | 34757.88888888889 | 0.8012237820448788 | 0.033147705925835505 | 0.0 | 22.841151135422116 | 2.7994188265782376 | 18 |
| stage65_adapter | q12 | 8 | 19 | 43381.0 | 35513.15789473684 | 0.8186339156482526 | 0.033867986578690376 | 0.0 | 21.39901144086742 | 2.692464222747122 | 19 |
| stage65_adapter | q12 | 16 | 23 | 43381.0 | 35406.434782608696 | 0.8161737807475322 | 0.03376620748768682 | 0.0 | 20.292022340267458 | 2.9637922008291264 | 23 |

## Notes

- Entropy payload uses sorted-index deltas and zlib-compressed metadata/index/residual components.
- Decode is compared against Stage91 fixed decode before rendering.
- Residuals are still teacher-derived; this is a codec smoke, not a deployable residual predictor.
