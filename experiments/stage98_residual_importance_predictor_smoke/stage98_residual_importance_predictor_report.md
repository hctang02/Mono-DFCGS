# Stage98 Residual Importance Predictor Smoke

## Configuration

- train tasks: `24`
- eval tasks: `12`
- train examples: `196608`
- keep fraction: `0.1`
- train steps: `200`
- labels are teacher top10 residual-energy masks generated from target dense anchors
- predictor inputs are decoder-available anchor features plus time and base method id

## Summary

| candidate | base | gap | tasks | precision@keep | energy recall | oracle recall | relative recall |
|---|---|---:|---:|---:|---:|---:|---:|
| endpoint_diff_baseline | linear | 4 | 6 | 0.3122626145680745 | 0.2655188664793968 | 0.6100062330563863 | 0.42121463517347973 |
| endpoint_diff_baseline | linear | 8 | 3 | 0.24724181989828745 | 0.23293344179789224 | 0.6551332771778107 | 0.34736502170562744 |
| endpoint_diff_baseline | linear | 16 | 3 | 0.2551998645067215 | 0.29756257434686023 | 0.6923819482326508 | 0.38490988314151764 |
| endpoint_diff_baseline | stage65_adapter | 4 | 6 | 0.27450714260339737 | 0.12835493932167688 | 0.21536186089118323 | 0.6087801853815714 |
| endpoint_diff_baseline | stage65_adapter | 8 | 3 | 0.2631578892469406 | 0.13744011024634042 | 0.27263158559799194 | 0.5076933900515238 |
| endpoint_diff_baseline | stage65_adapter | 16 | 3 | 0.18972689906756082 | 0.11895131816466649 | 0.2499206562836965 | 0.5244861443837484 |
| mlp_importance | linear | 4 | 6 | 0.310047022998333 | 0.2947804356614749 | 0.6100062330563863 | 0.46986613670984906 |
| mlp_importance | linear | 8 | 3 | 0.292729248603185 | 0.2800879975159963 | 0.6551332771778107 | 0.4127636154492696 |
| mlp_importance | linear | 16 | 3 | 0.26433352132638294 | 0.3037059058745702 | 0.6923819482326508 | 0.4129539728164673 |
| mlp_importance | stage65_adapter | 4 | 6 | 0.3889039605855942 | 0.14692467202742895 | 0.21536186089118323 | 0.6976491312185923 |
| mlp_importance | stage65_adapter | 8 | 3 | 0.37384698788324994 | 0.1679172863562902 | 0.27263158559799194 | 0.6167584160963694 |
| mlp_importance | stage65_adapter | 16 | 3 | 0.28431904315948486 | 0.14672287801901499 | 0.2499206562836965 | 0.6156713565190634 |

## Notes

- This stage does not render or transmit predicted residual values.
- It measures whether a feed-forward predictor can localize high-energy residual Gaussians.
- Target dense anchors are used only for offline labels and metrics, not as predictor inputs.
