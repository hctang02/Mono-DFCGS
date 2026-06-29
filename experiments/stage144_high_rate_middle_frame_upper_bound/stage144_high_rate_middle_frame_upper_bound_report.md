# Stage144 High-Rate Middle-Frame Upper Bound

## Upper-Bound Table

| gap | target | q12 adapter | q16 adapter | float32 adapter | float32 dense | float32-q12 adapter | dense-adapter | adapter-target | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 23.004337 | 18.255284 | 18.255333 | 18.255333 | 29.749654 | 0.000049 | 11.494322 | -4.749005 | model_training_or_sideinfo_required |
| 8 | 21.560049 | 17.067889 | 17.067879 | 17.067873 | 29.745505 | -0.000017 | 12.677632 | -4.492176 | model_training_or_sideinfo_required |

## Decisions

| item | decision | evidence |
|---|---|---|
| raise_anchor_qbit | reject_as_primary_fix | max abs(float32-q12 adapter middle) = 4.890061461537698e-05 dB |
| renderer_data_ceiling | not_bottleneck | min float32 dense gap to target = 6.745317142308661 dB |
| dynamic_model | primary_bottleneck | min dense-direct minus adapter middle = 11.49432172291868 dB; best float32 adapter target gap = -4.74900458061002 dB |
| next_stage | start_large_scale_adapter_training_and_prepare_rate_counted_sideinfo_fallback | High-rate anchors do not move adapter middle PSNR; dense-direct has ample ceiling. |

## Outputs

- upper-bound CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_rows.csv`
- decisions CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_decisions.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_report.md`
