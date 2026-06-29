# Stage151 Middle-Frame Recovery Policy Package

## Policy

- policy: `middle_frame_recovery_linear_base_entropy_sideinfo_v1`
- status: `target_recovered_on_full_q12_gap4_gap8_eval_rows`
- base: decoder-safe linear interpolation
- side-info: q6/top10 entropy index+value residual payload
- all side-info payload bytes are counted

## Evidence

| gap | target | achieved | achieved-target | tasks | positives | payload bytes | direct rate | amortized rate | decode diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.004337 | 23.104893 | 0.100556 | 1463 | 1463/1463 | 30062.867 | 0.210608 | 0.203441 | 0.000000 |
| 8 | 21.560049 | 22.020189 | 0.460140 | 1707 | 1707/1707 | 30203.919 | 0.126430 | 0.122830 | 0.000000 |

## Decoder Contract

- Allowed decoder inputs: left anchor, right anchor, normalized time, encoded side-info payload.
- Forbidden decoder inputs: target dense anchor, target RGB, unencoded target residual tensor, oracle labels not represented in payload.

## Outputs

- policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy.json`
- evidence CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_evidence.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage151_middle_frame_recovery_policy_package/stage151_middle_frame_recovery_policy_report.md`
