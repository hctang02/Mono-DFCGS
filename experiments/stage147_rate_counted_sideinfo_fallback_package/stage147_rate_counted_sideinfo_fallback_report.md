# Stage147 Rate-Counted Side-Info Fallback Package

## Summary

| gap | target | base PSNR | side PSNR | side-target | delta base | side MiB/frame | direct rate | amortized rate | positives | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 23.004337 | 20.041732 | 22.841151 | -0.163186 | 2.799419 | 0.033148 | 0.215086 | 0.206799 | 18/18 | promote_for_stage148_revalidation |
| 8 | 21.560049 | 18.706547 | 21.399011 | -0.161038 | 2.692464 | 0.033868 | 0.131493 | 0.127260 | 19/19 | promote_for_stage148_revalidation |

## Decisions

| item | decision | evidence |
|---|---|---|
| feedforward_only_path | not_enough_as_current_primary | Stage146 best_step=0; final mean PSNR delta vs init = -0.019897326878577815 dB |
| higher_qbit_path | already_rejected | quality collapse is dynamic-model-side; proceed to large-scale adapter training and keep side-info fallback |
| rate_counted_sideinfo_fallback | promote_for_stage148_revalidation | worst side-info gap to corrected target = -0.16318608560565906 dB with tolerance 0.25 dB |
| rate_accounting | all_payload_bytes_counted | max direct total rate among gap4/gap8 rows = 0.21508592669374865 MiB/frame; index payload is transmitted and counted |
| task_positivity | positive_on_all_stage96_gap4_gap8_adapter_tasks | min positive delta fraction = 1.0 |

## Contract

- policy: `rate_counted_entropy_index_value_residual_sideinfo_fallback_v1`
- side-info is a transmitted payload and all payload bytes are counted in total rate.
- Encoder-side target/intermediate information may be used only to form the payload.
- Decoder forbidden inputs: target dense anchor, target RGB, unencoded target residual tensor, oracle labels not represented in the payload.
- This is not the previously rejected uncounted teacher side-info framing.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv`
- decisions CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_decisions.csv`
- policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_policy.json`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_report.md`

## Limitation

Stage96 evidence is not the final full-video paper-protocol evaluation. Stage148 must revalidate this fallback before a final quality claim.
