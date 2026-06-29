# Stage143 Middle-Frame PSNR Collapse Diagnostic

## Key Summary

| codec | gap | method | rate | middle PSNR | given PSNR | delta vs linear | gap to Stage75 target |
|---|---:|---|---:|---:|---:|---:|---:|
| q12 | 4 | adapter | 0.18193822076791313 | 18.25528373980314 | 29.668876650960268 | 0.5952401824727076 | -4.749053481224635 |
| q12 | 4 | dense_direct | 0.18193822076791313 | 29.722410525860948 | 29.668876650960268 | 12.062366968530515 | 6.718073304833172 |
| q12 | 4 | linear | 0.18193822076791313 | 17.660043557330432 | 29.668876650960268 | 0.0 | -5.344293663697343 |
| q12 | 8 | adapter | 0.09762538675351436 | 17.06788939572344 | 29.64762580458447 | 0.5261505815582552 | -4.4921597037645675 |
| q12 | 8 | dense_direct | 0.09762538675351436 | 29.718262412631926 | 29.64762580458447 | 13.17652359846674 | 8.158213313143918 |
| q12 | 8 | linear | 0.09762538675351436 | 16.541738814165186 | 29.64762580458447 | 0.0 | -5.018310285322823 |
| q16 | 4 | adapter | 0.242579907661117 | 18.255332804105823 | 29.695719107336018 | 0.5950336233349702 | -4.749004416921952 |
| q16 | 4 | dense_direct | 0.242579907661117 | 29.749435689288717 | 29.695719107336018 | 12.089136508517864 | 6.745098468260942 |
| q16 | 4 | linear | 0.242579907661117 | 17.660299180770853 | 29.695719107336018 | 0.0 | -5.3440380402569225 |
| q16 | 8 | adapter | 0.13016482850108718 | 17.067879472412265 | 29.674268643648432 | 0.5260469625346609 | -4.492169627075743 |
| q16 | 8 | dense_direct | 0.13016482850108718 | 29.745294520564382 | 29.674268643648432 | 13.203462010686778 | 8.185245421076374 |
| q16 | 8 | linear | 0.13016482850108718 | 16.541832509877604 | 29.674268643648432 | 0.0 | -5.018216589610404 |
| float32 | 4 | adapter | 0.4851466552339325 | 18.255332640417755 | 29.69590326065567 | 0.595031334182476 | -4.74900458061002 |
| float32 | 4 | dense_direct | 0.4851466552339325 | 29.749654363336436 | 29.69590326065567 | 12.089353057101157 | 6.745317142308661 |
| float32 | 4 | linear | 0.4851466552339325 | 17.66030130623528 | 29.69590326065567 | 0.0 | -5.344035914792496 |
| float32 | 8 | adapter | 0.26032259549137843 | 17.067872741131573 | 29.674475107245403 | 0.5260375457415343 | -4.492176358356435 |
| float32 | 8 | dense_direct | 0.26032259549137843 | 29.74550454012203 | 29.674475107245403 | 13.203669344731992 | 8.185455440634023 |
| float32 | 8 | linear | 0.26032259549137843 | 16.54183519539004 | 29.674475107245403 | 0.0 | -5.018213904097969 |

## Findings

| severity | finding | evidence | required action |
|---|---|---|---|
| critical | gap4 float32 dense-direct ceiling greatly exceeds adapter prediction | dense_direct=29.749654363336436, adapter=18.255332640417755, dense-adapter=11.49432172291868 | Prioritize dynamic adapter training and/or rate-counted motion/residual side-info. |
| medium | gap4 q16-vs-q12 quantization sensitivity | q16_adapter=18.255332804105823, q12_adapter=18.25528373980314, q16-q12=4.9064302682921834e-05 | If q_gap is small, raising keyframe quantization alone will not recover paper-level middle PSNR. |
| critical | gap8 float32 dense-direct ceiling greatly exceeds adapter prediction | dense_direct=29.74550454012203, adapter=17.067872741131573, dense-adapter=12.677631798990458 | Prioritize dynamic adapter training and/or rate-counted motion/residual side-info. |
| medium | gap8 q16-vs-q12 quantization sensitivity | q16_adapter=17.067879472412265, q12_adapter=17.06788939572344, q16-q12=-9.92331117544154e-06 | If q_gap is small, raising keyframe quantization alone will not recover paper-level middle PSNR. |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_summary.csv`
- findings CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_findings.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_report.md`
