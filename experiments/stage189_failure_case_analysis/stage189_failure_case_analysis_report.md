# Stage189 Failure-Case Analysis

## Decision

- Decision: `failure_cases_identified_for_paper_and_next_selector_refinement`.
- Promotion rows analyzed: `66`; promoted rate-risk rows: `2`.
- Residual risk rows: `1179`.

## Candidate Summary

| candidate | keyframes | MiB/frame | dRate vs gap8 | dRate vs full | PSNR | dPSNR vs gap8 | LPIPS | changed frames vs full | worst changed dPSNR vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_top10pct_cells | 299 | 0.277374617752 | 0.001490326748 | -0.013390979986 | 29.381126 | 0.007161 | 0.168325 | 370 | -5.309561 |
| interval_score_ge4p0 | 324 | 0.282992049060 | 0.007107758057 | -0.007773548678 | 29.410133 | 0.036168 | 0.166827 | 223 | -2.389079 |
| interval_top90pct_cells | 353 | 0.289479501370 | 0.013595210367 | -0.001286096368 | 29.424507 | 0.050542 | 0.166018 | 35 | -1.277667 |

## Worst Promotion Rate Risks

| sequence | frame | dPSNR adaptive-gap8 | dLPIPS adaptive-gap8 | local payload delta | reason |
|---|---:|---:|---:|---:|---|
| drift-chicane | 6 | 0.166051 | -0.024552 | 594793 | small_unlabeled_gain large_local_payload_delta |
| horsejump-high | 15 | 0.177171 | -0.020872 | 506463 | small_unlabeled_gain large_local_payload_delta |

## Worst Residual False-Negative Risks

| sequence | frame | adaptive PSNR | adaptive LPIPS | payload | gap4 dPSNR | gap4 dLPIPS | flags |
|---|---:|---:|---:|---:|---:|---:|---|
| motocross-jump | 28 | 30.652791 | 0.390020 | 234163 | 4.539777 | -0.302427 | high_lpips high_payload |
| india | 36 | 31.894848 | 0.363657 | 247423 | 6.855280 | -0.294082 | high_lpips high_payload |
| motocross-jump | 11 | 29.957049 | 0.342437 | 251962 | 2.088034 | -0.081259 | high_lpips high_payload |
| india | 37 | 32.199871 | 0.334377 | 248468 | 2.597263 | -0.073308 | high_lpips high_payload |
| motocross-jump | 27 | 32.077485 | 0.349502 | 233001 | 1.862792 | -0.078807 | high_lpips high_payload |
| motocross-jump | 19 | 29.915212 | 0.306025 | 264542 | 2.731448 | -0.093566 | high_lpips high_payload |
| motocross-jump | 10 | 30.574945 | 0.330977 | 232116 | -0.756898 | 0.022831 | high_lpips high_payload |
| motocross-jump | 29 | 32.469014 | 0.330160 | 230914 | 2.230374 | -0.117290 | high_lpips high_payload |
| mbike-trick | 69 | 28.665846 | 0.335397 | 212087 | 1.233683 | -0.104075 | high_lpips |
| india | 35 | 32.863265 | 0.308717 | 245846 | 2.554295 | -0.110458 | high_lpips high_payload |
| mbike-trick | 75 | 28.057404 | 0.327847 | 217490 | 0.893577 | -0.071684 | high_lpips |
| mbike-trick | 68 | 28.848012 | 0.327685 | 216475 | 1.437641 | -0.174818 | high_lpips |

## Sequence Hotspots

| sequence | promotion risks | residual risks | low PSNR | high LPIPS | high payload | max residual risk | worst residual frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| cows | 0 | 86 | 86 | 1 | 62 | 1.904395 | 92 |
| parkour | 0 | 75 | 27 | 26 | 74 | 1.560360 | 68 |
| camel | 0 | 73 | 61 | 0 | 67 | 1.393465 | 60 |
| goat | 0 | 73 | 0 | 0 | 73 | 0.789140 | 5 |
| breakdance | 0 | 72 | 72 | 2 | 0 | 1.748598 | 5 |
| soapbox | 0 | 72 | 0 | 1 | 71 | 0.673080 | 67 |
| bmx-trees | 0 | 67 | 3 | 6 | 67 | 0.707380 | 44 |
| car-roundabout | 0 | 55 | 13 | 0 | 55 | 0.856995 | 61 |
| dance-twirl | 0 | 54 | 2 | 35 | 28 | 1.665491 | 60 |
| india | 0 | 52 | 0 | 7 | 52 | 3.421599 | 36 |

## Interpretation

- Rate overhead is driven by promoted keyframes whose local keyframe cost is much larger than the replaced gap8 residual payload.
- The lowest-rate Stage188 candidate keeps only the strongest cells and therefore gives only a small quality gain over gap8.
- The balanced candidate keeps enough high-score cells to preserve most visible/metric gains while cutting about half the full adaptive overhead.
- Remaining false negatives are residual frames with low PSNR, high LPIPS, or high residual payload that were not promoted.

## Outputs

- Promotion analysis CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage189_failure_case_analysis/stage189_promoted_keyframe_false_positive_analysis.csv`
- Residual risk CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage189_failure_case_analysis/stage189_unpromoted_residual_false_negative_risks.csv`
- Candidate dropped-frame CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage189_failure_case_analysis/stage189_candidate_dropped_frame_loss_analysis.csv`
- Candidate summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage189_failure_case_analysis/stage189_candidate_failure_summary.csv`
- Sequence summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage189_failure_case_analysis/stage189_sequence_level_failure_summary.csv`
- Sequence hotspot CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage189_failure_case_analysis/stage189_sequence_hotspots.csv`
