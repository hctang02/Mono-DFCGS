# Stage75 Corrected StreamSplat Paper-Protocol DAVIS Package

Date: 2026-06-27

## Goal

Package the corrected StreamSplat DAVIS baseline using Stage74 official-style outputs.

## Plan

- Use Stage74 full DAVIS val sliding-window results.
- Report `256x256` paper-style metrics.
- Highlight middle/non-input PSNR for gap5 and gap8.
- Compare against paper values where applicable.
- Keep Stage72 scoped results separate as Mono-DFCGS diagnostics.

## Inputs

```text
experiments/stage74_stage72_vs_actual_gap_diagnosis_full_val_sliding_per_frame/stage74_stage72_vs_actual_gap_diagnosis_aggregate.csv
experiments/stage74_stage72_vs_actual_gap_diagnosis_full_val_sliding_per_frame/stage74_stage72_vs_actual_gap_diagnosis_rows.csv
```

## Expected Outputs

```text
scripts/run_stage75_corrected_streamsplat_paper_protocol_package.py
experiments/stage75_corrected_streamsplat_paper_protocol_package/
```

## Result

Stage75 generated the corrected paper-protocol package from Stage74 outputs.

Outputs:

```text
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_paper_protocol_summary.json
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_paper_protocol_summary.csv
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_per_sequence.csv
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_stage72_vs_corrected_comparison.csv
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_paper_protocol_report.md
```

Corrected baseline:

| paper setting | local gap | pair count | all PSNR | middle PSNR | given PSNR | paper PSNR | local - paper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Middle-4 frames | 5 | 1849 | 26.994540075591946 | 23.004337221027775 | 34.97494578472027 | 23.66 | -0.6556627789722249 |
| 8-frame interval | 8 | 1759 | 24.534872014837706 | 21.56004909948801 | 34.94675221856166 | 22.1 | -0.5399509005119931 |

Stage75 should be used as the local StreamSplat DAVIS paper-protocol reference. Stage72 remains a scoped diagnostic baseline only.
