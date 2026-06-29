# Stage142 Middle-Frame Protocol Alignment Audit

## Target Protocol

| setting | local gap | paper PSNR | corrected local middle | corrected all | corrected given |
|---|---:|---:|---:|---:|---:|
| Middle-4 frames | 5 | 23.66 | 23.004337221027775 | 26.994540075591946 | 34.97494578472027 |
| 8-frame interval | 8 | 22.1 | 21.56004909948801 | 24.534872014837706 | 34.94675221856166 |

## Current Diagnostic Gap

| ours | gap | method | rate | ours middle | reference middle | gap to reference | final claim? |
|---|---:|---|---:|---:|---:|---:|---:|
| q12 | 4 | adapter | 0.18193822076791313 | 18.256196169477683 | 23.004337221027775 | -4.748141051550093 | 0 |
| q12 | 8 | adapter | 0.09762538675351436 | 17.06969395261803 | 21.56004909948801 | -4.490355146869977 | 0 |

## Findings

| severity | finding | required action |
|---|---|---|
| critical | Stage78 reference comparison is diagnostic, not a final apples-to-apples paper-protocol claim. | Rerun our anchor/adapter pipeline on the same full DAVIS val paper-style protocol before final claims. |
| critical | Current middle-frame quality is far below the corrected StreamSplat/paper-level target. | Run Stage143/144 to separate renderer/data/quantization/model causes, then train or add rate-counted side-info until middle PSNR recovers. |
| high | Stage141 final manifest is decoder-safe but is not a paper-level quality solution. | Keep Stage141 as a decoder-side accounting checkpoint only; do not treat it as final quality result. |
| high | q-bit increase from q8 to q12 has only small middle-frame gains, so quantization alone is unlikely to explain the full gap. | Stage144 must include uncompressed or high-rate anchors to prove the remaining ceiling; if ceiling stays low, prioritize dynamic model training. |

## Outputs

- target CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_targets.csv`
- comparison CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage142_middle_frame_protocol_alignment_audit/stage142_protocol_comparison_rows.csv`
- findings CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage142_middle_frame_protocol_alignment_audit/stage142_protocol_alignment_findings.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_protocol_alignment_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_protocol_alignment_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_protocol_alignment_report.md`
