# Stage170 Combined Adaptive Rendered Validation Execution

## Scope

This executes the Stage169 protocol by reusing Stage167/168 rows, rendering only missing rows, and marking adaptive/keyframe rows without claiming middle-render metrics.

## Source Coverage

| source | rows | rendered | keyframe markers |
|---|---:|---:|---:|
| stage167 | 24 | 23 | 1 |
| stage168 | 18 | 11 | 7 |
| stage170_keyframe_marker | 10 | 0 | 10 |
| stage170_rendered | 26 | 26 | 0 |

## Category/Schedule Summary

| category | schedule | rows | rendered | keyframes | reused | new renders | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | stage165_adaptive | 8 | 8 | 0 | 8 | 0 | 26.192677 | 0.833774 | 0.981409 | 0.208831 | 155993.625000 |
| false_negative_residual | uniform_gap4 | 8 | 7 | 1 | 8 | 0 | 25.683246 | 0.843334 | 0.983516 | 0.193556 | 149221.571429 |
| false_negative_residual | uniform_gap8 | 8 | 8 | 0 | 8 | 0 | 26.203641 | 0.834367 | 0.981569 | 0.207921 | 156389.625000 |
| high_payload_residual_control | stage165_adaptive | 4 | 4 | 0 | 0 | 4 | 31.354837 | 0.890757 | 0.987188 | 0.150451 | 239237.250000 |
| high_payload_residual_control | uniform_gap4 | 4 | 4 | 0 | 0 | 4 | 31.078070 | 0.881161 | 0.985986 | 0.166005 | 241069.750000 |
| high_payload_residual_control | uniform_gap8 | 4 | 4 | 0 | 0 | 4 | 31.039273 | 0.880826 | 0.985841 | 0.167492 | 239014.000000 |
| positive_promoted | stage165_adaptive | 14 | 0 | 14 | 6 | 0 | NA | NA | NA | NA | NA |
| positive_promoted | uniform_gap4 | 14 | 11 | 3 | 6 | 6 | 28.894486 | 0.854146 | 0.981621 | 0.205379 | 230308.818182 |
| positive_promoted | uniform_gap8 | 14 | 14 | 0 | 6 | 8 | 28.467816 | 0.846066 | 0.980992 | 0.215784 | 238069.214286 |

## Decision

- Decision: `combined_validation_ready_for_review`.
- New renders completed: `26`.
- Reused rows: `42`.
- Keyframe marker rows: `10`.
- Heavy contact sheet for new renders: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_adaptive_rendered_validation_contact_sheet.jpg`.
- This remains sampled validation; full-sequence RD should only follow if this combined evidence is acceptable.
