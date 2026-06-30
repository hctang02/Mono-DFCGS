# Stage174 Medium Rendered Validation Execution

## Scope

This executes the Stage173 protocol by reusing Stage167/168/170 rows, rendering missing rows, and marking target keyframes without middle metrics.

## Source Coverage

| source | rows | rendered | keyframe markers |
|---|---:|---:|---:|
| stage168 | 6 | 4 | 2 |
| stage170 | 78 | 60 | 18 |
| stage174_keyframe_marker | 12 | 0 | 12 |
| stage174_rendered | 54 | 54 | 0 |

## Category/Schedule Summary

| category | schedule | rows | rendered | keyframes | reused | new renders | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | stage165_adaptive | 8 | 8 | 0 | 8 | 0 | 26.192677 | 0.833774 | 0.981409 | 0.208831 | 155993.625000 |
| false_negative_residual | uniform_gap4 | 8 | 7 | 1 | 8 | 0 | 25.683246 | 0.843334 | 0.983516 | 0.193556 | 149221.571429 |
| false_negative_residual | uniform_gap8 | 8 | 8 | 0 | 8 | 0 | 26.203641 | 0.834367 | 0.981569 | 0.207921 | 156389.625000 |
| high_payload_residual_control | stage165_adaptive | 4 | 4 | 0 | 4 | 0 | 31.354837 | 0.890757 | 0.987188 | 0.150451 | 239237.250000 |
| high_payload_residual_control | uniform_gap4 | 4 | 4 | 0 | 4 | 0 | 31.078070 | 0.881161 | 0.985986 | 0.166005 | 241069.750000 |
| high_payload_residual_control | uniform_gap8 | 4 | 4 | 0 | 4 | 0 | 31.039273 | 0.880826 | 0.985841 | 0.167492 | 239014.000000 |
| high_payload_residual_control_extension | stage165_adaptive | 8 | 8 | 0 | 0 | 8 | 29.539710 | 0.862902 | 0.983418 | 0.173480 | 225694.750000 |
| high_payload_residual_control_extension | uniform_gap4 | 8 | 7 | 1 | 0 | 7 | 29.230789 | 0.866007 | 0.984011 | 0.167908 | 223694.428571 |
| high_payload_residual_control_extension | uniform_gap8 | 8 | 8 | 0 | 0 | 8 | 29.539710 | 0.862902 | 0.983418 | 0.173480 | 225694.750000 |
| normal_residual_control | stage165_adaptive | 4 | 4 | 0 | 0 | 4 | 33.406462 | 0.905324 | 0.981935 | 0.075904 | 193691.000000 |
| normal_residual_control | uniform_gap4 | 4 | 4 | 0 | 0 | 4 | 33.417668 | 0.904064 | 0.981687 | 0.078007 | 195828.250000 |
| normal_residual_control | uniform_gap8 | 4 | 4 | 0 | 0 | 4 | 33.406462 | 0.905324 | 0.981935 | 0.075904 | 193691.000000 |
| positive_promoted | stage165_adaptive | 14 | 0 | 14 | 14 | 0 | NA | NA | NA | NA | NA |
| positive_promoted | uniform_gap4 | 14 | 11 | 3 | 14 | 0 | 28.894486 | 0.854146 | 0.981621 | 0.205379 | 230308.818182 |
| positive_promoted | uniform_gap8 | 14 | 14 | 0 | 14 | 0 | 28.467816 | 0.846066 | 0.980992 | 0.215784 | 238069.214286 |
| positive_promoted_extension | stage165_adaptive | 8 | 0 | 8 | 2 | 0 | NA | NA | NA | NA | NA |
| positive_promoted_extension | uniform_gap4 | 8 | 7 | 1 | 2 | 5 | 26.718024 | 0.816261 | 0.979762 | 0.219088 | 221819.142857 |
| positive_promoted_extension | uniform_gap8 | 8 | 8 | 0 | 2 | 6 | 26.910021 | 0.823979 | 0.980747 | 0.211999 | 227130.875000 |
| selector_false_positive_keyframe_control | stage165_adaptive | 4 | 0 | 4 | 0 | 0 | NA | NA | NA | NA | NA |
| selector_false_positive_keyframe_control | uniform_gap4 | 4 | 4 | 0 | 0 | 4 | 30.188225 | 0.881831 | 0.985054 | 0.173437 | 154494.250000 |
| selector_false_positive_keyframe_control | uniform_gap8 | 4 | 4 | 0 | 0 | 4 | 30.071755 | 0.876903 | 0.984161 | 0.183046 | 157481.250000 |

## Decision

- Decision: `medium_validation_ready_for_decision`.
- Protocol rows covered: `150 / 150`.
- New renders completed: `54`.
- Reused rows: `84`.
- Keyframe marker rows: `32`.
- Heavy contact sheet for new renders: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage174_medium_rendered_validation_execution/stage174_medium_rendered_validation_contact_sheet.jpg`.
