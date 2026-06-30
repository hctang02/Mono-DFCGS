# Stage152 Subjective Visual Export

## Videos

| gap | frames | video | contact sheet | mean base PSNR | mean recovered PSNR |
|---:|---:|---|---|---:|---:|
| 4 | 24 | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap4_target_base_recovered.mp4` | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap4_contact_sheet.jpg` | 20.934637 | 24.157468 |
| 8 | 24 | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap8_target_base_recovered.mp4` | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export/stage152_gap8_contact_sheet.jpg` | 19.312466 | 22.586569 |

## Layout

Each frame is: target RGB | linear base render | recovered side-info render.

## Contract

The recovered panel uses the Stage151 policy: linear base plus decoded q6/top10 entropy index+value residual side-info payload.
