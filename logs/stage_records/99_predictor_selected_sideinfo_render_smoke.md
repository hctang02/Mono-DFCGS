# Stage99 Predictor-Selected Side-Info Rendered Smoke

Date: 2026-06-28

## Goal

Isolate residual side-info index-selection error by rendering predicted top10 indices with teacher residual values.

## Implementation

Added:

```text
scripts/run_stage99_predictor_selected_sideinfo_render_smoke.py
```

The stage trains the same small feed-forward importance predictor style as Stage98. At evaluation time it compares three index sets: teacher oracle top10, MLP predicted top10, and endpoint-difference top10. The selected indices receive q6 teacher residual values, so this stage tests selection quality only and is not a deployable final codec.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle, so the run used:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage99_predictor_selected_sideinfo_render_smoke.py
```

## Outputs

```text
experiments/stage99_predictor_selected_sideinfo_render_smoke/stage99_predictor_selected_sideinfo_rows.csv
experiments/stage99_predictor_selected_sideinfo_render_smoke/stage99_predictor_selected_sideinfo_summary.csv
experiments/stage99_predictor_selected_sideinfo_render_smoke/stage99_predictor_selected_sideinfo_train_log.csv
experiments/stage99_predictor_selected_sideinfo_render_smoke/stage99_predictor_selected_sideinfo_summary.json
experiments/stage99_predictor_selected_sideinfo_render_smoke/stage99_predictor_selected_sideinfo_report.md
```

## Results

| candidate | base | gap | precision | energy recall | side PSNR | teacher PSNR | delta | gap to teacher |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| endpoint_diff_baseline | linear | 4 | 0.3122626145680745 | 0.2655188664793968 | 22.458850438236322 | 23.885431661756513 | 1.1001223888947578 | -1.4265812235201907 |
| endpoint_diff_baseline | linear | 8 | 0.24724181989828745 | 0.23293344179789224 | 19.175455055729827 | 21.570274743383234 | 1.316684641288311 | -2.3948196876534076 |
| endpoint_diff_baseline | linear | 16 | 0.2551998645067215 | 0.29756257434686023 | 21.141569946002704 | 23.557251435789382 | 1.3992381366908475 | -2.4156814897866785 |
| endpoint_diff_baseline | stage65_adapter | 4 | 0.27450714260339737 | 0.12835493932167688 | 22.477210308778595 | 23.578794305433277 | 1.0642003339816728 | -1.1015839966546832 |
| endpoint_diff_baseline | stage65_adapter | 8 | 0.2631578892469406 | 0.13744011024634042 | 19.71179701504451 | 21.846056896259956 | 1.1427972088150398 | -2.134259881215447 |
| endpoint_diff_baseline | stage65_adapter | 16 | 0.18972689906756082 | 0.11895131816466649 | 20.34492281493807 | 22.273016824369922 | 1.0102426189350968 | -1.9280940094318566 |
| mlp_importance | linear | 4 | 0.310047022998333 | 0.2947804356614749 | 22.234808578712798 | 23.885431661756513 | 0.8760805293712325 | -1.650623083043716 |
| mlp_importance | linear | 8 | 0.292729248603185 | 0.2800879975159963 | 19.37854214011173 | 21.570274743383234 | 1.519771725670217 | -2.1917326032715017 |
| mlp_importance | linear | 16 | 0.26433352132638294 | 0.3037059058745702 | 21.05049723125056 | 23.557251435789382 | 1.308165421938705 | -2.506754204538821 |
| mlp_importance | stage65_adapter | 4 | 0.3889039605855942 | 0.14692467202742895 | 22.23873956177556 | 23.578794305433277 | 0.825729586978642 | -1.3400547436577142 |
| mlp_importance | stage65_adapter | 8 | 0.37384698788324994 | 0.1679172863562902 | 19.824055179121583 | 21.846056896259956 | 1.2550553728921099 | -2.0220017171383766 |
| mlp_importance | stage65_adapter | 16 | 0.28431904315948486 | 0.14672287801901499 | 20.408320198765917 | 22.273016824369922 | 1.0736400027629471 | -1.8646966256040063 |
| teacher_oracle_topk | linear | 4 | 1.0 | 0.6100062330563863 | 23.885431661756513 | 23.885431661756513 | 2.5267036124149485 | 0.0 |
| teacher_oracle_topk | linear | 8 | 1.0 | 0.6551332771778107 | 21.570274743383234 | 21.570274743383234 | 3.7115043289417184 | 0.0 |
| teacher_oracle_topk | linear | 16 | 1.0 | 0.6923819482326508 | 23.557251435789382 | 23.557251435789382 | 3.814919626477526 | 0.0 |
| teacher_oracle_topk | stage65_adapter | 4 | 1.0 | 0.21536186089118323 | 23.578794305433277 | 23.578794305433277 | 2.165784330636356 | 0.0 |
| teacher_oracle_topk | stage65_adapter | 8 | 1.0 | 0.27263158559799194 | 21.846056896259956 | 21.846056896259956 | 3.2770570900304867 | 0.0 |
| teacher_oracle_topk | stage65_adapter | 16 | 1.0 | 0.2499206562836965 | 22.273016824369922 | 22.273016824369922 | 2.938336628366953 | 0.0 |

## Conclusion

- MLP-selected q6 teacher residual side-info gives positive rendered PSNR gains for every base/gap group.
- MLP selection is not consistently better than endpoint-difference selection in rendered PSNR.
- The MLP remains `1.34-2.51 dB` below teacher-oracle top10, so selection error is still a major bottleneck.
- Next step should improve the selector objective or scale training before attempting residual value prediction.
