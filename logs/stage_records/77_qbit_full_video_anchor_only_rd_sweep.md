# Stage77 Q-Bit Full-Video Anchor-Only RD Sweep

Date: 2026-06-27

## Goal

Evaluate whether q10/q12 static anchor codecs improve full-video anchor-only RD relative to the Stage70 q8 protocol.

## Plan

- Scope: `DAVIS/val/bmx-trees`, `DAVIS/val/car-shadow`, `DAVIS/val/goat`, `DAVIS/val/soapbox`.
- Gaps: `4`, `8`, `16`.
- Codecs: `q8`, `q10`, `q12`.
- Methods: linear anchor interpolation and Stage65 `rgb_h256` adapter.
- Metrics: all/middle/given PSNR.
- Rate: q-bit static anchor payload MiB/frame with fp16 min/scale metadata.

## Expected Outputs

```text
scripts/run_stage77_qbit_full_video_anchor_only_rd_sweep.py
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/
```

## Result

Stage77 completed on the Stage70 scoped DAVIS val sequences.

Outputs:

```text
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_summary.json
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_summary.csv
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_rows.csv
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_report.md
```

Adapter summary:

| codec | gap | MiB/frame | all PSNR | middle PSNR | given PSNR | delta all vs q8 | delta middle vs q8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| q8 | 4 | 0.12129653387470925 | 20.57270098931695 | 18.247303018014392 | 27.02327983720894 | 0.0 | 0.0 |
| q10 | 4 | 0.1516173773213112 | 21.191095902140322 | 18.25601918552348 | 29.327984245556006 | 0.6183949128233728 | 0.00871616750908899 |
| q12 | 4 | 0.18193822076791313 | 21.284133638556813 | 18.256196169477683 | 29.678129037019875 | 0.7114326492398639 | 0.008893151463290394 |
| q8 | 8 | 0.06508594500594155 | 18.480296729121694 | 17.068303922946175 | 27.01806414475214 | 0.0 | 0.0 |
| q10 | 8 | 0.08135566587972795 | 18.809757721047486 | 17.06974217895638 | 29.320664241574747 | 0.32946099192579226 | 0.0014382560102035313 |
| q12 | 8 | 0.09762538675351436 | 18.85917872255155 | 17.06969395261803 | 29.668274733935373 | 0.3788819934298573 | 0.0013900296718567517 |
| q8 | 16 | 0.036980650571557694 | 16.865547380070296 | 15.978911066116073 | 26.997379801213054 | 0.0 | 0.0 |
| q10 | 16 | 0.046224810158936334 | 17.052739413791674 | 15.97580514912888 | 29.328475996349997 | 0.18719203372137727 | -0.0031059169871934245 |
| q12 | 16 | 0.055468969746314975 | 17.081329746423872 | 15.975915248513115 | 29.679767860200425 | 0.21578236635357584 | -0.002995817602958084 |

Conclusion:

- q10/q12 improve all-frame PSNR mostly through higher-quality keyframes.
- Middle-frame PSNR barely changes for the Stage65 adapter, so the remaining bottleneck is static-anchor-only dynamic prediction/modeling, not just q8 quantization.
- Future improvements should include stronger predictor training, rendered-label selector, or dynamic side information in addition to q10/q12 codec points.
