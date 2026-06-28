# Stage78 Integrated DAVIS RD Package

Date: 2026-06-28

## Goal

Integrate the corrected StreamSplat DAVIS reference from Stage75 with the q-bit anchor-only RD sweep from Stage77.

## Scope

- No FCGS/D-FCGS baseline work in this stage.
- CPU-only package generation from existing CSV/JSON artifacts.
- Methods included: q8/q10/q12 linear anchor, q8/q10/q12 Stage65 adapter.
- Reference included: corrected StreamSplat paper-protocol DAVIS results from Stage75.

## Inputs

```text
experiments/stage75_corrected_streamsplat_paper_protocol_package/
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/
```

## Expected Outputs

```text
scripts/run_stage78_integrated_davis_rd_package.py
experiments/stage78_integrated_davis_rd_package/
```

Expected package contents:

- q-bit anchor-only rate table.
- all/middle/given PSNR tables.
- method summary table.
- reference gap table against corrected StreamSplat.
- RD curve plot.

## Result

Stage78 completed as a CPU-only package generation stage.

Outputs:

```text
experiments/stage78_integrated_davis_rd_package/stage78_integrated_davis_rd_summary.json
experiments/stage78_integrated_davis_rd_package/stage78_integrated_davis_rd_report.md
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_psnr_table.csv
experiments/stage78_integrated_davis_rd_package/stage78_method_summary.csv
experiments/stage78_integrated_davis_rd_package/stage78_reference_gap_table.csv
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_all_rd_curve.png
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_middle_rd_curve.png
```

Method averages across gaps `4/8/16`:

| method | codec | mean rate | mean all PSNR | mean middle PSNR | mean given PSNR |
|---|---|---:|---:|---:|---:|
| linear | q8 | 0.0744543764840695 | 18.196689289875984 | 16.561587772918216 | 27.01290792772471 |
| linear | q10 | 0.09306595111999183 | 18.589141793730672 | 16.581582825535197 | 29.32570816116025 |
| linear | q12 | 0.11167752575591416 | 18.648513030391765 | 16.58455938471112 | 29.67539054371856 |
| adapter | q8 | 0.0744543764840695 | 18.639515032836314 | 17.098172669025544 | 27.01290792772471 |
| adapter | q10 | 0.09306595111999183 | 19.017864345659827 | 17.100522171202915 | 29.32570816116025 |
| adapter | q12 | 0.11167752575591416 | 19.074880702510743 | 17.100601790202944 | 29.67539054371856 |

Reference gap diagnostics:

- Best q12 adapter gap4 middle PSNR: `18.256196169477683`.
- Corrected StreamSplat Middle-4 reference: `23.004337221027775`.
- Diagnostic gap: `-4.748141051550093 dB`.
- Best q12 adapter gap8 middle PSNR: `17.06969395261803`.
- Corrected StreamSplat 8-frame reference: `21.56004909948801`.
- Diagnostic gap: `-4.490355146869977 dB`.

Conclusion:

- Stage78 is the new clean anchor-only RD package for our innovation path.
- q10/q12 are useful codec operating points, but the middle-frame gap to corrected StreamSplat remains about `4.5-4.8 dB` on comparable diagnostic settings.
- Next stages should prioritize stronger adapter training, rendered-label selector, and possible dynamic side information rather than FCGS/D-FCGS.
