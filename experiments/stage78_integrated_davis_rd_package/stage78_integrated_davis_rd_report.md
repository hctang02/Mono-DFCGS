# Stage78 Integrated DAVIS RD Package

## Scope

- Anchor-only RD rows come from Stage77 scoped DAVIS val q8/q10/q12 sweep.
- StreamSplat reference rows come from Stage75 corrected paper-protocol package.
- FCGS/D-FCGS are intentionally not included in this package.
- Primary anchor-only metric remains all-frame PSNR; middle/given are diagnostic.

## Anchor-Only RD Summary

| codec | gap | method | MiB/frame | all PSNR | middle PSNR | given PSNR |
|---|---:|---|---:|---:|---:|---:|
| q8 | 4 | linear | 0.12129653387470925 | 20.110768144827567 | 17.61796945986582 | 27.02327983720894 |
| q8 | 4 | adapter | 0.12129653387470925 | 20.57270098931695 | 18.247303018014392 | 27.02327983720894 |
| q8 | 8 | linear | 0.06508594500594155 | 18.013422943966617 | 16.523376394314827 | 27.01806414475214 |
| q8 | 8 | adapter | 0.06508594500594155 | 18.480296729121694 | 17.068303922946175 | 27.01806414475214 |
| q8 | 16 | linear | 0.036980650571557694 | 16.465876780833767 | 15.543417464574004 | 26.997379801213054 |
| q8 | 16 | adapter | 0.036980650571557694 | 16.865547380070296 | 15.978911066116073 | 26.997379801213054 |
| q10 | 4 | linear | 0.1516173773213112 | 20.750844388443603 | 17.656157407040766 | 29.327984245556006 |
| q10 | 4 | adapter | 0.1516173773213112 | 21.191095902140322 | 18.25601918552348 | 29.327984245556006 |
| q10 | 8 | linear | 0.08135566587972795 | 18.356024730400843 | 16.54012288287085 | 29.320664241574747 |
| q10 | 8 | adapter | 0.08135566587972795 | 18.809757721047486 | 17.06974217895638 | 29.320664241574747 |
| q10 | 16 | linear | 0.046224810158936334 | 16.66055626234757 | 15.548468186693983 | 29.328475996349997 |
| q10 | 16 | adapter | 0.046224810158936334 | 17.052739413791674 | 15.97580514912888 | 29.328475996349997 |
| q12 | 4 | linear | 0.18193822076791313 | 20.84685363395712 | 17.660375119349375 | 29.678129037019875 |
| q12 | 4 | adapter | 0.18193822076791313 | 21.284133638556813 | 18.256196169477683 | 29.678129037019875 |
| q12 | 8 | linear | 0.09762538675351436 | 18.407949259402884 | 16.542993366151862 | 29.668274733935373 |
| q12 | 8 | adapter | 0.09762538675351436 | 18.85917872255155 | 17.06969395261803 | 29.668274733935373 |
| q12 | 16 | linear | 0.055468969746314975 | 16.690736197815284 | 15.550309668632122 | 29.679767860200425 |
| q12 | 16 | adapter | 0.055468969746314975 | 17.081329746423872 | 15.975915248513115 | 29.679767860200425 |

## Method Averages

| method | codec | mean rate | mean all PSNR | mean middle PSNR | mean given PSNR |
|---|---|---:|---:|---:|---:|
| linear | q8 | 0.0744543764840695 | 18.196689289875984 | 16.561587772918216 | 27.01290792772471 |
| linear | q10 | 0.09306595111999183 | 18.589141793730672 | 16.581582825535197 | 29.32570816116025 |
| linear | q12 | 0.11167752575591416 | 18.648513030391765 | 16.58455938471112 | 29.67539054371856 |
| adapter | q8 | 0.0744543764840695 | 18.639515032836314 | 17.098172669025544 | 27.01290792772471 |
| adapter | q10 | 0.09306595111999183 | 19.017864345659827 | 17.100522171202915 | 29.32570816116025 |
| adapter | q12 | 0.11167752575591416 | 19.074880702510743 | 17.100601790202944 | 29.67539054371856 |

## StreamSplat Reference Gap

| ref setting | anchor codec | anchor gap | method | anchor middle | ref middle | gap | note |
|---|---|---:|---|---:|---:|---:|---|
| Middle-4 frames | q8 | 4 | linear | 17.61796945986582 | 23.004337221027775 | -5.3863677611619565 | diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4 |
| Middle-4 frames | q8 | 4 | adapter | 18.247303018014392 | 23.004337221027775 | -4.757034203013383 | diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4 |
| 8-frame interval | q8 | 8 | linear | 16.523376394314827 | 21.56004909948801 | -5.036672705173181 | diagnostic: both are treated as local 8-frame interval references |
| 8-frame interval | q8 | 8 | adapter | 17.068303922946175 | 21.56004909948801 | -4.4917451765418335 | diagnostic: both are treated as local 8-frame interval references |
| Middle-4 frames | q10 | 4 | linear | 17.656157407040766 | 23.004337221027775 | -5.348179813987009 | diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4 |
| Middle-4 frames | q10 | 4 | adapter | 18.25601918552348 | 23.004337221027775 | -4.748318035504294 | diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4 |
| 8-frame interval | q10 | 8 | linear | 16.54012288287085 | 21.56004909948801 | -5.019926216617158 | diagnostic: both are treated as local 8-frame interval references |
| 8-frame interval | q10 | 8 | adapter | 17.06974217895638 | 21.56004909948801 | -4.49030692053163 | diagnostic: both are treated as local 8-frame interval references |
| Middle-4 frames | q12 | 4 | linear | 17.660375119349375 | 23.004337221027775 | -5.3439621016784 | diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4 |
| Middle-4 frames | q12 | 4 | adapter | 18.256196169477683 | 23.004337221027775 | -4.748141051550093 | diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4 |
| 8-frame interval | q12 | 8 | linear | 16.542993366151862 | 21.56004909948801 | -5.0170557333361465 | diagnostic: both are treated as local 8-frame interval references |
| 8-frame interval | q12 | 8 | adapter | 17.06969395261803 | 21.56004909948801 | -4.490355146869977 | diagnostic: both are treated as local 8-frame interval references |

## Outputs

- Summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_integrated_davis_rd_summary.json`
- Rate table: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv`
- PSNR table: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_psnr_table.csv`
- Method summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_method_summary.csv`
- Reference gap table: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_reference_gap_table.csv`
- All-frame RD plot: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_all_rd_curve.png`
- Middle-frame RD plot: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_middle_rd_curve.png`
