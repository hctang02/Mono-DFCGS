# Stage89 Broader q6 Residual Side-Info Eval

Date: 2026-06-28

## Goal

Validate the Stage88 low-rate q6 top10% residual side-info point on a larger eval slice.

## Configuration

- Tasks: `60` q12 eval tasks.
- Gaps: `4`, `8`, `16`.
- Base methods: linear and Stage65 adapter.
- Keep fraction: `0.1`.
- Side bits: `6`.
- Side-info rate: `0.041353702545166016 MiB/intermediate-frame`.

## Implementation

Updated:

```text
scripts/run_stage87_quantized_residual_sideinfo_smoke.py
```

The script now supports stage number, mode, output prefix, summary prefix, and report title parameters. Defaults preserve the Stage87 output naming.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle, so the successful run used:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage87_quantized_residual_sideinfo_smoke.py --stage 89 --mode "broader q6 quantized rendered residual side-info eval" --summary_root experiments/stage89_broader_q6_residual_sideinfo_eval --output_prefix stage89_broader_q6_residual_sideinfo_eval --summary_prefix stage89_broader_q6_residual_sideinfo_eval --report_title "Stage89 Broader q6 Residual Side-Info Eval" --max_tasks 60 --keep_fractions 0.1 --side_bits 6
```

An initial attempt with system `python` failed before rendering because `torch` was not installed in that interpreter.

## Outputs

```text
experiments/stage89_broader_q6_residual_sideinfo_eval/stage89_broader_q6_residual_sideinfo_eval_rows.csv
experiments/stage89_broader_q6_residual_sideinfo_eval/stage89_broader_q6_residual_sideinfo_eval_summary.csv
experiments/stage89_broader_q6_residual_sideinfo_eval/stage89_broader_q6_residual_sideinfo_eval_summary.json
experiments/stage89_broader_q6_residual_sideinfo_eval/stage89_broader_q6_residual_sideinfo_eval_report.md
```

## Results

| base | gap | tasks | base PSNR | side PSNR | delta PSNR | positives |
|---|---:|---:|---:|---:|---:|---:|
| linear | 4 | 18 | 19.984796422852106 | 23.403666365261454 | 3.4188699424093465 | 18 |
| linear | 8 | 19 | 18.457589807006293 | 21.513744759131683 | 3.056154952125395 | 19 |
| linear | 16 | 23 | 17.08421543617396 | 20.344221350298344 | 3.2600059141243887 | 23 |
| stage65_adapter | 4 | 18 | 20.041732308843873 | 22.84099865483423 | 2.7992663459903557 | 18 |
| stage65_adapter | 8 | 19 | 18.706547218120296 | 21.398942062762643 | 2.6923948446423474 | 19 |
| stage65_adapter | 16 | 23 | 17.32823013943833 | 20.29200458015274 | 2.9637744407144093 | 23 |

## Conclusion

- The q6 top10% residual side-info point remains positive for every evaluated task in this 60-task slice.
- Gains remain multi-dB for both linear and Stage65 adapter bases.
- This is still teacher residual side-info, not a final entropy-coded bitstream or deployable residual predictor.
