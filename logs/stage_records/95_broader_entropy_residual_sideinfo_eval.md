# Stage95 Broader Entropy-Coded Residual Side-Info Eval

Date: 2026-06-28

## Goal

Validate entropy codec v2 on the broader 60-task q12 eval slice.

## Implementation

Updated:

```text
scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py
```

The script now supports stage number, mode, output prefix, and report title parameters. Defaults preserve Stage93 output naming.

## Run

GPU check was performed before execution. GPU0 was busy and GPU1 was idle, so the run used:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage93_residual_sideinfo_entropy_codec_smoke.py --stage 95 --mode "broader entropy-coded residual side-info codec eval" --summary_root experiments/stage95_broader_entropy_residual_sideinfo_eval --output_prefix stage95_broader_entropy_residual_sideinfo_eval --report_title "Stage95 Broader Entropy-Coded Residual Side-Info Eval" --max_tasks 60
```

## Outputs

```text
experiments/stage95_broader_entropy_residual_sideinfo_eval/stage95_broader_entropy_residual_sideinfo_eval_rows.csv
experiments/stage95_broader_entropy_residual_sideinfo_eval/stage95_broader_entropy_residual_sideinfo_eval_summary.csv
experiments/stage95_broader_entropy_residual_sideinfo_eval/stage95_broader_entropy_residual_sideinfo_eval_summary.json
experiments/stage95_broader_entropy_residual_sideinfo_eval/stage95_broader_entropy_residual_sideinfo_eval_report.md
```

## Results

| base | gap | tasks | entropy MiB/intermediate | ratio vs fixed | max decode diff | delta PSNR | positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 18 | 0.028486092885335285 | 0.6885464450642754 | 0.0 | 3.4187356956292225 | 18 |
| linear | 8 | 19 | 0.02903069947895251 | 0.7017103048994284 | 0.0 | 3.05603754877287 | 19 |
| linear | 16 | 23 | 0.028838696687117867 | 0.6970693441228029 | 0.0 | 3.2599322732336695 | 23 |
| stage65_adapter | 4 | 18 | 0.033147705925835505 | 0.8012237820448788 | 0.0 | 2.7994188265782376 | 18 |
| stage65_adapter | 8 | 19 | 0.033867986578690376 | 0.8186339156482526 | 0.0 | 2.692464222747122 | 19 |
| stage65_adapter | 16 | 23 | 0.03376620748768682 | 0.8161737807475322 | 0.0 | 2.9637922008291264 | 23 |

## Conclusion

- Entropy codec v2 remains decode-equivalent to fixed q6 side-info on the 60-task slice.
- All tasks remain positive after entropy decode and rendering.
- The next step is broader entropy RD accounting using these actual side-info rates.
