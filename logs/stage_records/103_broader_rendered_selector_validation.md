# Stage103 Broader Rendered Selector Validation

Date: 2026-06-28

## Goal

Validate whether Stage100-102 shared learned selector improvements in offline energy recall translate into rendered RGB PSNR.

## Implementation

Added:

```text
scripts/run_stage103_broader_rendered_selector_validation.py
```

The script trains shared `topk_bce` and `energy_regression` selectors, then renders q6 selected residual side-info for `60` eval tasks. Predicted indices still receive teacher residual values, so this stage isolates selection error and is not a deployable residual-value codec.

## Run

GPU check was performed before execution. GPU0 was busy, GPU1 had a small existing Python process, and GPU2 was idle. Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage103_broader_rendered_selector_validation.py
```

Full run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage103_broader_rendered_selector_validation.py
```

## Outputs

```text
experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv
experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_summary.csv
experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_train_log.csv
experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_summary.json
experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_report.md
```

## Configuration

| item | value |
|---|---:|
| train tasks | 96 |
| eval tasks | 60 |
| train examples | 589824 |
| keep fraction | 0.1 |
| side bits | 6 |
| train steps/model | 300 |

## Results

| base | gap | candidate | endpoint PSNR | learned PSNR | teacher PSNR | learned delta | gap to teacher |
|---|---:|---|---:|---:|---:|---:|---:|
| linear | 4 | shared_energy_regression | 20.950348386765448 | 20.977176264225726 | 22.564021045406168 | 1.3280279007214773 | -1.586844781180443 |
| linear | 8 | shared_energy_regression | 20.891092240107927 | 20.921955460500982 | 22.828205501612626 | 1.511430236827558 | -1.9062500411116432 |
| linear | 16 | shared_energy_regression | 18.68835971009511 | 18.821899379666544 | 20.772060171450324 | 1.2486889254734588 | -1.9501607917837827 |
| stage65_adapter | 4 | shared_energy_regression | 21.16056131878281 | 21.006753486754278 | 22.478052503501885 | 1.0415692337164726 | -1.4712990167476079 |
| stage65_adapter | 8 | shared_energy_regression | 20.97013749190139 | 20.778146373867948 | 22.491323232804557 | 1.1403157927234944 | -1.7131768589366099 |
| stage65_adapter | 16 | shared_energy_regression | 18.76182012897664 | 18.7456227991363 | 20.571596638113725 | 1.0267017588076504 | -1.8259738389774265 |

## Conclusion

- Shared learned selector improves rendered PSNR over endpoint-difference for the linear base.
- Shared learned selector underperforms endpoint-difference for the Stage65 adapter base despite better offline energy recall.
- Offline residual-energy recall is not sufficient as a selector objective for rendered quality.
- Next steps should diagnose render-aware selector labels or losses before residual value prediction.
