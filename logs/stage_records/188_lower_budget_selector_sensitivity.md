# Stage188 Lower-Budget Selector Sensitivity

Date: 2026-07-01

## Goal

Evaluate lower-budget adaptive keyframe selector variants after Stage185/186 showed the frozen Stage165 adaptive schedule is a measured middle RD point, not lower-rate than uniform gap8.

## Plan

- Reuse Stage184 payload measurements and Stage186 quality measurements whenever schedule coverage is complete.
- Separate row-level feature ablations from fully measured schedule candidates.
- For fully reusable candidates, use gap8 interval-level schedules: each gap8 interval either keeps all Stage165 adaptive extra keyframes or falls back completely to uniform gap8.
- Compute measured additive RD with single-anchor q12 keyframe bytes, measured Stage158 residual bytes, and exact schedule metadata for every candidate and baseline.
- Report quality with PSNR, SSIM, MS-SSIM, and LPIPS.
- Do not compare new candidates against Stage185 packed-keyframe rates as if they were the same rate scope.

## Success Criteria

- At least one lower-budget candidate has complete measured payload/quality coverage without new rendering.
- The report identifies whether any lower-budget candidate reduces the full Stage165 additive rate while preserving positive quality delta over gap8.
- Row-level Stage187 ablations are either fully covered or explicitly marked as requiring new measurement.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01; Stage188 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage188_lower_budget_selector_sensitivity.py`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage188_lower_budget_selector_sensitivity.py`

## Outputs

- Package: `experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_package.json`
- Report: `experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_report.md`
- RD-quality CSV: `experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_rd_quality.csv`
- Interval ranking CSV: `experiments/stage188_lower_budget_selector_sensitivity/stage188_interval_cell_ranking.csv`
- Row-level coverage audit CSV: `experiments/stage188_lower_budget_selector_sensitivity/stage188_row_level_ablation_coverage_audit.csv`
- Validation CSV: `experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_validation.csv`

## Results

Rate scope: `measured_single_anchor_additive_keyframes_plus_measured_stage158_residuals_plus_exact_metadata`. This is an apples-to-apples sensitivity scope and must not be mixed numerically with Stage185 schedule-packed keyframe rates.

Fully covered candidates: `23`. Interval cells: `59`. Row-level ablation audits: `10/10` fully covered after duplicate target collapse.

Baseline additive RD-quality:

| schedule | keyframes | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | `292` | `0.27588429100338135` | `29.373964871839874` | `0.867625699572828` | `0.9843430183660156` | `0.16869177970254404` |
| Stage165 adaptive full | `358` | `0.29076559773798644` | `29.425582692060658` | `0.8692941793565335` | `0.9846469353830415` | `0.16593745923142186` |
| uniform_gap4 | `536` | `0.3308038587508171` | `29.535715839048738` | `0.8739438994697716` | `0.9855294218654929` | `0.15947172297849663` |

Recommended lower-budget points:

| candidate | keyframes | cells kept | MiB/frame | delta vs gap8 | delta vs full | PSNR | delta PSNR vs gap8 | LPIPS | delta LPIPS vs gap8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `interval_top10pct_cells` | `299` | `6` | `0.2773746177516859` | `+0.0014903267483045712` | `-0.01339097998630051` | `29.38112562842953` | `+0.007160756589655648` | `0.16832458856830065` | `-0.0003671911342433831` |
| `interval_score_ge4p0` | `324` | `27` | `0.2829920490602662` | `+0.00710775805688485` | `-0.007773548677720232` | `29.41013285788653` | `+0.03616798604665661` | `0.16682702663534876` | `-0.001864753067195274` |
| `interval_top90pct_cells` | `353` | `54` | `0.289479501370253` | `+0.013595210366871668` | `-0.0012860963677334136` | `29.424507356466457` | `+0.05054248462658251` | `0.16601754864188598` | `-0.002674231060658061` |

## Decision

- Decision: `lower_budget_positive_quality_candidates_found_but_gap8_rate_not_reached`.
- Stage188 found fully covered lower-budget candidates that improve all measured quality metrics over gap8 and reduce rate versus full Stage165 adaptive under the additive sensitivity scope.
- The lowest-rate positive candidate `interval_top10pct_cells` reduces most of the full adaptive overhead but still remains `+0.0014903267483045712` MiB/frame above gap8 additive rate.
- A more balanced point is `interval_score_ge4p0`: it keeps `324` keyframes, preserves `+0.03616798604665661` dB PSNR over gap8, improves LPIPS by `-0.001864753067195274`, and cuts full adaptive overhead roughly in half.
- No Stage188 candidate should be reported as measured schedule-packed RD unless its schedule-packed keyframe bitstreams are measured separately.

## Next

- Stage189 should analyze false positives/false negatives and candidate-specific failure cases, focusing on why gap8-rate parity is not reached.
- Stage190 should use Stage188 as a sensitivity/ablation result and keep Stage185/186 as the measured packed-keyframe baseline/frozen-candidate result.
