# Stage189 Failure-Case Analysis

Date: 2026-07-01

## Goal

Diagnose why Stage165/188 adaptive schedules improve quality over uniform gap8 but still do not reach gap8 rate, and identify concrete false-positive/false-negative failure cases.

## Plan

- Reuse Stage184 payload measurements, Stage186 quality rows, Stage187 labels, and Stage188 candidate schedules.
- Analyze promoted keyframes as potential false positives when local gain is small relative to local keyframe cost.
- Analyze unpromoted residual frames as false negatives when quality is weak or residual payload is high.
- Compare Stage188 candidates against full adaptive and gap8 to identify which dropped cells drive quality loss or rate savings.
- Produce CSV/JSON/Markdown summaries only; no new rendering or heavy artifacts.

## Success Criteria

- Tables identify the worst promoted-keyframe false-positive risks.
- Tables identify the worst unpromoted residual false-negative risks.
- Candidate-specific dropped-frame analysis explains the tradeoff between `interval_top10pct_cells`, `interval_score_ge4p0`, `interval_top90pct_cells`, and full Stage165 adaptive.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01; Stage189 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage189_failure_case_analysis.py`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage189_failure_case_analysis.py`

## Outputs

- Package: `experiments/stage189_failure_case_analysis/stage189_failure_case_analysis_package.json`
- Report: `experiments/stage189_failure_case_analysis/stage189_failure_case_analysis_report.md`
- Promoted-keyframe analysis: `experiments/stage189_failure_case_analysis/stage189_promoted_keyframe_false_positive_analysis.csv`
- Residual risk table: `experiments/stage189_failure_case_analysis/stage189_unpromoted_residual_false_negative_risks.csv`
- Candidate dropped-frame analysis: `experiments/stage189_failure_case_analysis/stage189_candidate_dropped_frame_loss_analysis.csv`
- Candidate summary: `experiments/stage189_failure_case_analysis/stage189_candidate_failure_summary.csv`
- Sequence summary: `experiments/stage189_failure_case_analysis/stage189_sequence_level_failure_summary.csv`
- Sequence hotspots: `experiments/stage189_failure_case_analysis/stage189_sequence_hotspots.csv`

## Results

Candidate summary:

| candidate | keyframes | MiB/frame | delta rate vs gap8 | delta rate vs full | PSNR | delta PSNR vs gap8 | LPIPS | changed frames vs full | worst changed delta PSNR vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `interval_top10pct_cells` | `299` | `0.2773746177516859` | `0.0014903267483045712` | `-0.01339097998630051` | `29.38112562842953` | `0.007160756589655648` | `0.16832458856830065` | `370` | `-5.309560997374348` |
| `interval_score_ge4p0` | `324` | `0.2829920490602662` | `0.00710775805688485` | `-0.007773548677720232` | `29.41013285788653` | `0.03616798604665661` | `0.16682702663534876` | `223` | `-2.3890793771766035` |
| `interval_top90pct_cells` | `353` | `0.289479501370253` | `0.013595210366871668` | `-0.0012860963677334136` | `29.424507356466457` | `0.05054248462658251` | `0.16601754864188598` | `35` | `-1.2776672922430699` |

Promoted keyframes:

- Promoted rows analyzed: `66`.
- Promoted rate-risk rows after refined criterion: `2`.
- Rate-risk rows: `drift-chicane` frame `6`, `horsejump-high` frame `15`; both have small unlabeled gains and large local payload deltas.

Residual risks:

- Residual risk rows: `1179`.
- Top residual risks by nonnegative score are high-LPIPS/high-payload cases in `motocross-jump`, `india`, and `mbike-trick`, followed by low-PSNR/high-payload `cows` cases.
- Sequence hotspots by residual-risk count: `cows` `86`, `parkour` `75`, `camel` `73`, `goat` `73`, `breakdance` `72`, `soapbox` `72`, `bmx-trees` `67`.

## Decision

- Decision: `failure_cases_identified_for_paper_and_next_selector_refinement`.
- The Stage188 lowest-rate candidate works by dropping many cells, but it changes `370` frames vs full adaptive and loses up to `-5.309560997374348` dB on changed frames.
- The balanced `interval_score_ge4p0` point changes fewer frames (`223`) but still loses up to `-2.3890793771766035` dB on changed frames.
- Rate overhead is not mostly caused by obvious bad promotions: only `2/66` promoted keyframes are strong rate-risk rows after refined criteria.
- Remaining failure modes are unpromoted residual risks, especially low-PSNR `cows`/`breakdance`/`camel` frames and high-LPIPS/high-payload `motocross-jump`/`india` frames.

## Next

- Stage190 should package the measured RD-quality, Stage188 sensitivity, and Stage189 failure cases into paper-facing tables and limitations.
