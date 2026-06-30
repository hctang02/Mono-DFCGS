# Stage187 Selector Feature Ablation Validation

Date: 2026-07-01

## Goal

Validate which encoder-side RGB/motion feature groups support the Stage165 adaptive keyframe selector.

## Plan

- Reuse the Stage163/165 labeled selector rows.
- Keep the Stage165 selected gate fixed: rank threshold `0.65`, min votes `1`.
- Evaluate feature ablations and feature subsets at the selector-label level.
- Report hard-quality and high-payload precision/recall/F1, selected count, resulting keyframe count, and metadata bytes.
- Do not claim full measured RD for ablation schedules unless every schedule row has measured payload/quality coverage.

## Success Criteria

- A feature-ablation table identifies which feature groups are necessary for recall/precision.
- The report explicitly separates label/protocol ablation from Stage185/186 full measured RD.
- Candidate lower-budget variants are identified for Stage188 sensitivity.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01; GPU1 was idle, though Stage187 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage187_selector_feature_ablation_validation.py`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage187_selector_feature_ablation_validation.py`

## Outputs

- Package: `experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_validation_package.json`
- Report: `experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_validation_report.md`
- Summary CSV: `experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_summary.csv`
- Row dump CSV: `experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_rows.csv`
- Context CSV: `experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_context.csv`

## Results

Stage187 is a selector-label/protocol ablation over Stage163 rows. It does not claim measured full-sequence RD for ablation schedules.

Fixed gate: rank threshold `0.65`, minimum votes `1`.

Current full Stage165 five-feature gate:

| metric | value |
|---|---:|
| selected rows | `70` |
| adaptive keyframes | `358` |
| extra keyframes vs gap8 | `66` |
| metadata bytes | `327` |
| hard-quality recall | `0.7333333333333333` |
| high-payload recall | `0.8194444444444444` |

Feature-ablation highlights:

| variant | selected | keyframes | hard recall | payload recall | delta selected vs full |
|---|---:|---:|---:|---:|---:|
| `drop_interp_rgb` | `69` | `357` | `0.7333333333333333` | `0.8055555555555556` | `-1` |
| `motion_proxy_edge_hist` | `68` | `357` | `0.7333333333333333` | `0.7916666666666666` | `-2` |
| `edge_hist_only` | `67` | `356` | `0.7333333333333333` | `0.7777777777777778` | `-3` |
| `drop_hist_motion` | `61` | `349` | `0.7333333333333333` | `0.7361111111111112` | `-9` |
| `drop_edge_motion` | `60` | `348` | `0.43333333333333335` | `0.75` | `-10` |
| `proxy_only` | `42` | `331` | `0.4` | `0.5416666666666666` | `-28` |
| `rgb_only` | `48` | `336` | `0.4` | `0.6388888888888888` | `-22` |

## Decision

- Decision: `feature_ablation_ready_for_budget_sensitivity`.
- The five-feature Stage165 gate remains the highest-recall selector among evaluated variants.
- `drop_interp_rgb` is the conservative Stage188 low-budget candidate because it keeps hard-quality recall unchanged and loses only `0.01388888888888884` high-payload recall.
- Stage188 should also test more aggressive low-budget variants, especially `drop_hist_motion`, because it removes `9` selected rows while preserving hard-quality recall, but it lowers high-payload recall to `0.7361111111111112`.

## Next

- Stage188 should evaluate lower-budget selector sensitivity as explicit schedule/RD points, reusing Stage184/186 measured rows wherever schedule coverage allows.
- Stage188 should not rely on Stage187 label metrics alone for final rate/quality claims.
