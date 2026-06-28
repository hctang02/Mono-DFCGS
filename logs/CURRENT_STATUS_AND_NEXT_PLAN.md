# Current Status And Next Plan

Date: 2026-06-29

## Current Task

Continue the Mono-DFCGS self-innovation line from teacher residual side-info toward a more deployable residual selector / residual side-info codec.

The current focus is not FCGS/D-FCGS comparison and not residual value prediction yet. The current focus is to make residual side-info index selection and switching more decoder-safe and render-aware.

## Current Repo State

- Repo: `/mnt/hdd2tC/haocheng/Mono-DFCGS`
- Remote: `git@github.com:hctang02/Mono-DFCGS.git`
- Python env: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv`
- Latest pushed commit before Stage112: `c808413 Evaluate broader switch predictor`
- Canonical continuation file: `logs/CURRENT_STATUS_AND_NEXT_PLAN.md`
- Current best adapter checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors`
- Main DAVIS root: `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS`
- Heavy anchor root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full`

## Important Constraints

- Target dense anchors may be used only for train/encoder-side labels or offline diagnostics, never as decoder-side input.
- Every transmitted side-info byte must be counted in RD.
- Current Stage99-108 rendered selector tests still use teacher residual values at selected indices, so they are not a full deployable residual-value codec.
- Do not commit heavy tensors, `.pt` anchors, checkpoints, datasets, cache, or large generated artifacts.
- Continue to leave unrelated dirty files untouched unless explicitly asked.
- Check `nvidia-smi` before running code; use an idle GPU when GPU work is needed.
- User confirmed on 2026-06-28 that future stages should continue from this file and keep this filename.

## Recently Completed

### Residual Side-Info Codec / RD

- Stage87-90: q6 top10 residual side-info rendered/RD eval.
- Stage91: fixed residual bitstream smoke, payload decode-equivalent to tensor construction.
- Stage92-96: entropy-coded residual side-info and broader RD package.
- Preferred current entropy RD package: `experiments/stage96_broader_entropy_residual_sideinfo_rd_package/`

Key Stage96 direct total rates:

| base | gap4 | gap8 | gap16 |
|---|---:|---:|---:|
| linear | 0.21042431365324843 | 0.12665608623246688 | 0.08430766643343285 |
| stage65_adapter | 0.21508592669374865 | 0.13149337333220473 | 0.08923517723400179 |

### Residual Predictor Data Entry

- Stage97: residual predictor task manifest, `15554` q12 train/eval tasks.
- Stage98: residual importance predictor smoke using decoder-available features.
- Stage98 showed learned selector improves residual energy recall over endpoint-diff baseline, but remains well below oracle top10.

### Rendered Selector / Switch Line

- Stage99: rendered predicted-index smoke with teacher residual values; learned selector gives positive PSNR but remains `1.34-2.51 dB` below teacher oracle.
- Stage100: objective sweep; learned objectives all beat endpoint in energy recall, but objective changes did not solve oracle gap.
- Stage101: endpoint/gap/rank feature sweep; hand-crafted extras did not materially beat Stage100 base features.
- Stage102: group-specific heads; did not beat shared selector.
- Stage103: broader rendered selector validation; learned selector helps linear base but hurts Stage65 adapter base.
- Stage104: mismatch diagnostic; energy recall improvements often produce PSNR drops, especially on Stage65 adapter.
- Stage105: render-aware switching preflight; simple group switch gives `+0.030059636844502392 dB` over endpoint-only.
- Stage106: packaged decoder-side metadata group switch policy.
- Stage107: metadata-only task-level switch predictor; did not beat endpoint or Stage106.
- Stage108: anchor-stat task-level switch predictor; better than metadata-only and endpoint, but still below Stage106 and overfits.
- Stage109: selector-score task-level switch predictor; score statistics have signal but still do not beat Stage106.
- Stage110: broader rendered selector labels with 240 eval tasks; Stage106 fixed policy remains slightly positive but much weaker, and a broader group-best pattern changes linear gap4 back to endpoint.
- Stage111: broader switch predictor on Stage110 labels; score-stat MLP beats fixed group policies overall but still has Stage65 adapter gap4 regression.
- Stage112: packaged conservative broader metadata group switch policy `render_aware_group_switch_v2`; it uses only `base_method` and `reference_gap` and selects endpoint for linear gap4 plus all Stage65 adapter groups.

## Current Best Selector Policy

Current packaged candidate for held-out validation: Stage112 `render_aware_group_switch_v2`.

Previous safe baseline: Stage106 `render_aware_group_switch_v1`.

Policy JSON:

```text
experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy.json
```

Selection table:

| base | gap | selected candidate |
|---|---:|---|
| linear | 4 | endpoint_diff_baseline |
| linear | 8 | shared_energy_regression |
| linear | 16 | shared_energy_regression |
| stage65_adapter | 4 | endpoint_diff_baseline |
| stage65_adapter | 8 | endpoint_diff_baseline |
| stage65_adapter | 16 | endpoint_diff_baseline |

Validation summary on Stage110 broader rows:

| metric | value |
|---|---:|
| task count | 480 |
| endpoint PSNR | 20.3212149854921 |
| group policy PSNR | 20.327046871072337 |
| gain vs endpoint | 0.005831885580240304 |
| oracle task best PSNR | 20.382843220952523 |
| teacher oracle top10 PSNR | 22.077800340877268 |

## Current Interpretation

- Residual-energy topk is useful but not render-aligned enough.
- Always using learned selection is unsafe because Stage65 adapter groups degrade.
- Metadata-only task-level switching is too weak.
- Anchor-stat task-level switching has signal but overfits on the current 120 rendered-label tasks.
- Selector-score task-level switching has signal but does not fix adapter-group regressions and remains below Stage106.
- Stage112 is the current packaged conservative selector-switch candidate, but it still needs Stage113 held-out validation before being treated as final.
- Stage106 remains the previous packaged baseline and should remain in comparisons.
- Stage110 group-best pattern has been frozen into Stage112 v2 for validation.
- Stage111 learned switch is not safe enough to package because adapter gap4 still regresses.
- Residual value prediction should wait until selector switching and index/value accounting are more stable.

## Next Plan

### Stage109: Selector-Score Feature Preflight

Status: completed on 2026-06-28.

Result:

- `score_stat_mlp_cv` PSNR `20.32781855154445`, gain vs endpoint `+0.01100584121880117 dB`.
- `anchor_score_mlp_cv` PSNR `20.328372726184103`, gain `+0.01156001585845603 dB`.
- `anchor_stat_mlp_cv` remains `20.33017523703834`, gain `+0.013362526712690951 dB`.
- Stage106 fixed group policy remains better at `20.34687234717015`, gain `+0.030059636844502392 dB`.
- Conclusion: do not replace Stage106; proceed to Stage110 broader rendered selector labels.

Goal: test whether selector-score statistics can improve task-level switching beyond Stage106.

Actions:

- Build decoder-side score features from endpoint scores and learned selector logits.
- Include score margin, entropy/spread, topk overlap between endpoint and learned selector, and disagreement counts.
- Do not load target dense anchors as input.
- Use Stage103 rendered best candidate only as train/eval label.
- Compare against endpoint-only, Stage106 fixed group policy, metadata MLP, anchor-stat MLP, and oracle task best.

Success condition:

- Out-of-fold task switch policy exceeds Stage106 `+0.030059636844502392 dB` without hurting Stage65 adapter groups.

Fallback:

- If it does not beat Stage106, keep Stage106 fixed group policy and stop optimizing switch predictors until more rendered labels are available.

### Stage110: Broader Rendered Selector Labels

Status: completed on 2026-06-28.

Result:

- Eval tasks increased from `60` to `240`; policy task count is `480` across two base methods.
- Endpoint-only PSNR: `20.3212149854921`.
- Stage106 fixed group policy PSNR: `20.322996715243953`, gain `+0.0017817297518578745 dB`.
- Stage110 group-best policy PSNR: `20.327046871072337`, gain `+0.005831885580240304 dB`.
- Oracle task best PSNR: `20.382843220952523`, gain `+0.06162823546041816 dB`.
- Broader group-best choices: linear gap4 -> endpoint; linear gap8/16 -> shared_energy_regression; Stage65 adapter gap4/8/16 -> endpoint.
- Conclusion: Stage106 exact policy remains slightly positive but is much less robust than on Stage105; use Stage110 rows for Stage111 broader switch predictor.

Goal: reduce overfitting from the 120-task rendered-label set.

Actions:

- Extend Stage103 rendered validation from 60 eval tasks to a larger sample, such as 240 or 600 eval tasks if runtime permits.
- Keep outputs small: CSV, JSON, Markdown report only.
- Recompute Stage104/105 policy diagnostics on the broader rows.

Success condition:

- Stage106-style policy remains positive on broader rendered labels, or a better switch policy emerges consistently.

### Stage111: Broader Switch Predictor

Status: completed on 2026-06-28.

Result:

- Task count: `480` broader switch rows from Stage110.
- `score_stat_mlp_cv` PSNR `20.33325220653739`, gain vs endpoint `+0.012037221045259486 dB`.
- Stage110 group-best policy PSNR `20.32704687107235`, gain `+0.005831885580240304 dB`.
- Stage106 fixed policy PSNR `20.322996715243978`, gain `+0.0017817297518578745 dB`.
- Oracle task best PSNR `20.382843220952545`, gain `+0.06162823546041816 dB`.
- `score_stat_mlp_cv` still has Stage65 adapter gap4 regression `-0.00797889356792674 dB`.
- Conclusion: do not package learned switch; package conservative Stage110 group policy candidate in Stage112.

Goal: train a more reliable switch predictor using metadata + anchor stats + selector-score features on broader labels.

Actions:

- Use sequence-aware or task-id-aware folds.
- Compare against Stage106 group policy and train-fold group policy.
- Do not save model checkpoint unless the policy clearly beats Stage106.

Success condition:

- Stable out-of-fold PSNR gain over Stage106 and no adapter-group regression.

### Stage112: Package Best Switch Policy

Status: completed on 2026-06-29.

Result:

- Packaged `render_aware_group_switch_v2` in `experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy.json`.
- Decoder inputs: `base_method`, `reference_gap`.
- Forbidden decoder inputs: `target_dense_anchor`, `target_residual`, `rendered_psnr`, `oracle_task_label`, `target_rgb`.
- Stage110 broader validation: endpoint PSNR `20.3212149854921`, policy PSNR `20.327046871072337`, gain `+0.005831885580240304 dB`.
- Stage111 `score_stat_mlp_cv` is not packaged despite higher overall gain because Stage65 adapter gap4 still regresses.

Goal: freeze the selector-switching rule.

Actions:

- If learned switch wins, package it with explicit decoder inputs and forbidden inputs.
- If learned switch does not win, keep Stage106 policy as the final selector-switch baseline.
- Document validation split and limitations.

### Stage113: Held-Out Switch Validation

Goal: validate the selected switch policy on held-out sequences or a broader eval set.

Actions:

- Re-run rendered selector validation with the frozen switch policy.
- Compare endpoint-only, learned-only, group switch, task switch, and oracle task switch.

## Later Plan After Selector Stabilizes

### Deterministic-Index Side-Info Codec

- Stage114: build deterministic-index residual codec where decoder predicts indices and bitstream carries values/scales only.
- Stage115: compare index+value entropy side-info vs deterministic-index value-only side-info.
- Stage116: sweep q-bit and keep fraction.
- Stage117: package broader RD with all side-info bytes counted.

### Residual Value Prediction

- Stage118: build selected-index residual value predictor manifest.
- Stage119: non-rendered predicted-value smoke.
- Stage120: rendered predicted-value smoke without teacher residual values.
- Stage121: hybrid low-rate correction if pure prediction is too weak.
- Stage122: package complete residual codec fields and RD accounting.

### Final Evaluation

- Stage123: broaden training labels.
- Stage124: final selected policy held-out eval.
- Stage125: integrated RD table.
- Stage126: ablations.
- Stage127: final method docs.
- Stage128: only after internal line stabilizes, resume FCGS/D-FCGS fair comparison.

## Key Reference Files

### Logs

- Request log: `logs/USER_REQUESTS.md`
- Execution log: `logs/STAGE_EXECUTION_LOG.md`
- Pitfalls: `logs/PITFALLS.md`
- Stage records: `logs/stage_records/`
- Current summary: `logs/CURRENT_STATUS_AND_NEXT_PLAN.md`

### Core Code

- Anchor codec utilities: `mono_dfcgs/gaussian_codec.py`
- Residual side-info codec: `mono_dfcgs/residual_sideinfo_codec.py`
- Anchor predictor: `mono_dfcgs/anchor_predictor.py`
- Render wrapper: `mono_dfcgs/render_adapter.py`

### Important Scripts

- Stage97 manifest: `scripts/run_stage97_residual_predictor_task_manifest.py`
- Stage98 selector smoke: `scripts/run_stage98_residual_importance_predictor_smoke.py`
- Stage99 rendered selector smoke: `scripts/run_stage99_predictor_selected_sideinfo_render_smoke.py`
- Stage100 objective sweep: `scripts/run_stage100_residual_selector_objective_sweep.py`
- Stage101 feature sweep: `scripts/run_stage101_enhanced_selector_feature_sweep.py`
- Stage102 group-specific heads: `scripts/run_stage102_group_specific_selector_heads.py`
- Stage103 broader rendered validation: `scripts/run_stage103_broader_rendered_selector_validation.py`
- Stage104 mismatch diagnostic: `scripts/run_stage104_render_energy_selector_mismatch_diagnostic.py`
- Stage105 policy preflight: `scripts/run_stage105_render_aware_selector_policy_preflight.py`
- Stage106 policy package: `scripts/run_stage106_render_aware_group_policy_package.py`
- Stage107 metadata switch predictor: `scripts/run_stage107_metadata_task_switch_predictor_preflight.py`
- Stage108 anchor-stat switch predictor: `scripts/run_stage108_anchor_stat_task_switch_predictor_preflight.py`
- Stage109/111 selector-score switch predictor: `scripts/run_stage109_selector_score_switch_feature_preflight.py`
- Stage112 policy package: `scripts/run_stage112_package_broader_group_switch_policy.py`

### Important Outputs

- Stage96 broader entropy RD: `experiments/stage96_broader_entropy_residual_sideinfo_rd_package/`
- Stage97 task manifest: `experiments/stage97_residual_predictor_task_manifest/`
- Stage103 rendered selector rows: `experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv`
- Stage104 mismatch diagnostic: `experiments/stage104_render_energy_selector_mismatch_diagnostic/`
- Stage105 render-aware policy preflight: `experiments/stage105_render_aware_selector_policy_preflight/`
- Stage106 packaged policy: `experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy.json`
- Stage107 metadata switch predictor: `experiments/stage107_metadata_task_switch_predictor_preflight/`
- Stage108 anchor-stat switch predictor: `experiments/stage108_anchor_stat_task_switch_predictor_preflight/`
- Stage109 selector-score switch predictor: `experiments/stage109_selector_score_switch_feature_preflight/`
- Stage110 broader rendered labels: `experiments/stage110_broader_rendered_selector_labels/`
- Stage111 broader switch predictor: `experiments/stage111_broader_switch_predictor/`
- Stage112 packaged policy: `experiments/stage112_broader_group_switch_policy_package/`

### Heavy External Paths

- Stage61 dense anchors: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full`
- Stage65 adapter: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors`
- DAVIS dataset: `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS`
- Python env: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv`

## Files To Avoid Unless Asked

These existing dirty/untracked items are unrelated to this line and should remain untouched unless explicitly requested:

- `scripts/run_stage53_baseline_comparison_scaffold.py`
- `experiments/exp_stage_streamsplat_comparison/`
- `experiments/stage53_main_vs_fcgs_dfcgs/`
- `scripts/run_exp_stage02_ours_center_square_252.py`
- `scripts/run_exp_stage03_fcgs_anchor_adapter_smoke.py`
- `scripts/run_exp_stage04_fcgs_decoded_center_square_smoke.py`
- `scripts/run_exp_stage05_fcgs_perframe_representatives.py`
- `scripts/run_exp_stage_streamsplat_comparison.py`
