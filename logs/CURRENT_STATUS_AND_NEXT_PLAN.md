# Current Status And Next Plan

Date: 2026-06-29

## Current Task

Continue the Mono-DFCGS self-innovation line from teacher residual side-info toward a more deployable residual selector / residual side-info codec.

The current focus is not FCGS/D-FCGS comparison and not residual value prediction yet. The current focus is to make residual side-info index selection and switching more decoder-safe and render-aware.

## Current Repo State

- Repo: `/mnt/hdd2tC/haocheng/Mono-DFCGS`
- Remote: `git@github.com:hctang02/Mono-DFCGS.git`
- Python env: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv`
- Latest pushed commit before Stage136: `cc153f8 Package render-aware predictor protocol`
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
- Stage113: held-out switch diagnostic over Stage111 out-of-fold rows; Stage112 is aggregate group-safe but not fold-group safe under a zero-regression criterion.
- Stage114: packaged strict-safe endpoint-only selector fallback `strict_safe_endpoint_selector_v1` after user chose strict safety.
- Stage115: deterministic-index residual side-info codec smoke; value-only payload removes endpoint-diff selected index bytes and decodes identically to fixed index+value payload.
- Stage116: deterministic vs entropy side-info accounting; all transmitted bytes are counted, and deterministic endpoint-diff quality is explicitly marked rate-only / not rendered.
- Stage117: deterministic q-bit / keep-fraction side-info sweep; lower-rate settings can beat the Stage96 q6 top10 entropy reference in rate, but quality is unknown until rendered validation.
- Stage118: compressed deterministic value-only codec smoke; q6/top10 compressed deterministic payload decodes identically and beats Stage96 entropy-coded index+value side-info reference in all groups.
- Stage119: actual compressed deterministic q-bit / keep-fraction sweep over real task residual values; all rows decode exactly and shortlist is ready for rendered smoke.
- Stage120: rendered compressed deterministic shortlist smoke; q4/top20 is best on 12-task smoke and q4/top10 is a strong low-rate candidate.
- Stage121: 60-task broader rendered compressed deterministic validation confirms q4/top20 and q4/top10 are stable candidates.
- Stage122: packaged compressed deterministic RD rows/points; q4/top20 is primary, q4/top10 low-rate, q5/top10 near-anchor, q6/top10 anchor.
- Stage123: packaged codec policy manifest around strict-safe endpoint selector and compressed deterministic value-only side-info.
- Stage124: feed-forward residual value predictor smoke using Stage65 adapter delta over linear base completed.
- Stage125: broadened no-teacher feed-forward residual value predictor validation from 12 to 60 eval tasks.
- Stage126: packaged selected residual value predictor dataset manifest and normalization stats.
- Stage127: trained dedicated selected residual value predictor smoke with checkpoints saved outside git.
- Stage128: packaged predictor-only codec integration manifest.
- Stage129: broader rendered validation of predictor-only integrated codec completed; MSE-trained MLP regresses rendered PSNR.
- Stage130: compared teacher side-info, adapter-delta predictor, and dedicated MLP predictor RD points.
- Stage131: ablated keep fraction, predictor source, validation scale, and MSE/render mismatch.
- Stage132: packaged current deployable no-teacher policy.
- Stage133: completed final RD report and plots for teacher reference vs predictor-only policies.
- Stage134: diagnosed why the dedicated MLP residual predictor regresses rendered PSNR.
- Stage135: packaged render-aware adapter-delta calibration protocol.
- Stage136: completed render-aware adapter-delta scale sweep smoke.

## Current Best Selector Policy

Current strict-safe selector: Stage114 `strict_safe_endpoint_selector_v1`.

Aggregate-safe but not final candidate: Stage112 `render_aware_group_switch_v2`.

Previous safe baseline: Stage106 `render_aware_group_switch_v1`.

Policy JSON:

```text
experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_policy.json
```

Selection table:

| base | gap | selected candidate |
|---|---:|---|
| linear | 4 | endpoint_diff_baseline |
| linear | 8 | endpoint_diff_baseline |
| linear | 16 | endpoint_diff_baseline |
| stage65_adapter | 4 | endpoint_diff_baseline |
| stage65_adapter | 8 | endpoint_diff_baseline |
| stage65_adapter | 16 | endpoint_diff_baseline |

Validation summary on Stage113 rows:

| metric | value |
|---|---:|
| task count | 480 |
| endpoint PSNR | 20.3212149854921 |
| selected policy PSNR | 20.3212149854921 |
| gain vs endpoint | 0.0 |
| oracle task best PSNR | 20.382843220952523 |
| teacher oracle top10 PSNR | 22.077800340877268 |

Stage113 held-out diagnostic:

| metric | value |
|---|---:|
| Stage112 overall gain vs endpoint | 0.005831885580240304 |
| Stage112 min fold gain | -0.0059710516126523045 |
| Stage112 min group gain | 0.0 |
| Stage112 min fold-group gain | -0.03366017781158855 |
| Stage112 negative fold-group count | 4 |
| Stage112 Stage65 adapter gap4 gain | 0.0 |
| Stage114 strict-safe fold-group safe | 1 |

## Current Interpretation

- Residual-energy topk is useful but not render-aligned enough.
- Always using learned selection is unsafe because Stage65 adapter groups degrade.
- Metadata-only task-level switching is too weak.
- Anchor-stat task-level switching has signal but overfits on the current 120 rendered-label tasks.
- Selector-score task-level switching has signal but does not fix adapter-group regressions and remains below Stage106.
- Stage114 freezes endpoint-only as the strict-safe selector fallback chosen by the user.
- Stage112 is aggregate group-safe and improves over endpoint overall, but Stage113 shows it is not fold-group safe under a strict zero-regression criterion.
- Stage115 confirms index bytes can be removed when selected indices are decoder-reproducible, reducing q6 top10 side-info payload from `43381` to `36009` bytes on the smoke tasks.
- Stage116 shows deterministic value-only side-info is still larger than zlib entropy-coded index+value side-info for Stage96 broader linear groups (`1.1907909303963997`-`1.2055306636015155x`) and slightly larger for Stage65 adapter groups (`1.0139622082252686`-`1.0359950259093857x`).
- Stage117 shows q5/top10 deterministic side-info is rate-competitive (`30019 bytes`, below `5/6` Stage96 q6 top10 entropy reference groups), and q4/top10 or q6/top5 are below all `6/6` reference groups, but these are cross-setting rate-only comparisons.
- Stage118 resolves the q6/top10 rate gap by compressing deterministic value-only payloads: compressed q6/top10 is `0.024463971455891926`-`0.0309539794921875 MiB/intermediate` and below Stage96 entropy reference for `6/6` groups.
- Stage119 actual compressed sweep confirms q6/top10 mean compressed payload `29235.55 bytes`, q5/top10 `24537.56111111111`, q4/top10 `14982.574999999999`, q6/top5 `15040.72222222222`, and q4/top20 `28043.888888888887`; all these shortlist settings are below Stage96 entropy reference for `6/6` groups.
- Stage120 rendered smoke shows q4/top20 improves over q6/top10 by `+1.02156556263925 dB` on average while slightly lowering rate; q4/top10 is only `-0.03468351854032611 dB` vs q6/top10 at roughly half the side-info rate; q6/top5 drops `-0.6174121509811845 dB` and should be dropped.
- Stage121 broader validation confirms q4/top20 remains best: PSNR `20.689270746602087`, `+0.9223020959187475 dB` vs q6/top10, direct rate `0.1337680887378662` vs q6/top10 `0.1348375550108561`; q4/top10 gives PSNR `19.73848817438193`, only `-0.028480476301425663 dB` vs q6/top10 at direct rate `0.12124604296658527`.
- Stage122 RD package compares against Stage96 entropy reference: q4/top20 is lower rate by `-0.004194490114847849 MiB/frame` but lower PSNR by `-0.8305321438068527 dB`; q4/top10 is lower rate by `-0.016716535886128772 MiB/frame` and lower PSNR by `-1.781314716027025 dB`.
- Stage123 codec policy package freezes policy `compressed_deterministic_value_only_residual_codec_v1`; status remains `package_not_full_residual_predictor` because residual values are still teacher-derived.
- Stage124 no-teacher residual value predictor smoke: `adapter_delta_selected_v1` q4/top10 improves over linear base by `+0.027863362265533247 dB` and q4/top20 by `+0.019370720622054066 dB` on 12 rendered tasks, with zero residual/index payload bytes and no target dense anchor input.
- Stage125 broader validation confirms the no-teacher predictor gain is stable on 60 tasks: q4/top10 improves over linear by `+0.04401048394920189 dB`, q4/top20 by `+0.059456354043700026 dB`, both with zero residual/index payload bytes.
- Stage126 packaged selected residual predictor training data metadata: q4/top20 has `884760` train samples, q4/top10 has `442320`, feature dim `56`, residual dim `13`; no per-Gaussian tensors are saved.
- Stage127 trained small selected-residual MLPs: q4/top20 eval residual MSE reduction `0.08808295024199653`, q4/top10 eval reduction `0.10293283119315721`; checkpoints live under `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke` and are not committed.
- Stage128 integrated the Stage127 predictors into policy `predictor_only_selected_residual_codec_v1`; q4/top20 and q4/top10 checkpoints both exist/load, and predictor residual/index payload bytes are `0`.
- Stage129 shows the dedicated MSE-trained MLP is not render-safe: q4/top10 predictor PSNR `18.865777753557193` (`-0.08502524287394495 dB` vs linear) and q4/top20 `18.76520305064309` (`-0.1855999457880447 dB` vs linear), despite zero residual/index payload bytes.
- Stage130 comparison identifies the current best no-teacher deployable point as Stage125 `adapter_delta_selected_predictor` q4/top20: direct rate `0.11729838135687401`, PSNR `19.010259350474836`, zero residual/index payload bytes.
- Stage131 recommends `adapter_delta_selected_predictor/q4_top20` for Stage132 deployable policy and rejects `dedicated_mlp_selected_predictor_v1` as final because it regresses rendered PSNR.
- Stage132 packaged deployable policy `deployable_adapter_delta_selected_residual_codec_v1`: primary q4/top20 direct rate `0.11729838135687401`, PSNR `19.010259350474836`, zero residual/index payload bytes, decoder forbidden target dense/residual/RGB/oracle inputs.
- Stage133 final report confirms the current final deployable predictor is adapter-delta q4/top20; teacher q4/top20 remains reference-only at PSNR `20.689270746602087` with `28320.791666666668` residual payload bytes, and dedicated MLP remains rejected due to render regression.
- Stage134 confirms MLP underperforms adapter-delta on `47/60` tasks for both q4/top10 and q4/top20; q4/top20 mean MLP delta vs adapter is `-0.24505629983174473 dB`, so render-aware gating/training is required.
- Stage135 defines protocol `render_aware_adapter_delta_scale_calibration_v1`: sweep adapter-delta scales `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25]` for q4/top20 and q4/top10 without teacher residuals or target dense anchors.
- Stage136 smoke selects q4/top20 scale `0.5`: mean PSNR `20.135994499746502`, delta vs base `+0.056354919137378445`, improving over q4/top20 scale `1.0` PSNR `20.099010301231182` on the 12-task smoke slice.
- Stage106 remains the previous packaged baseline and should remain in comparisons.
- Stage110 group-best pattern has been frozen into Stage112 v2 for validation.
- Stage111 learned switch is not safe enough to package because adapter gap4 still regresses.
- If broader held-out validation later re-qualifies Stage112, it can be revisited as an optional gain candidate.
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

Status: completed on 2026-06-29.

Result:

- Stage112 alias mismatch count vs packaged selection table: `0`.
- Stage112 overall gain vs endpoint: `+0.005831885580240304 dB`.
- Stage112 aggregate group safety: pass (`min_group_gain = 0.0`, Stage65 adapter gap4 gain `0.0`).
- Stage112 fold-group safety: fail if zero regression is required (`min_fold_group_gain = -0.03366017781158855`, `4` negative fold-groups).
- Stage111 `score_stat_mlp_cv` remains unsafe: overall gain `+0.012037221045259486 dB`, but Stage65 adapter gap4 gain `-0.00797889356792674 dB`.
- Conclusion: do not treat Stage112 as final under strict held-out fold-group safety; either freeze endpoint-only fallback or collect broader rendered validation before using v2 as final.

Goal: validate the selected switch policy on held-out sequences or a broader eval set.

Actions:

- Re-run rendered selector validation with the frozen switch policy.
- Compare endpoint-only, learned-only, group switch, task switch, and oracle task switch.

### Stage114: Strict-Safe Selector Fallback Package

Status: completed on 2026-06-29.

Result:

- Packaged `strict_safe_endpoint_selector_v1` in `experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_policy.json`.
- Fixed candidate: `endpoint_diff_baseline`.
- Decoder inputs: none beyond normal decoder-available left/right anchors.
- Forbidden decoder inputs: `target_dense_anchor`, `target_residual`, `rendered_psnr`, `oracle_task_label`, `target_rgb`.
- Stage113 validation: selected PSNR `20.3212149854921`, gain vs endpoint `0.0`, min fold/group/fold-group gains all `0.0`, aggregate safe `1`, fold-group safe `1`.
- Stage112 v2 is rejected as final under strict safety due to fold-group regression.

Goal: freeze the strict-safe selector before deterministic-index side-info codec work.

### Stage115: Deterministic-Index Residual Codec Smoke

Status: completed on 2026-06-29.

Result:

- Added deterministic value-only helpers in `mono_dfcgs/residual_sideinfo_codec.py`.
- Added `scripts/run_stage115_deterministic_index_residual_codec_smoke.py`.
- Selector policy: Stage114 `strict_safe_endpoint_selector_v1` / `endpoint_diff_baseline`.
- Task count: `12` q12 eval tasks, both linear and Stage65 adapter base methods.
- Fixed index+value payload: `43381 bytes`, `0.04137134552001953 MiB/intermediate`.
- Deterministic value-only payload: `36009 bytes`, `0.034340858459472656 MiB/intermediate`.
- Savings: `7372 bytes`, ratio `0.8300638528388004`.
- Max deterministic decode diff vs fixed decode: `0.0`.
- Limitation: residual values are still teacher-derived; this is not residual value prediction.

Goal: remove transmitted selected-index bytes when decoder can reproduce selected indices.

### Stage116: Deterministic vs Entropy Side-Info Accounting

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage116_deterministic_vs_entropy_sideinfo_accounting.py`.
- Output: `experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/`.
- Inputs: Stage93 entropy smoke summary, Stage96 broader entropy RD rows, Stage115 deterministic summary, Stage78 q12 main anchor rate table.
- Row count: `12`; point count: `72`.
- Deterministic value-only payload: `36009 bytes`, `0.034340858459472656 MiB/intermediate`.
- Stage96 broader linear entropy side-info: `0.028486092885335285`, `0.02903069947895251`, `0.028838696687117867 MiB/intermediate`; deterministic ratio `1.2055306636015155`, `1.182915295732714`, `1.1907909303963997`.
- Stage96 broader Stage65 adapter entropy side-info: `0.033147705925835505`, `0.033867986578690376`, `0.03376620748768682 MiB/intermediate`; deterministic ratio `1.0359950259093857`, `1.0139622082252686`, `1.017018522793695`.
- Deterministic direct total rates with all side-info counted: gap4 `0.2162790792273858`, gap8 `0.131966245212987`, gap16 `0.08980982820578763 MiB/frame`.
- Limitation: deterministic endpoint-diff residual values are not rendered in this package; quality status is `not_rendered_rate_only`.

Goal: make index+value entropy vs deterministic value-only side-info accounting explicit before q-bit / keep-fraction sweeps.

### Stage117: Deterministic Side-Info q-bit / Keep-Fraction Sweep

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage117_deterministic_sideinfo_sweep.py`.
- Output: `experiments/stage117_deterministic_sideinfo_sweep/`.
- Derived geometry from Stage115: gaussian count `36860`, attr dim `13`.
- Sweep settings: keep fractions `[0.025, 0.05, 0.1, 0.15, 0.2]`, side bits `[2, 3, 4, 5, 6, 8]`.
- Row count: `180`; setting count: `30`.
- Stage115 q6/top10 deterministic setting reproduced: `36009 bytes`, `0.034340858459472656 MiB/intermediate`, `0/6` groups below Stage96 q6/top10 entropy reference.
- q5/top10 deterministic: `30019 bytes`, `0.02862834930419922 MiB/intermediate`, `5/6` groups below Stage96 q6/top10 entropy reference.
- q4/top10 deterministic: `24029 bytes`, `0.02291584014892578 MiB/intermediate`, `6/6` groups below Stage96 q6/top10 entropy reference.
- q6/top5 deterministic: `18040 bytes`, `0.01720428466796875 MiB/intermediate`, `6/6` groups below Stage96 q6/top10 entropy reference.
- Limitation: all non-q6/top10 comparisons are cross-setting rate-only; rendered quality is unknown.

Goal: identify rate-feasible deterministic settings before rendered validation.

### Stage118: Compressed Deterministic Value-Only Codec Smoke

Status: completed on 2026-06-29.

Result:

- Added compressed deterministic codec helpers in `mono_dfcgs/residual_sideinfo_codec.py`.
- Added `scripts/run_stage118_compressed_deterministic_codec_smoke.py`.
- Output: `experiments/stage118_compressed_deterministic_codec_smoke/`.
- Selector policy: Stage114 `strict_safe_endpoint_selector_v1` / `endpoint_diff_baseline`.
- Task count: `12` q12 eval tasks, both linear and Stage65 adapter base methods.
- Compressed deterministic decode vs raw deterministic decode: max diff `0.0`.
- Compressed deterministic decode vs fixed index+value decode: max diff `0.0`.
- Linear q6/top10 compressed payloads: `25652.333333333332`, `27322.2`, `26884.0 bytes` for gap4/8/16.
- Stage65 adapter q6/top10 compressed payloads: `30934.666666666668`, `32457.6`, `32162.5 bytes` for gap4/8/16.
- All `6/6` groups are below the Stage96 q6/top10 entropy-coded index+value reference.
- Limitation: residual values remain teacher-derived; this is still not residual value prediction.

Goal: remove selected-index bytes and entropy-compress value-only metadata/residual payload before rendered shortlist validation.

### Stage119: Actual Compressed Deterministic Sweep

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage119_actual_compressed_deterministic_sweep.py`.
- Output: `experiments/stage119_actual_compressed_deterministic_sweep/`.
- Sweep settings: keep fractions `[0.025, 0.05, 0.1, 0.15, 0.2]`, side bits `[2, 3, 4, 5, 6, 8]`.
- Row count: `720`; group count: `180`; setting count: `30`.
- Max decode diff vs raw deterministic across all rows: `0.0`.
- q6/top10: mean compressed payload `29235.55 bytes`, `0.02788119316101074 MiB/intermediate`, `6/6` groups below Stage96 entropy reference.
- q5/top10: `24537.56111111111 bytes`, `0.023400841818915472 MiB/intermediate`, `6/6` below reference.
- q4/top10: `14982.574999999999 bytes`, `0.01428849697113037 MiB/intermediate`, `6/6` below reference.
- q6/top5: `15040.72222222222 bytes`, `0.01434395048353407 MiB/intermediate`, `6/6` below reference.
- q4/top20: `28043.888888888887 bytes`, `0.02674473656548394 MiB/intermediate`, `6/6` below reference.
- Suggested Stage120 shortlist: q6/top10, q5/top10, q4/top10, q6/top5, q4/top20.
- Limitation: non-q6/top10 settings remain rate-only until rendered validation.

Goal: select candidate compressed deterministic settings for rendered validation.

### Stage120: Rendered Compressed Deterministic Shortlist Smoke

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage120_rendered_compressed_deterministic_shortlist.py`.
- Output: `experiments/stage120_rendered_compressed_deterministic_shortlist/`.
- Rendered shortlist: q6/top10, q5/top10, q4/top10, q6/top5, q4/top20.
- Row count: `120`; group count: `30`; setting count: `5`.
- Max decode diff vs raw deterministic: `0.0`.
- q6/top10: PSNR `20.509201246149463`, payload `29368.583333333332 bytes`, direct rate `0.13265951988895094`.
- q5/top10: PSNR `20.503631356341803`, delta vs q6 `-0.00556988980765986`, payload `24682.291666666668 bytes`.
- q4/top10: PSNR `20.474517727609136`, delta vs q6 `-0.03468351854032611`, payload `15117.083333333334 bytes`.
- q6/top5: PSNR `19.89178909516828`, delta vs q6 `-0.6174121509811845`, payload `15099.25 bytes`; drop this candidate.
- q4/top20: PSNR `21.530766808788716`, delta vs q6 `+1.02156556263925`, payload `28241.333333333332 bytes`, direct rate `0.131584490515782`.
- Suggested Stage121 broader validation settings: q6/top10 anchor, q5/top10, q4/top10, q4/top20.
- Limitation: still teacher-derived residual values; not residual value prediction.

Goal: identify rendered shortlist candidates for broader validation.

### Stage121: Broader Rendered Compressed Deterministic Validation

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage121_broader_rendered_compressed_deterministic_validation.py`.
- Output: `experiments/stage121_broader_rendered_compressed_deterministic_validation/`.
- Broader shortlist: q6/top10, q5/top10, q4/top10, q4/top20.
- Task count: `60`; row count: `480`; group count: `24`; setting count: `4`.
- Max decode diff vs raw deterministic: `0.0`.
- q6/top10: PSNR `19.766968650683353`, payload `29442.208333333332 bytes`, direct rate `0.1348375550108561`.
- q5/top10: PSNR `19.761047533309117`, delta vs q6 `-0.005921117374238853`, payload `24809.95 bytes`, direct rate `0.13041988921139727`.
- q4/top10: PSNR `19.73848817438193`, delta vs q6 `-0.028480476301425663`, payload `15190.475 bytes`, direct rate `0.12124604296658527`.
- q4/top20: PSNR `20.689270746602087`, delta vs q6 `+0.9223020959187475`, payload `28320.791666666668 bytes`, direct rate `0.1337680887378662`.
- q4/top20 is stable across all base/gap groups and should be the main packaged candidate.
- q4/top10 is the low-rate package candidate.
- Limitation: residual values remain teacher-derived; not residual value prediction.

Goal: validate Stage120 candidates on the same 60-task broader scale as Stage95/96.

### Stage122: Compressed Deterministic RD Package

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage122_compressed_deterministic_rd_package.py`.
- Output: `experiments/stage122_compressed_deterministic_rd_package/`.
- RD row count: `24`; RD point count: `60`.
- Package roles:
  - primary: `q4_top20`
  - low-rate: `q4_top10`
  - near-anchor: `q5_top10`
  - anchor: `q6_top10`
- q4/top20: direct rate `0.1337680887378662`, PSNR `20.689270746602087`, delta vs Stage96 entropy direct rate `-0.004194490114847849`, delta vs Stage96 entropy PSNR `-0.8305321438068527`.
- q4/top10: direct rate `0.12124604296658527`, PSNR `19.73848817438193`, delta vs Stage96 entropy direct rate `-0.016716535886128772`, delta vs Stage96 entropy PSNR `-1.781314716027025`.
- q5/top10: direct rate `0.13041988921139727`, PSNR `19.761047533309117`, delta vs Stage96 entropy direct rate `-0.007542689641316756`, delta vs Stage96 entropy PSNR `-1.7587553570998395`.
- q6/top10: direct rate `0.1348375550108561`, PSNR `19.766968650683353`, delta vs Stage96 entropy direct rate `-0.0031250238418579373`, delta vs Stage96 entropy PSNR `-1.7528342397255992`.
- Limitation: residual values remain teacher-derived; not residual value prediction.

Goal: freeze RD package for compressed deterministic value-only residual side-info.

### Stage123: Compressed Deterministic Codec Policy Package

Status: completed on 2026-06-29.

Result:

- Added `scripts/run_stage123_compressed_deterministic_codec_policy_package.py`.
- Output: `experiments/stage123_compressed_deterministic_codec_policy_package/`.
- Policy: `compressed_deterministic_value_only_residual_codec_v1`.
- Status: `package_not_full_residual_predictor`.
- Selector: `strict_safe_endpoint_selector_v1`.
- Selected candidate: `endpoint_diff_baseline`.
- Index rule: `endpoint_diff_topk_v1` with `keep_count=round(N * keep_fraction)`, left/right attr L2 endpoint-diff scores, top-k largest scores, sorted selected indices.
- Side-info codec: `compressed_deterministic_value_only_residual_sideinfo_v1`, magic `RSDZ`, header bytes `26`, zlib level `9`.
- Settings: q4/top20 primary, q4/top10 low-rate, q5/top10 near-anchor, q6/top10 anchor.
- Decoder forbidden inputs: target dense anchor, target residual, target RGB, oracle task label, transmitted selected indices.
- Limitation: residual values remain teacher-derived; not a residual value predictor.

Goal: freeze codec policy manifest for compressed deterministic value-only residual side-info.

### Stage124: Feed-Forward Residual Value Predictor Smoke

Status: completed on 2026-06-29.

Result:

- Added `mono_dfcgs/residual_value_predictor.py`.
- Added `scripts/run_stage124_feedforward_residual_value_predictor_smoke.py`.
- Output: `experiments/stage124_feedforward_residual_value_predictor_smoke/`.
- Predictor: `adapter_delta_selected_v1`.
- Task count: `12`; row count: `24`.
- Residual payload bytes: `0`; selected-index payload bytes: `0`.
- Target dense anchors are not loaded or used; target RGB is used only for offline rendered metrics.
- q4/top10: selected PSNR `20.107502942874657`, linear base PSNR `20.079639580609125`, full adapter PSNR `20.18242498795657`, delta vs base `+0.027863362265533247`, delta vs full `-0.0749220450819122`.
- q4/top20: selected PSNR `20.099010301231182`, delta vs base `+0.019370720622054066`, delta vs full `-0.08341468672539139`.

Goal: create first no-teacher feed-forward residual value predictor smoke.

## Later Plan After Selector Stabilizes

### Deterministic-Index Side-Info Codec

- Stage125: broaden feed-forward residual value predictor validation and/or train a dedicated selected residual value predictor.

### Residual Value Prediction

- Stage119: build selected-index residual value predictor manifest.
- Stage120: non-rendered predicted-value smoke.
- Stage121: rendered predicted-value smoke without teacher residual values.
- Stage122: hybrid low-rate correction if pure prediction is too weak.
- Stage123: package complete residual codec fields and RD accounting.

### Final Evaluation

- Stage124: broaden training labels.
- Stage125: final selected policy held-out eval.
- Stage126: integrated RD table.
- Stage127: ablations.
- Stage128: final method docs.
- Stage129: only after internal line stabilizes, resume FCGS/D-FCGS fair comparison.

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
- Stage113 held-out switch validation: `scripts/run_stage113_heldout_switch_validation.py`
- Stage114 strict-safe selector fallback package: `scripts/run_stage114_package_strict_safe_selector_fallback.py`
- Stage115 deterministic-index codec smoke: `scripts/run_stage115_deterministic_index_residual_codec_smoke.py`
- Stage116 deterministic vs entropy accounting: `scripts/run_stage116_deterministic_vs_entropy_sideinfo_accounting.py`
- Stage117 deterministic side-info sweep: `scripts/run_stage117_deterministic_sideinfo_sweep.py`
- Stage118 compressed deterministic codec smoke: `scripts/run_stage118_compressed_deterministic_codec_smoke.py`
- Stage119 actual compressed deterministic sweep: `scripts/run_stage119_actual_compressed_deterministic_sweep.py`
- Stage120 rendered compressed deterministic shortlist: `scripts/run_stage120_rendered_compressed_deterministic_shortlist.py`
- Stage121 broader rendered compressed deterministic validation: `scripts/run_stage121_broader_rendered_compressed_deterministic_validation.py`
- Stage122 compressed deterministic RD package: `scripts/run_stage122_compressed_deterministic_rd_package.py`
- Stage123 compressed deterministic codec policy package: `scripts/run_stage123_compressed_deterministic_codec_policy_package.py`
- Stage124 feed-forward residual value predictor smoke: `scripts/run_stage124_feedforward_residual_value_predictor_smoke.py`

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
- Stage113 held-out switch validation: `experiments/stage113_heldout_switch_validation/`
- Stage114 strict-safe selector fallback: `experiments/stage114_strict_safe_selector_fallback_package/`
- Stage115 deterministic-index codec smoke: `experiments/stage115_deterministic_index_residual_codec_smoke/`
- Stage116 deterministic vs entropy accounting: `experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/`
- Stage117 deterministic side-info sweep: `experiments/stage117_deterministic_sideinfo_sweep/`
- Stage118 compressed deterministic codec smoke: `experiments/stage118_compressed_deterministic_codec_smoke/`
- Stage119 actual compressed deterministic sweep: `experiments/stage119_actual_compressed_deterministic_sweep/`
- Stage120 rendered compressed deterministic shortlist: `experiments/stage120_rendered_compressed_deterministic_shortlist/`
- Stage121 broader rendered compressed deterministic validation: `experiments/stage121_broader_rendered_compressed_deterministic_validation/`
- Stage122 compressed deterministic RD package: `experiments/stage122_compressed_deterministic_rd_package/`
- Stage123 compressed deterministic codec policy package: `experiments/stage123_compressed_deterministic_codec_policy_package/`
- Stage124 feed-forward residual value predictor smoke: `experiments/stage124_feedforward_residual_value_predictor_smoke/`

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
- `scripts/run_exp_stage07_davis_ours_center_square_252.py`
- `scripts/run_exp_stage08_dfcgs_mono_adapter_diagnostic.py`
- `scripts/run_exp_stage_streamsplat_comparison.py`
