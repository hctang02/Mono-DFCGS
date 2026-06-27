# Pitfalls And Notes

## Storage

- `/mnt/ssd2tB` is almost full and should not receive copied StreamSplat checkpoints or output videos.
- `/mnt/hdd2tC/tmp/opencode` is acceptable for temporary experiments but not ideal for a long-term source repository.
- Long-term project source should live under `/mnt/hdd2tC/haocheng/Mono-DFCGS`.

## Git Hygiene

- Do not commit official StreamSplat checkpoints.
- Do not commit Python virtual environments.
- Do not commit compiled rasterizer build directories.
- Do not commit rendered videos or large image outputs.
- Keep NeoVerse/CompactWorld work separate from Mono-DFCGS work.

## StreamSplat Dependency State

The official StreamSplat checkout currently has untracked local runtime artifacts such as checkpoints, `__pycache__`, and compiled rasterizer build products. It should be treated as an external dependency rather than the main project repository.

## Stage 1 Notes

- StreamSplat baseline must report middle-only metrics; all-frame PSNR can be inflated by input keyframes.
- The last segment can be shorter than the nominal gap, especially for `robot`. Stage 1 handles this by grouping pairs by actual segment length instead of forcing a fixed output length.
- xFormers is unavailable/disabled in the current environment. The model still runs, but inference may be slower.
- Heavy frame/depth caches should remain outside git under `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs`.

## Stage 2 Notes

- Stage 2 initially completed GPU inference but failed at CSV writing because `total_bytes` was missing from the CSV field list. The script was fixed and CSV was regenerated from the already-written JSON, avoiding duplicate GPU inference.
- `static_anchor` and `full_half_anchor` must be kept distinct. The former is the intended transmitted keyframe Gaussian payload; the latter is a conservative upper-bound profile.
- Simple opacity pruning is weak for current StreamSplat anchors because most base opacities remain above low thresholds. Future rate control should include top-K or learned importance pruning.

## Stage 4 Notes

- Scripts under `scripts/` need to insert repo root into `sys.path` when executed directly; otherwise `mono_dfcgs` cannot be imported.
- PyTorch tensors do not have a `.softplus()` method. Use `torch.nn.functional.softplus` instead.
- The stage-4 predictor is only a smoke interface. Its MSE is measured against a synthetic linear target and should not be interpreted as reconstruction quality.

## Stage 6 Notes

- Real anchor `.pt` files are large and must stay outside git. Stage 6 writes them to `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage6_real_anchor_dataset`.
- The manifest in git stores absolute paths to the external `.pt` files. If the dataset is moved, the manifest should be regenerated or path-remapped.
- Frame cache filenames are one-indexed because they come from ffmpeg (`000001.png` for zero-based frame index 0). The manifest stores zero-based frame indices plus explicit paths, so downstream code should use the paths rather than reconstructing filenames.
- Stage 6 exports static keyframe anchors only. It does not yet export intermediate teacher Gaussians; renderer/RGB supervision will be added in a later stage.

## Stage 7 Notes

- Full DAVIS / YouTube-VOS / RE10K / CO3D roots were not found in default candidate paths. Any DAVIS-based experiment needs a download/mount/preprocess step before anchor export.
- `cv2.VideoCapture` reports `robot.mp4` as 79 frames, while previous ffmpeg extraction used by stage 1 produced 77 frames. For actual experiments, use extracted frame lists rather than container metadata alone.
- StreamSplat's DAVIS provider expects `ImageSets/2017/*.txt`, `JPEGImages/Full-Resolution`, and `Annotations_unsupervised/Full-Resolution`; DAVIS 480p downloads may need path adaptation or a custom provider for Mono-DFCGS experiments.

## Stage 8 Notes

- Stage 8 is intentionally non-strict by default. It writes empty manifests when no DAVIS / YouTube-VOS root exists, so the repo still records the required layout and a reproducible preflight result.
- DAVIS depth images are not part of the standard RGB/annotation download. They must be generated before anchor export; otherwise `ready_for_depth=true` but `ready_for_anchor_export=false` will appear in the sequence manifest.
- StreamSplat compatibility and Mono-DFCGS convenience differ slightly: the stage 8 script accepts DAVIS `480p` directories, but the original StreamSplat provider may still expect `Full-Resolution` unless adapted.

## Stage 9 Notes

- The first stage 9 run used the predictor's default `sigmoid` / `softplus` / rotation normalization output constraints. That mismatched the stage6 StreamSplat raw anchor attribute space and produced a much larger proxy loss than simple q8 linear interpolation.
- `GaussianAnchorDynamicPredictor` now keeps output constraints enabled by default but allows them to be disabled for raw-attribute proxy training. Renderer/RGB training should revisit the correct output-domain constraints.
- `torch.load(..., weights_only=True)` is used for stage6 `.pt` items to avoid the PyTorch pickle safety warning for this local tensor-only dataset format.

## Stage 10 Notes

- `gaussian_renderer_dynamic.render` expects StreamSplat dynamic Gaussian format: `xyz` and `rot` include static plus dynamic components. A single static predicted anchor must be wrapped with zero dynamic components before rendering.
- The renderer uses `opt.down_resolution` when it is non-empty. Stage 10 sets `opt.down_resolution = ()` and uses the default inference resolution `512x288` from `Options`.
- Stage 10's linear-anchor PSNR is a smoke metric only. It validates renderer wiring and target alignment, not the final Mono-DFCGS learned codec quality.

## Stage 10b Notes

- A freshly initialized predictor includes a random residual on top of linear interpolation, so initial RGB quality can be worse than the pure linear-anchor renderer smoke.
- Stage 10b verifies differentiability only. For meaningful RGB training, initialize from stage 9 proxy training or suppress residual at initialization, then fine-tune on multiple pairs and samples.
- Full 36864-Gaussian differentiable rendering is feasible on an empty L20 for a single pair, but this should not be scaled blindly without batching and memory checks.

## Stage 11 Notes

- Stage 11 keyframe selection is a selection/rate baseline only. It does not run reconstruction quality evaluation for non-uniform keyframes yet.
- Motion-aware selection can cluster keyframes around high-motion bursts. Later evaluation may need minimum temporal spacing constraints to avoid wasting budget.
- Gaussian-aware scores currently use stage6 gap=2 anchor MSE as a proxy. This requires the stage6 external `.pt` dataset to remain available at its manifest paths.

## Stage 12 Notes

- Non-uniform selected keyframes require grouping pairs by actual segment length because `opt.output_frames` and timestamp grids differ per segment.
- Top-k motion/RD selection without temporal spacing can cluster keyframes and create long uncovered segments. The `robot + rd_aware + gap4` smoke had max segment length 16 despite a gap4 budget.
- Stage 12 validates the selected-keyframe evaluation pipeline, but it still uses StreamSplat pair inference with RGB/depth inputs. It is not yet the final Gaussian-anchor-only decoder evaluation.

## Stage 13 Notes

- The first spacing-constrained greedy attempt still allowed clustering because it spent the budget on high-score frames before fully covering long gaps. The script now uses coverage-first splitting before score-based fill.
- With the same keyframe budget as uniform, strict `max_segment_length <= reference_gap` often collapses to uniform-like selection. Stage 13 therefore defaults to `max_segment_multiplier=2` to allow non-uniform choices while bounding uncovered intervals.
- Stage 13 only fixes the selection candidate set. Quality must be measured by Stage 12-style reconstruction using the new `*_spaced` methods.

## Stage 14 Notes

- The Stage 12 evaluator originally restricted methods to `uniform`, `motion_aware`, `gaussian_aware`, and `rd_aware`; it now also accepts `motion_spaced`, `gaussian_spaced`, and `rd_spaced`.
- Stage 14 uses a separate summary root so Stage 12 smoke outputs are not overwritten.
- `rd_spaced` improves the robot gap4 smoke but still trails the original uniform gap4 Stage 3 baseline; more samples and methods must be evaluated before claiming selection superiority.

## Stage 15 Notes

- The expanded RD curve confirms that current `rd_spaced` is safer than unconstrained top-k selection but still worse than uniform on n3dv and robot for gap4/8/16.
- Uniform Stage 15 rows reproduce Stage 1/3 metrics closely, which validates that the Stage 12/15 selected-keyframe evaluator is aligned with the earlier uniform pipeline.
- Future keyframe selection should optimize segment coverage and expected reconstruction error jointly; simple frame-wise scores are not enough.

## Stage 16 Notes

- Segment-level greedy selection is less prone to wasting the budget on local frame clusters than frame-wise top-k selection.
- The current segment cost is still a proxy based on image motion and stage6 Gaussian anchor differences, not measured reconstruction distortion.
- For final dataset RD curves, the comparison must include more than n3dv/robot; Stage 16 already emits selections for n3dv, meetroom, driving, and robot.

## Stage 17 Notes

- The first Stage 17 run completed all 24 reconstructions but failed during plotting because the plotting code expected flattened metric keys. The script now reads nested summary fields and reuses existing per-run JSON outputs.
- `segment_rd` is promising on n3dv and driving but not consistently better than uniform across all current samples. Future selection should learn or calibrate segment difficulty per dataset/content type.
- Stage 17 includes all four currently available samples, but final paper-level RD curves still need DAVIS / YouTube-VOS or broader StreamSplat protocol data once mounted.

## Stage 18 Notes

- `safetensors.torch.load_file` does not accept `device="cuda"`; use `cuda:0` or `cpu`. Stage 18 now maps CUDA devices to `cuda:0` under `CUDA_VISIBLE_DEVICES`.
- Instantiating `SplatModel` with `use_dino=True` may trigger DINOv2 weight download if the cache is missing. This happened on first Stage 18 run and downloaded to the default torch cache.
- The first module prefix list used `model.gs_predictor.upsampler`, but the actual static upsampler prefix is `model.gs_predictor.gaussian_upsampler`; the report was corrected and rerun.

## Stage 19 Notes

- Stage 19 is an aggregation stage, not a new inference stage. It reuses Stage1 quality and Stage2 q8 static-anchor rate estimates to avoid accidentally changing the pre-finetune baseline.
- Keep `raw_pred_gs_mib_per_frame` separate from `estimated_q8_static_mib_per_frame`: the former is decoder output tensor size, while the latter is the transmitted keyframe-anchor rate used for codec RD comparisons.

## Stage 20 Notes

- `model.opt.output_frames` must be updated for every variable-length segment before rendering. Otherwise the dynamic renderer indexes beyond the provided timestamps, e.g. default `output_frames=6` with a 5-frame segment raises `IndexError: index 5 is out of bounds`.
- For smoke training, save only trainable tensors outside the repo. The Stage20 trainable-state checkpoint is about 378M and should not be committed.
- A few low-lr smoke steps validate the training path but are not enough to improve quality. Treat Stage20 as wiring verification, not a final fine-tuned decoder result.

## Stage 21 Notes

- The Gaussian-anchor-only adapter smoke is valid as a payload test only if inputs remain limited to transmitted q8 keyframe anchors plus timestamp. Do not accidentally feed target RGB/depth except as training supervision.
- The current `GaussianAnchorDynamicPredictor` starts from random residual parameters, so early RGB quality can be worse than pure linear anchor interpolation. Future runs should add residual-zero initialization or initialize from the Stage9 proxy task.
- Full-anchor rendering is feasible for small smoke runs, but multi-sample training should keep checkpoints outside git and may need task caching to avoid repeated `.pt` loads.

## Stage 21b Notes

- Residual-zero initialization is necessary for fair comparison against linear anchor interpolation. Without it, a randomly initialized residual can make the adapter worse than the baseline before training.
- Do not select training rows by raw manifest order for small runs; the manifest is grouped by sample, so `rows[:N]` can accidentally train only on n3dv. Stage21b now uses sample-balanced row selection.
- The Stage21b improvement over linear on robot is positive but extremely small. Treat it as a validation of initialization/training direction, not as a meaningful final gain.

## Stage 21c Notes

- Multi-gap anchor-only training gives a larger positive average margin than Stage21b, but gap-wise behavior is not uniformly positive. In Stage21c, gap4 is slightly below linear while gap2/gap8/gap16 are positive.
- Before using an anchor adapter for RD curves, add validation-based checkpoint selection. A single final checkpoint may improve the average while hurting one GOP setting.
- Stage21c remains a medium development run on four local samples. It is not a substitute for DAVIS/YouTube-VOS or StreamSplat protocol-scale experiments.

## Stage 21d Notes

- The first Stage21d run completed training but failed when writing `stage21d_validation_log.csv` because validation rows included nested gap-wise dictionaries not listed in CSV fieldnames. The writer now emits only flat validation fields.
- Validation-based checkpoint selection is now in place, but the current run selected the final checkpoint. Keep the mechanism for future longer runs where overfitting may appear.
- Stage21d improves all robot eval gaps over q8 linear anchor interpolation, but the gain is still small and measured only on intermediate eval tasks.

## Stage 22 Notes

- Stage22 RD uses Stage21d robot intermediate-target eval and Stage2 q8 static-anchor rate. It is not comparable one-to-one with Stage19 full-video all-frame PSNR.
- Rate uses transmitted keyframe anchors only. The Stage21d adapter weights are reported separately and not counted in MiB/frame.

## Stage 23 Notes

- In full-video anchor-only evaluation, keyframes should be rendered directly from transmitted anchors for both methods. Do not pass keyframes through the adapter; the adapter is only for non-keyframe prediction.
- Stage23 PSNR/SSIM is an anchor-only reconstruction metric. It should not be directly compared against Stage19 original StreamSplat RGB/depth-conditioned decoder metrics without clearly separating input conditions.
- Stage23 full evaluation can be slow because it renders full 36k-Gaussian anchors for all middle frames across 16 sample-gap points. Run a single sample/gap smoke before full evaluation.

## Stage 24 Notes

- Stage24 plots are derived from Stage23 full-video anchor-only metrics. They should be labeled as anchor-only RD plots, not original StreamSplat decoder RD plots.
- The x-axis is q8 static keyframe-anchor MiB/frame from Stage2 estimates; no entropy-coded bitstream is included yet.

## Stage 25 Notes

- Leave-one-out validation is more meaningful than the previous fixed robot validation because every sample becomes held-out once.
- Stage25 still validates on intermediate tasks; full-video held-out evaluation must be run separately using the per-fold best checkpoints.
- In the current run, all best checkpoints are at the final step 384. Keep validation selection anyway for longer runs where overfitting may occur.

## Stage 26 Notes

- Stage26 uses one adapter checkpoint per held-out sample. Do not accidentally evaluate all samples with the same Stage21d development checkpoint when reporting leave-one-out results.
- The given keyframe rows still have zero delta by construction because both linear and adapter methods render transmitted q8 anchors directly for keyframes.
- Stage26 full-video gains are smaller than Stage25 intermediate-task gains, especially on robot gap16. Report both but treat full-video held-out RD as the stronger metric.

## Stage 27 Notes

- Stage16 unconstrained selected indices cannot be directly used as transmitted keyframe anchors with the current Stage6 dataset because Stage6 only contains anchors for even endpoint frames.
- Do not silently snap odd selected keyframes to nearby even anchors; that changes the selected frames and may duplicate/drop budget. Stage27 instead uses an explicit anchor-available constrained selector.
- The current anchor-available `anchor_segment_rd` selector is a negative result: 4/12 points beat uniform and the mean adapter PSNR delta is negative. Treat this as evidence that the selector objective needs redesign.

## Stage 28 Notes

- Stage28 does not replace the primary rate metric. Keep reporting primary q8 static anchor MiB/frame for continuity with Stage2/23/26.
- Stage28 entropy numbers are zero-order symbol entropy estimates, not a real entropy-coded bitstream. They exclude entropy model/header details beyond the simple metadata budget.
- Quantization parameter overhead is small only because anchors are large; if future pruning reduces Gaussian count substantially, overhead should be recomputed and may no longer be negligible.

## Stage 29 Notes

- Stage29 is an oracle/proxy selector because it uses intermediate q8 anchors from the held-out video to choose keyframes. Do not present it as a deployable encoder-side selector yet.
- The result is useful as an upper bound: aligned anchor-attribute costs improve 10/12 points, unlike Stage27's motion/RD heuristic.
- Driving gap4 and gap8 remain negative, so anchor-attribute MSE alone is not a perfect surrogate for full-video rendered PSNR.

## Stage 31 Notes

- Stage31 raw bitstream size is not identical to Stage28's compact binary estimate because Stage31 intentionally uses a JSON header for prototype readability.
- Roundtrip error is measured against direct q8 dequantized anchors, not original float anchors. The q8 quantization loss itself is already part of previous RD evaluations.
- zlib-compressed size is a practical baseline, but it should not be described as a learned entropy-coded bitstream.

## Stage 30 Notes

- Stage30 was executed after Stage31, so stage numbering is not chronological in the execution log.
- Current Stage6 unique anchor coverage is approximately half of all frames because only even endpoint anchors are available.
- Dense anchor export should write to external storage and should preferably deduplicate per-frame anchors; naive gap1 pair storage duplicates adjacent frame anchors.

## Stage 32 Notes

- Stage32 quality numbers are inherited from Stage26; only the rate axis changes from estimated q8 anchor payload to Stage31 actual raw/zlib bitstream sizes.
- Raw bitstream uses a readable JSON header, so it is slightly larger than the compact Stage28 estimate.
- Zlib RD curves are useful practical baselines, but should be labeled as generic-compressed q8 anchor bitstream RD, not final learned entropy coding.

## Stage 33 Notes

- Stage33 gap1 export duplicates anchors across adjacent pair files. Use it for selector/evaluation coverage first; later optimize storage with deduplicated per-frame anchors.
- Gap1 pair records have zero middle frames by construction. They are anchor-source records, not training pairs for intermediate reconstruction.
- External Stage33 `.pt` files are large and must stay outside git.

## Stage 34 Notes

- Dense anchors do not rescue the Stage16 `segment_rd` heuristic. The main issue is objective mismatch, not just anchor availability.
- Stage34 uniform numbers differ slightly from Stage26 because keyframe anchors now come from Stage33 gap1 pair exports rather than Stage6 multi-gap pair exports.
- Keep Stage34 as a negative ablation and use it to motivate learned/deployable selector costs derived from Stage29-style anchor-quality targets.

## Stage 35 Notes

- Stage35 is still an oracle/proxy selector: it uses dense intermediate anchors from the same video to choose keyframes.
- The 12/12 positive result should be reported as an upper bound and a training target for deployable selector research, not as the final encoder-side selector.
- Stage35 uniform rows use Stage33 gap1 anchors, so they may differ slightly from Stage26 uniform rows that used Stage6 multi-gap anchors.

## Stage 36 Notes

- Stage36 bitstreams are generated from Stage35 selections and Stage33 dense anchors. They are actual q8 anchor containers, but selector decisions are still oracle/proxy.
- Raw/zlib rate differences between uniform and oracle rows at equal keyframe count are caused by header length and selected-anchor compressibility, not by different keyframe counts.
- Roundtrip is measured against q8 dequantized anchors, not original fp16 anchors.

## Stage 37 Notes

- Stage37 labels are oracle/proxy labels derived from dense intermediate anchors. The features are deployable, but the labels are not available at encoder inference time.
- Segment length is highly correlated with the label, so predictor evaluation must check whether it learns more than just length.
- RGB motion features are weak in this first dataset; endpoint anchor features appear more useful for the oracle target.

## Stage 38 Notes

- Full-feature linear ridge can overfit or fail under leave-one-sample-out domain shift; robot held-out collapses badly.
- Length-only is a very strong ranking baseline for oracle cost, but a pure length-based selector may reduce to near-uniform under fixed keyframe budget.
- Before Stage39, consider sample-normalized targets/features or rank-based models rather than raw cross-sample linear regression.

## Stage 39 Notes

- Strong cost-prediction ranking metrics are not sufficient for keyframe-selection RD gains. Stage39 length-only ridge has high Stage38 Spearman but still loses to uniform on all 12 RD points.
- Raw predicted segment costs can produce DP selections that optimize the proxy label but hurt full-video rendered PSNR. Treat Stage39 as evidence for objective mismatch, not just model underfitting.
- Future deployable selectors should normalize costs per sample/candidate budget or optimize a selector-level/rank objective before re-running full-video RD.

## Stage 40 Notes

- Sample/candidate normalization fixes part of the cross-sample scale problem, especially for full-feature rank prediction.
- Predictor ranking quality must still be validated with actual DP selection and rendered RD; Stage39 showed that high Spearman alone can be misleading.
- Rank targets are useful for within-sample selection but no longer represent calibrated distortion magnitudes. Any DP cost derived from ranks should be treated as a relative selector score.

## Stage 41 Notes

- Stage41 confirms that better predictor ranking does not guarantee full-video RD improvement. All normalized/rank predicted selectors underperform uniform on all 12 evaluated points.
- Relative rank costs can distort DP layout under a fixed budget. Without a uniform prior or spacing regularization, DP may choose segments that look cheap under proxy cost but are bad for rendered PSNR.
- Future selector work should report uniform as a strong baseline and include layout-level constraints or directly optimized rendered-RD validation before claiming deployable selection gains.

## Stage 42 Notes

- Adding a uniform segment-length prior recovers Stage41 failures toward uniform but does not create a deployable gain; high prior weights mostly collapse to exact uniform selections.
- A selector can look better by becoming uniform-like. Track `exact_uniform_points` to avoid mistaking fallback-to-uniform for a learned selection improvement.
- The current strongest selector result remains the dense oracle/proxy upper bound, not the deployable predicted selector.

## Stage 43 Notes

- Keep three claims separate: adapter improvement over linear interpolation, oracle/proxy selector upper bound, and deployable predicted selector performance.
- The current deployable predicted selector result is negative; do not present Stage36 oracle gains as if they were deployable encoder-side selection.
- Stage43 Markdown is the safest source for current selector claim wording.

## Stage 44 Notes

- Rendered segment distortion is an offline label for selector training/oracle analysis, not an inference-time input. The final adaptive selector must use a frozen feed-forward predictor.
- The first Stage44 run produced a tensor shape warning because target RGB lacked the singleton time dimension. Fix target tensors to `[1, 1, 3, H, W]` before trusting MSE labels.
- The default Stage44 label is sampled for speed: mean sampled adapter MSE multiplied by full middle-frame count. Rerun with `--max_targets_per_segment=0` for all-middle-frame labels when scaling or finalizing results.

## Stage 45 Notes

- Rendered-distortion oracle is a stronger selector target than anchor-attribute proxy, but the sampled Stage44 label still has negative gap8/gap16 cases.
- Do not yet claim adaptive selector solved. Current result is promising first RD evidence: 8/12 positive and modest mean PSNR gain.
- Large negative gap16 points suggest adding min-gap/layout regularization or using all-middle-frame segment labels before moving to final predicted selector training.

## Stage 45b Notes

- A small uniform segment-length prior improves robustness more than hard minimum segment length. `alpha=0.1` removes large negative points while preserving a positive mean gain.
- Exact-uniform fallback must be counted explicitly; a robust selector may look safe because some points collapse to uniform.
- `minhalf` can be harmful despite sounding safer, producing a larger worst-case negative than raw rendered oracle.

## Stage 46 Notes

- Stage46 quality comes from Stage45b; only the rate axis changes to actual raw/zlib q8 anchor bitstream sizes.
- The calibrated adaptive selector is still rendered-oracle based. Do not call it fully feed-forward until Stage47/48 predictor selection is trained and evaluated.
- Rate deltas at equal keyframe count are tiny and mainly reflect header length and zlib compressibility of selected anchors, not different keyframe budgets.

## Stage 47 Notes

- High rendered-cost prediction correlation is necessary but not sufficient. Stage48 must validate the final DP-selected keyframes with rendered full-video RD.
- Segment length remains a very strong predictor even for rendered distortion labels. Always compare against length-only predictor variants.
- Sample-normalized rank predictors produce relative costs, not calibrated physical distortion. DP selection may need the same uniform prior used in Stage45b.

## Stage 48 Notes

- Stage48 confirms that feed-forward cost prediction still does not guarantee adaptive RD gains, even with rendered-distortion supervision.
- The best-looking rank predictor by correlation can produce very poor keyframe layouts. Selector training likely needs decision-aware objectives, not only segment-level regression/ranking.
- For current claims, keep Stage46 as oracle/calibrated upper-bound evidence and Stage48 as a deployable-selector gap that motivates the next research step.

## Stage 49 Notes

- `gap1` has no middle frames. Evaluators and comparison helpers must explicitly support empty middle metrics instead of subtracting `None` values.
- Stage49 all-frame RD can include `gap1`, but middle-only RD should exclude `gap1` by definition.
- For long rendering stages, write robust finalize/reuse paths. Stage49 now supports `--reuse_existing_csv` to regenerate summary and plots without rerendering.

## Stage 50 Notes

- Stage50 multi-bit payload is not bit-packed. `q6` uses uint8 storage, and q10/q12/q16 use uint16 storage. Label raw rates accordingly.
- Theoretical bitpacked MiB/frame is reported separately and should not be mixed with actual prototype raw container size.
- Stage50 validates container roundtrip only; quality at higher bit-depth needs Stage51 rendering with decoded/dequantized anchors.

## Stage 51 Notes

- Stage51 quality must be rerendered for each bit-depth; changing only the rate axis would be invalid.
- CSV writers for flattened eval rows must include `estimated_q8_static_mib_per_frame` even when the main rate is multi-bit raw/zlib.
- q12 to q16 gives limited extra PSNR, suggesting remaining quality is bounded by anchors/adapter/rendering rather than quantization alone.

## Stage 52 Notes

- D-FCGS `.log` files are single P-frame compression/decompression records. They expose motion/prior bits and P-frame metrics but are not complete video RD points without I-frame accounting and aggregation.
- FCGS/D-FCGS GOP summaries report full codec MiB/frame. Do not mix this rate with Mono-DFCGS transmitted keyframe Gaussian-anchor MiB/frame without labeling the protocol difference.
- `dummy_reference_images=true` summaries do not provide input-video PSNR/SSIM; use their `codec_psnr` only for compression-fidelity diagnostics.
- When classifying external result paths, use the most specific matching root. A broad `/mnt/hdd2tC/tmp/opencode` root can otherwise swallow multisequence or lowrate FCGS summaries into the wrong source group.

## Stage 53 Notes

- A comparison scaffold is not a final fair comparison. Keep `fair_local_run=false` for external FCGS/D-FCGS rows until inputs, frame sets, source Gaussian generation, and rate accounting are matched.
- The unified scaffold intentionally keeps two rate units: Mono-DFCGS anchor-only MiB/frame and full FCGS/D-FCGS codec MiB/frame. Do not plot them together without explicit labeling.
- For rows with dummy references, leave input-video PSNR/SSIM blank and keep `diagnostic_codec_psnr` separate.
- Use generic `secondary_rate_value` plus `secondary_rate_unit`; external rows may store total sequence MiB while Mono-DFCGS rows store raw anchor MiB/frame.

## Stage 54 Notes

- Stage54 is an analysis-only reuse of Stage48/49 RD outputs. It should not be described as a new rendered evaluation.
- The Stage48 predicted candidate layout pool has a very low oracle upper bound over uniform (`+0.0063 dB` mean all PSNR), so a better fallback policy alone cannot create large gains from these candidates.
- Layout imitation via simple overlap/Jaccard to the rendered-oracle layout collapses to uniform on all current points; oracle-layout similarity is not a deployable selector objective by itself.
- Leave-one-sample-out layout-threshold fallback remains slightly negative, so the final feed-forward selector likely needs decision-aware training/calibration on more samples rather than only segment-cost regression.

## Stage 55 Notes

- Stage55 is read-only preflight. Do not describe it as a dataset expansion run, depth preprocessing run, anchor export, or training stage.
- StreamSplat `provider_davis.py` is stricter than the Stage8 convenience manifest: it expects `JPEGImages/Full-Resolution`, `Annotations_unsupervised/Full-Resolution`, `ImageSets/2017/*.txt`, and provider-derived `depthImages/*_pred.png` at access time.
- StreamSplat `provider_vos.py` expects `train/valid/JPEGImages`; training split needs `train/Annotations`, while valid split uses dummy masks. Depth files are still expected as `depthImages/*_pred.png`.
- RE10K and CO3D should be treated as possible pretraining sources only unless a single-view extraction/evaluation protocol is explicitly defined. They must not leak multiview/camera information into final monocular codec claims.

## Stage 51b Notes

- Stage51b is plotting-only and reuses existing Stage51 CSV results. Do not describe it as rerendered quality or a new RD experiment.
- For future user-facing summaries, default to all-frame PSNR only unless middle-only PSNR is explicitly requested.
- Plot quantization bits and keyframe gaps with separate visual encodings. Mixing all q/g points into one same-color line makes the RD curve hard to read and can imply a false traversal order.
- Keep labeling `rendered_prior_0p1` as oracle/calibrated; it is not a deployable feed-forward selector.

## Recent Q&A Notes

- Stage3 PSNR must be described as an early StreamSplat-conditioned upper-reference, not as strict compressed Gaussian-anchor-only decoder quality.
- Future plans must explicitly include both deployable feed-forward keyframe selection and optional rate-counted side-information exploration.
- If depth, motion hints, residuals, or other side information are transmitted for non-keyframes, they must be included in the rate and clearly separated from the current keyframe-Gaussian-only main rate.
- Future plans should be logged before implementation, and each stage should get an individual record under `logs/stage_records/`.
- Treat short runs as smoke/infrastructure checks only. Medium and long training runs are required before making strong claims about adapter or selector gains.

## Stage 56 Notes

- Stage56 is a protocol-lock stage only. It does not change codec behavior or produce new RD numbers.
- Future side-information variants must report anchor rate, side-info rate, and total rate; do not mix them with keyframe-Gaussian-only main rate without labeling.
- `rendered_prior_0p1` remains an oracle/calibrated upper bound, not a deployable method.

## Stage56-70 Plan Update Notes

- DAVIS and YouTube-VOS preparation must be explicit in the plan because they are StreamSplat-scale video datasets needed for medium/long training.
- Keep dataset downloads and extracted roots outside git, preferably under `/mnt/hdd2tC/tmp/opencode/datasets`.

## Stage57 Compact Codec Notes

- True bit-packing reduces raw payload for non-byte-aligned bit depths, but it can reduce generic zlib effectiveness. In the Stage57 formal table, q6 compact raw saves 25% payload while compact+zlib is worse than legacy dtype+zlib.
- q8 and q16 are byte-aligned; compact payload size is identical to dtype storage, and only small metadata/header differences remain.
- Full Stage57 default coverage was CPU-time limited. Use scoped representative runs for codec correctness, then do broader RD integration in Stage58.
- Stage50 is intentionally pinned to `payload_encoding="dtype"`; otherwise rerunning Stage50 after Stage57 would silently change the meaning of its historical prototype table.

## Stage58 Compression RD Notes

- Stage58 does not rerender. All quality values are Stage51 `adapter_all_psnr`; codec variants only change the x-axis rate.
- `compact_bitpack_raw_payload_estimate` is payload-only and should not be mixed with actual container rates without labeling.
- Actual Stage57 compact raw/zlib rates are currently available only for the Stage57 formal subset, so their mean PSNR reflects uniform gap16 quality, not the full Stage51 distribution.

## Stage59 Dataset Prepare Notes

- Stage59 does not download DAVIS or YouTube-VOS. Official datasets may require manual download, login, or acceptance of terms.
- StreamSplat provider files exist, but no checked candidate root is provider-ready or anchor-export-ready.
- Stage60/61 cannot run on DAVIS/YouTube-VOS until frames, annotations, split files, and then predicted depth files are present.

## Stage59 Data Acquisition Notes

- DAVIS Full-Resolution unsupervised trainval can be prepared under `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` and becomes StreamSplat provider-ready after depth images are generated.
- YouTube-VOS `valid.tar` alone is not enough for StreamSplat provider readiness. The provider still expects `train/JPEGImages` and `train/Annotations` to exist.
- YouTube-VOS 2019 `train.tar` is about `9.26G` before extraction. Do not start it when `/mnt/hdd2tC` has only a few GiB free; extraction requires additional headroom.
- Remove partial `.part`, zip, and tar files after failed or completed downloads to avoid silently consuming scarce `/mnt/hdd2tC` space.

## Stage60 Depth Preprocess Notes

- Depth preprocessing writes many `*_pred.png` files into dataset roots and can consume several GiB. Always check `df -h /mnt/hdd2tC` before full runs.
- Stage60 uses DepthAnything V2 `vitl` through StreamSplat `DepthAnythingWrapper`; keep the checkpoint under the external StreamSplat checkout and do not commit it.
- DAVIS smoke frames should be skipped on the full run when `--skip_existing` is enabled; this explains `skipped_frames=2` in the formal DAVIS summary.
- A provider-ready dataset is not necessarily anchor-export-ready until depth files exist. Rerun Stage59 after Stage60 to confirm the transition.
- Python `csv.DictWriter` defaults to CRLF line endings. Use `lineterminator="\n"` for tracked experiment CSVs so `git diff --check` does not report trailing whitespace from carriage returns.

## Stage61 DAVIS Anchor Export Notes

- Do not launch full DAVIS all-gap anchor export without a storage preflight. With 90 sequences and gaps `1/2/4/8/16`, Stage61 estimates about `21950 MiB` of pair `.pt` anchor payload, before a safety reserve.
- The default DAVIS exporter args intentionally run a one-sequence one-pair smoke export. Full export requires explicit `--max_sequences 0 --max_pairs_per_sequence 0` and enough external space.
- Use `cuda:0` under `CUDA_VISIBLE_DEVICES=...` for `safetensors.torch.load_file`; plain `cuda` may fail in some safetensors versions.
- External Stage61 `.pt` files must stay under `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export` and must not be committed.
- DAVIS gap16 train+val export is feasible on the current mount and produced 425 pair records, but it still reduced free space to only about `3.3G`. Avoid additional gap exports until storage is freed.

## /data DAVIS Root Notes

- `/data` has enough free space for full DAVIS expansion and large anchor exports; prefer `/data/hctang/tmp/opencode/...` for new heavy outputs.
- The official DAVIS Full-Resolution root under `/data` includes 2017 semi-supervised train/val/test splits, 2017 unsupervised trainval annotations, 2019 unsupervised test splits, semantics, and scribbles.
- The official test-dev/test-challenge splits have frames but limited or no full masks. Use them carefully for anchor export or encoder-side evaluation, not supervised mask/RGB training claims unless protocol is explicit.
- Keep `/mnt/hdd2tC` for source, logs, and small summaries only; it is too full for more data or anchors.
- `/data` DAVIS train/val frames match the previous `/mnt/hdd2tC` DAVIS root exactly, so the generated train/val depth can be copied instead of recomputed. Do not assume this for test-dev/test-challenge; they still need depth generation if used.
- Long Stage61 exports should run with `--skip_existing` enabled. A timeout after 11311/12007 pairs was recovered by rerunning the same command.
- Avoid loading RGB/depth for already-complete sequence/gap pairs during resume; this was optimized in the Stage61 exporter.

## Stage62 Adapter Infra Notes

- Stage61 DAVIS manifest rows use `dataset/split/sequence`, while Stage21 helper functions expect `sample`. Stage62 maps them to `sample=DAVIS/<split>/<sequence>` before task construction.
- Stage62 smoke validates infra only. A `+0.0044 dB` margin after 5 tiny steps is not a meaningful adapter contribution claim.
- Keep best/final checkpoints and resume `.pt` state under `/data/hctang/tmp/opencode/mono_dfcgs_runs`; do not commit them.
- Resume runs can overwrite tracked summary CSV/JSON with the resumed run configuration. Record `start_step` and `resume_state` context in the summary before interpreting the logs.

## Stage63 Medium Pilot Notes

- Stage63 pilot reuses the Stage62 script, so output filenames still start with `stage62_` inside `experiments/stage63_medium_adapter_training_pilot`. Treat the directory name and stage record as the authoritative stage context.
- The 128-step pilot shows monotonic eval gain but covers only 16 train rows and 8 eval tasks. It is evidence that longer training is worth trying, not a final medium-training claim.
- Gap16 gains remain smaller than gap2/4 gains, suggesting longer-GOP behavior still needs more data, more steps, or stronger adapter architecture.

## Stage64 Adapter Teacher Study Notes

- Dense gap1 anchors can be used as offline teacher targets, but they must not be counted as test-time inputs or uncounted side information.
- In the 48-step Stage64 ablation, teacher loss improves teacher-anchor MSE but does not beat RGB render loss on rendered PSNR. Do not assume anchor-space teacher MSE is a sufficient proxy for final rendered quality.
- Hidden dim `256` is the better short-run architecture for both RGB and teacher variants, at about `402445` adapter parameters versus `102925` for hidden dim `128`.
- Stage64 is a small ablation over 16 train rows and 8 eval tasks. Use it to choose the next training route, not as a final adapter quality result.

## Stage65 Medium Training Notes

- Stage65 reuses the Stage64 training harness, so files inside `experiments/stage65_*` still start with `stage64_`. Treat the Stage65 directories and stage record as authoritative.
- The Stage65 medium run peaks at step `4000`; the final step `5000` remains positive but is worse. Report and use the best checkpoint unless explicitly studying final-step behavior.
- RGB render-loss training can improve rendered PSNR while making dense-anchor teacher MSE much worse. Do not use teacher MSE to choose RGB-only checkpoints.
- Stage65 validation is still intermediate-task evaluation, not full-video all-frame RD. A full all-frame evaluation is needed before using this as a final codec point.

## Stage66 Selector Dataset Notes

- Stage66 labels are offline dense-anchor MSE labels generated with the Stage65 adapter. They are allowed as training supervision but must not be used at selector test time.
- The Stage65 RGB adapter is worse than linear for every Stage66 segment in anchor-space MSE, despite improving rendered eval PSNR in Stage65. Treat anchor-space labels as a difficulty proxy, not as a final rendered-quality objective.
- Endpoint-anchor and RGB-motion features correlate strongly with the anchor-space label, so a feed-forward predictor is feasible, but Stage67/68 still need rendered-quality validation before selector claims.
- `endpoint_rot_mse` can have undefined Pearson correlation when the feature has near-zero variance; do not interpret blank correlation cells as failures.

## Stage67 Selector Predictor Notes

- Stage67 predicts the Stage66 anchor-space proxy label well, but this does not prove selector all-frame PSNR gains.
- Endpoint-anchor features dominate length-only and RGB-motion-only baselines on the scoped DAVIS split.
- Keep the fitted ridge parameters as a feed-forward predictor candidate, but Stage68 must evaluate deterministic DP selections and should prefer rendered/full-video labels before final claims.

## Stage68 Selector Rendered Validation Notes

- A high-quality anchor-space proxy predictor does not guarantee robust rendered all-frame PSNR gains. Stage68 is positive on average but has non-positive points.
- The predicted selector can collapse exactly to uniform for some settings, as in `DAVIS/val/bmx-trees gap16`.
- `DAVIS/val/goat gap8` is a clear negative case and should be inspected before final selector claims.
- Future selector stages should add rendered-distortion labels or a calibrated fallback-to-uniform policy rather than relying only on anchor-space proxy cost.

## Stage69 Fallback Calibration Notes

- Same-data threshold fallback can remove Stage68 negatives, but it is an analysis upper bound because it is trained and evaluated on the same rendered outcomes.
- Leave-one-sequence-out threshold fallback is worse than fixed predicted on the small Stage68 set. Do not claim that simple layout/cost fallback solves selector instability.
- The useful conclusion is negative/diagnostic: robust selector fallback needs more rendered labels or a stronger decision-aware objective.

## Stage70 Scoped RD Package Notes

- Stage70 is a scoped DAVIS eval-subset package, not the final benchmark.
- FCGS/D-FCGS apples-to-apples local baselines are still missing and must be added before final comparison claims.
- Stage70 uses q8 static keyframe-anchor MiB/frame; it does not include model weights as per-video rate.
- The predicted selector curve remains mixed because Stage68 selector robustness is unresolved.

## Stage71 Baseline Preflight Notes

- FCGS and D-FCGS code exist locally, but neither checkout contains a DAVIS/Mono-DFCGS adapter. Do not treat code availability as benchmark availability.
- FCGS expects FCGS-compatible static 3DGS `.ply` plus Gaussian-Splatting Scene validation inputs; Stage61 DAVIS anchors are tensor manifests and need an explicit conversion/wrapper.
- D-FCGS upstream examples assume multiview per-frame Gaussian sequences and 3DGStream/Colmap-style layouts. Any DAVIS adaptation must avoid introducing multiview information into a monocular-video claim.
- Old Stage52/53 FCGS/D-FCGS rows and legacy CWGS summaries are useful diagnostics, but they are non-DAVIS or protocol-mismatched and must not be added to final DAVIS apples-to-apples RD tables.

## Stage72/73 Low-PSNR Diagnosis Notes

- Original StreamSplat full dynamic DAVIS baseline is substantially higher than Stage70 q8 static-anchor-only results, so Stage70 should not be interpreted as a failure of DAVIS data/depth/metric alignment.
- Float static keyframe anchors render almost the same given-keyframe PSNR as original StreamSplat. This is a useful alignment check for RGB resize, frame index, and static-anchor-to-renderer bridge.
- Stage70 low all-frame PSNR mainly comes from discarding original StreamSplat dynamic Gaussian fields for middle-frame reconstruction. The current zero-dynamic static anchor wrapper is a deliberate reduced representation, not equivalent to full `pred_gs`.
- q8 anchor quantization has a large keyframe penalty on DAVIS, roughly `2.75 dB` given-keyframe PSNR in Stage73. Treat q8 as an aggressive lossy point; future RD should include q-bit or per-field quantization sweeps.

## Stage74 Stage72-vs-Actual Protocol Notes

- Stage72 is a scoped Mono-DFCGS diagnostic baseline, not the official StreamSplat paper benchmark protocol.
- The paper states PSNR/SSIM/LPIPS are evaluated at `256x256`; Stage72 originally reported `512x288` uint8 metrics. This alone can change PSNR by multiple dB.
- Paper-style dynamic interpolation should focus on non-input/middle frames under fixed interval settings; Stage72's all-frame PSNR mixes middle frames with rendered input/key frames.
- Full DAVIS val matters. The Stage72 four-sequence subset is harder than the full 30-sequence validation split.
- Always audit checkpoint loading when using `strict=False`. Stage74 found `missing_count=0` and `unexpected_count=0`, so the Stage72-vs-paper gap is protocol-driven rather than checkpoint mismatch.

## Stage76 Quantization Notes

- q8 static-anchor quantization is too aggressive for DAVIS direct keyframe rendering; Stage76 measured about `-2.69 dB` at `512x288` relative to float16.
- q10 recovers most keyframe quality with a moderate rate increase over q8; q12 is nearly lossless relative to float16 in direct keyframe rendering.
- Future RD curves should include q10/q12 operating points before concluding that the anchor-only representation is intrinsically too low quality.
