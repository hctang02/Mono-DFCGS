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
