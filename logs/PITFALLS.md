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
