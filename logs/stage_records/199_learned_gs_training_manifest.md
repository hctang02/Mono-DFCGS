# Stage199 Learned GS Training Manifest

Date: 2026-07-01

## Goal

Build a lightweight multi-gap train/eval task manifest for the Stage197/198 GS-native predictive codec route.

## Plan

- Use Stage61 dense gap1 DAVIS anchor exports as canonical per-frame GS references.
- Generate uniform-gap tasks for gaps `2,4,6,8,12,16` without copying anchors or tensors.
- Keep target dense anchors and RGB paths marked as training/encoder-side labels only.
- Audit split separation, dense anchor coverage, RGB coverage, gap coverage, and Stage197 decoder-contract compliance.

## Success Criteria

- Task manifest covers train/eval DAVIS sequences with all target gaps.
- Missing dense anchor and RGB coverage counts are zero.
- Train/eval sequence overlap is zero.
- Report states that runtime decoder inputs exclude target dense anchors, target RGB/image residuals, and oracle labels.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage199_learned_gs_training_manifest.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 23:24:58; Stage199 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage199_learned_gs_training_manifest.py`

## Outputs

- Output root: `experiments/stage199_learned_gs_training_manifest/`
- Task manifest: `experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv`
- Summary: `experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_summary.csv`
- Sequence coverage: `experiments/stage199_learned_gs_training_manifest/stage199_sequence_coverage.csv`
- Contract audit: `experiments/stage199_learned_gs_training_manifest/stage199_contract_audit.csv`
- Missing references: `experiments/stage199_learned_gs_training_manifest/stage199_missing_references.csv`
- Package: `experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_manifest_package.json`
- Report: `experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_manifest_report.md`

## Results

- Total task rows: `29204`.
- Missing reference rows: `0`.
- Split coverage: train `60` sequences / `4209` frames; eval `30` sequences / `1999` frames.
- Gap coverage: `2,4,6,8,12,16` for both train and eval.
- Eval q12 task rows by gap: gap2 `974`, gap4 `1463`, gap6 `1627`, gap8 `1707`, gap12 `1788`, gap16 `1830`.
- Train q12 task rows by gap: gap2 `2055`, gap4 `3087`, gap6 `3430`, gap8 `3604`, gap12 `3776`, gap16 `3863`.
- Contract audit passed: dense-anchor coverage, RGB-label coverage, split separation, gap coverage, Stage197 decoder contract, and lightweight-reference-only checks.
- Runtime decoder fields exclude target dense anchors, target RGB/image residuals, and oracle schedule/quality labels; target dense anchors and RGB are training/encoder-side label sources only.

## Decision

- Decision: `manifest_ready_for_stage200_architecture_package`.
- Proceed to Stage200 architecture package.
