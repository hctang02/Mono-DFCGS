# Stage190 Paper-Facing Package

Date: 2026-07-01

## Goal

Package the Stage158/165/183-189 evidence into paper-facing tables, method framing, decoder contract, limitations, and next-action notes without running new rendering or payload measurement.

## Plan

- Reuse Stage185/186 measured RD-quality, Stage187 selector ablation, Stage188 lower-budget sensitivity, and Stage189 failure-case outputs.
- Generate Markdown/JSON/CSV artifacts under `experiments/stage190_paper_facing_package/`.
- Include a title/abstract draft that emphasizes Mono-DFCGS and recovery-aware adaptive keyframe scheduling rather than naming the baseline in the title.
- Include decoder allowed/forbidden inputs and RD accounting caveats.
- Include paper-ready claim boundaries: adaptive is a middle RD point, lower-budget sensitivity remains above gap8 under additive scope, and Stage188 additive rates must not be mixed with Stage185 schedule-packed rates.

## Success Criteria

- Report has compact paper tables for measured RD-quality, feature ablation, lower-budget sensitivity, and failure-case analysis.
- Package JSON records claims, non-claims, decoder contract, limitations, recommended title, and next refinement.
- CSV summary rows are easy to import into a manuscript or slides.
- No heavy media, anchors, checkpoints, or tensors are created.

## Execution

- Pre-run GPU check: `nvidia-smi` recorded on 2026-07-01 11:54:57; Stage190 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage190_paper_facing_package.py`
- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage190_paper_facing_package.py`

## Outputs

- Package: `experiments/stage190_paper_facing_package/stage190_paper_facing_package.json`
- Report: `experiments/stage190_paper_facing_package/stage190_paper_facing_report.md`
- Measured RD-quality table: `experiments/stage190_paper_facing_package/stage190_paper_table_measured_rd_quality.csv`
- Selector ablation table: `experiments/stage190_paper_facing_package/stage190_paper_table_selector_ablation.csv`
- Lower-budget sensitivity table: `experiments/stage190_paper_facing_package/stage190_paper_table_lower_budget_sensitivity.csv`
- Candidate failure table: `experiments/stage190_paper_facing_package/stage190_paper_table_candidate_failures.csv`
- Promoted rate-risk table: `experiments/stage190_paper_facing_package/stage190_paper_table_promoted_rate_risks.csv`
- Residual hotspot table: `experiments/stage190_paper_facing_package/stage190_paper_table_residual_hotspots.csv`
- Claims/limitations table: `experiments/stage190_paper_facing_package/stage190_claims_and_limitations.csv`

## Results

Package counts:

| table | rows |
|---|---:|
| measured RD-quality | `3` |
| selector ablation | `8` |
| lower-budget sensitivity | `6` |
| candidate failures | `3` |
| promoted rate risks | `2` |
| residual hotspots | `10` |
| claims and limitations | `9` |

Recommended title:

- `Mono-DFCGS: Recovery-Aware Adaptive Keyframe Scheduling for Monocular Dynamic Gaussian Splatting Compression`

Core paper framing:

- Current strongest claim: Mono-DFCGS adaptive is a measured middle RD point, with quality above uniform gap8 at higher rate and rate below uniform gap4 at lower quality.
- Non-claim: do not claim the frozen adaptive schedule is lower-rate than uniform gap8.
- Scope caveat: Stage188 additive rates compare lower-budget candidates internally and must not be numerically mixed with Stage185 schedule-packed rates.
- Decoder contract: decoder receives original StreamSplat endpoint/base inputs, normalized time, encoded q6/keep1.0 entropy residual payload, counted one-byte half selector, and transmitted schedule/keyframe metadata.
- Forbidden decoder inputs: target dense anchor, target RGB, unencoded target residual, rendered quality/oracle labels, and selector features not represented by transmitted schedule metadata.

## Decision

- Decision: `paper_facing_tables_and_claim_boundaries_packaged`.
- Stage190 is ready as the paper/slides handoff package for the current Mono-DFCGS adaptive keyframe line.
- If final lower-budget RD claims need same-scope measurement, the next experiment should schedule-pack selected Stage188 candidates before claiming rate-frontier improvements.
