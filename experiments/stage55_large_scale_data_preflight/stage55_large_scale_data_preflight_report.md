# Stage55 Large-Scale Data Preflight

## Status

- Root candidates checked: 20
- Provider-layout-ready roots: 0
- Anchor-export-ready roots: 0
- DAVIS/YouTube-VOS sampled sequence rows: 0
- Current local anchor samples: 4

## Protocol Matrix

| Protocol | Dataset | Local status | Mono-DFCGS role | Next action |
|---|---|---|---|---|
| current_local_development_set | n3dv/meetroom/driving/robot | ready for development anchors: 4 samples | debug/RD development only | Use for selector/adapter iteration while waiting for larger mounted datasets. |
| streamsplat_original_davis | DAVIS | ready roots: 0 | preferred next large-scale single-view expansion | Mount DAVIS 2017 Full-Resolution layout, run depth preprocessing, then export anchors. |
| streamsplat_original_vos | YouTube-VOS | ready roots: 0 | secondary single-view expansion after DAVIS | Mount YouTube-VOS train/valid, generate depthImages/*_pred.png, then export anchors. |
| streamsplat_combined_training | RE10K/CO3D/DAVIS/VOS | provider files present | possible pretraining source only | Use only after defining a single-view extraction/evaluation protocol. |

## Interpretation

- Stage55 is a read-only preflight. It does not download data, preprocess depth, export anchors, or train models.
- DAVIS and YouTube-VOS are the cleanest next expansion targets because they are single-view video sequence datasets in the StreamSplat codebase.
- RE10K and CO3D can be useful for pretraining only after a single-view extraction protocol is specified; they should not be used to support final monocular codec claims with multiview information.
- A root is anchor-export-ready only when provider layout exists and predicted depth files are already present in the provider-derived `depthImages/*_pred.png` locations.
