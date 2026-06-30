# Stage161 Stage158 Method Narrative Package

## Claim

`streamsplat_guided_half_anchor_entropy_residual_v1` is the current quality-first middle-frame recovery method.
It keeps original StreamSplat as the target-time motion/geometry base and corrects one selected half-anchor with counted Gaussian-domain entropy residual side-info.
Rate is explicitly counted but not aggressively optimized at this stage, per user direction.

## Decoder Contract

Allowed decoder inputs:

- Original StreamSplat endpoint/base inputs.
- Normalized time.
- Encoded q6/keep1.0 entropy residual payload for the selected half-anchor.
- Counted one-byte half selector.

Forbidden decoder inputs:

- Target dense anchor.
- Target RGB.
- Unencoded target residual tensors.
- Oracle labels not represented in the transmitted payload.

## Gap-Level Evidence

| gap | method | tasks | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | direct rate ref | status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | stage151_linear_base_entropy_sideinfo_reference | 60 | 22.895456 | 0.602004 | 0.772679 | 0.347528 | 29908.516667 |  | historical_psnr_recovery_reference_not_final_visual_base |
| 4 | original_streamsplat_middle_base | 60 | 22.064218 | 0.600809 | 0.801367 | 0.301495 | 0.000000 |  | perceptual_motion_geometry_base |
| 4 | stage155_q4_image_residual_upper_bound | 30 | 32.280546 | 0.878672 | 0.977598 | 0.152828 | 78548.733333 | 0.256848 | upper_bound_only_not_final_gs_method |
| 4 | stage156_sampled_half_anchor_keep1_q6 | 30 | 29.880609 | 0.879575 | 0.985351 | 0.164580 | 207591.666667 | 0.379913 | sampled_gs_domain_discovery |
| 4 | stage158_streamsplat_guided_half_anchor_entropy_residual_v1 | 60 | 29.780485 | 0.877938 | 0.985088 | 0.166020 | 209392.833333 | 0.381631 | current_quality_first_middle_recovery_policy |
| 8 | stage151_linear_base_entropy_sideinfo_reference | 60 | 21.809852 | 0.563607 | 0.722634 | 0.384234 | 30091.616667 |  | historical_psnr_recovery_reference_not_final_visual_base |
| 8 | original_streamsplat_middle_base | 60 | 20.337275 | 0.520331 | 0.700537 | 0.359337 | 0.000000 |  | perceptual_motion_geometry_base |
| 8 | stage155_q4_image_residual_upper_bound | 30 | 31.718739 | 0.863465 | 0.974260 | 0.170844 | 82349.400000 | 0.176160 | upper_bound_only_not_final_gs_method |
| 8 | stage156_sampled_half_anchor_keep1_q6 | 30 | 29.547440 | 0.870055 | 0.984077 | 0.177570 | 214067.133333 | 0.301776 | sampled_gs_domain_discovery |
| 8 | stage158_streamsplat_guided_half_anchor_entropy_residual_v1 | 60 | 29.578682 | 0.869660 | 0.983847 | 0.178535 | 215967.883333 | 0.303588 | current_quality_first_middle_recovery_policy |

## Stage160 Subjective Evidence

- Video: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4`
- Contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg`
- Video bytes: `4180215`
- Contact sheet bytes: `8739496`

| sequence | tasks | key PSNR/LPIPS | Stage158 middle PSNR/LPIPS | original middle PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | rate ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| bike-packing | 2 | 26.328800/0.172648 | 25.916473/0.200759 | 20.774123/0.330460 | 5.142350/-0.129701 | 173028.500000 | 0.346951 |
| breakdance | 2 | 25.790860/0.149225 | 25.628598/0.170077 | 20.273111/0.230913 | 5.355487/-0.060835 | 123729.000000 | 0.299935 |
| camel | 2 | 25.609043/0.182846 | 25.580679/0.187973 | 21.878505/0.245850 | 3.702174/-0.057876 | 228002.500000 | 0.399378 |
| car-shadow | 2 | 29.794095/0.127689 | 29.522755/0.151694 | 23.489481/0.248052 | 6.033274/-0.096358 | 214037.000000 | 0.386060 |
| cows | 2 | 24.714921/0.205230 | 24.705746/0.206553 | 22.211391/0.252227 | 2.494355/-0.045674 | 216943.000000 | 0.388831 |
| dance-twirl | 2 | 27.554268/0.204801 | 26.756008/0.249167 | 18.425364/0.389624 | 8.330645/-0.140457 | 206271.500000 | 0.378654 |
| goat | 2 | 27.469379/0.169521 | 27.287905/0.175273 | 17.199419/0.406487 | 10.088486/-0.231214 | 251509.000000 | 0.421796 |
| gold-fish | 2 | 33.873268/0.067136 | 33.412370/0.079089 | 26.858579/0.145390 | 6.553792/-0.066301 | 191200.000000 | 0.364281 |
| kite-surf | 2 | 32.445948/0.094652 | 32.252243/0.108727 | 25.022019/0.203144 | 7.230225/-0.094417 | 217840.500000 | 0.389687 |
| motocross-jump | 2 | 34.905151/0.075903 | 31.391121/0.300439 | 13.738418/0.635566 | 17.652703/-0.335127 | 247438.500000 | 0.417914 |
| scooter-black | 2 | 27.821884/0.095169 | 27.336615/0.138989 | 16.436703/0.339682 | 10.899911/-0.200693 | 246170.000000 | 0.416704 |
| soapbox | 2 | 31.489589/0.091349 | 30.906671/0.125260 | 21.463903/0.280982 | 9.442768/-0.155722 | 241610.500000 | 0.412356 |

## Evidence Chain

| stage | role | decision | key result |
|---:|---|---|---|
| 151 | historical_psnr_reference | not_final_visual_base | Linear-base q6/top10 side-info recovered corrected PSNR targets but Stage153/152 showed visual/perceptual risk. |
| 153 | multi_metric_diagnostic | psnr_alone_insufficient | Stage151 improved PSNR/SSIM but LPIPS and bad cases motivated returning to original StreamSplat guidance. |
| 154 | streamsplat_base_alignment | use_original_streamsplat_as_base | Original StreamSplat had lower PSNR than Stage151 but better LPIPS, making it the preferred motion/geometry base. |
| 155 | achievability_upper_bound | image_residual_is_upper_bound_only | q4 image residual on StreamSplat base reached 31-32 dB but is not the final Gaussian-domain method. |
| 156 | gs_domain_discovery | select_best_half_keep1_q6 | Half-anchor Gaussian residual keep1/q6 exceeded 29 dB sampled with improved LPIPS/SSIM over original StreamSplat. |
| 158 | policy_contract | freeze_quality_first_policy | Broader 120-task validation passes: gap4 29.7805 dB, gap8 29.5787 dB, LPIPS improves on both gaps. |
| 160 | subjective_evidence | subjective_examples_available | Extended 24-frame gap4 video and contact sheet are available: /data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4; /data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg. |

## Next Direction

Proceed to keyframe selector work. Encoder-side RGB/motion cues are allowed for selecting keyframes if keyframe indices are transmitted and counted; Stage162 will audit feature sources and feed-forward validity.
