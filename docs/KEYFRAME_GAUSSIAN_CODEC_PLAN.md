# Keyframe-Gaussian-Only Dynamic 3DGS Codec Plan

## Core Setting

We will build a StreamSplat-style dynamic 3DGS video codec where the transmitted payload contains only sparse keyframe Gaussian anchors and timestamps/metadata.

The encoder receives the full monocular video, selects keyframes, converts selected keyframes into Gaussian anchors, compresses these anchors, and transmits only those compressed keyframe Gaussians.

The decoder receives decoded keyframe Gaussians, predicts inter-keyframe dynamic Gaussians, and renders the full reconstructed video.

## Transmitted Payload

The bitstream contains:

- compressed keyframe Gaussian anchors
- keyframe indices / timestamps
- lightweight metadata such as resolution, frame count, FPS, GOP settings, and quantization parameters

The bitstream does not contain:

- non-keyframe RGB frames
- non-keyframe Gaussian parameters
- optical flow
- explicit deformation fields
- residual payloads
- decoder-generated intermediate dynamic Gaussians

## Rate Accounting

For comparison with FCGS, D-FCGS, and CWGS, the main rate is:

```text
total transmitted Gaussian bytes / total video frames
```

Decoder model weights and generated intermediate Gaussians are excluded by default and reported separately when needed.

## Proposed Pipeline

```text
full video
-> keyframe selection
-> keyframe Gaussian generation
-> keyframe Gaussian compression
-> transmitted keyframe Gaussian bitstream
-> Gaussian anchor decoding
-> inter-keyframe dynamic prediction
-> dynamic Gaussian rendering
-> reconstructed video / dynamic 3DGS
```

## Stage 1: Fair StreamSplat Baseline

Goals:

- compute all-frame, middle-only, and given-keyframe metrics
- evaluate gap 2/4/8/16
- separate model reconstruction ability from codec bitrate
- keep StreamSplat as a pretrained reconstruction baseline, not a Gaussian codec yet

Outputs:

- per-sample JSON summaries
- per-gap CSV tables
- RD-compatible rows with clearly labeled metric meaning

## Stage 2: Keyframe Gaussian Baseline

Goals:

- generate Gaussian anchors for selected keyframes
- evaluate keyframe Gaussian reconstruction quality
- measure anchor size under different point counts and quantization settings

Candidate anchor sources:

- StreamSplat static Gaussian encoder
- FCGS-style per-frame Gaussian anchors

## Stage 3: Uniform Keyframe Gaussian Codec

Goals:

- fixed-gap keyframe Gaussian transmission
- decode only keyframe Gaussians
- predict all non-keyframes at the decoder
- compare using average transmitted Gaussian size per video frame

Sweeps:

- keyframe gap: 2, 4, 8, 16
- anchor quality: point count, opacity pruning, field-wise quantization

## Stage 4: Gaussian-Anchor Dynamic Predictor

Goals:

- replace RGB/depth-conditioned StreamSplat dynamic prediction with Gaussian-anchor-conditioned prediction
- encode transmitted Gaussian attributes into anchor tokens
- predict inter-keyframe deformation, opacity lifecycle, and intermediate dynamic Gaussians

Recommended design:

```text
G_a, G_b, normalized time t
-> Gaussian anchor encoder
-> bidirectional dynamic predictor
-> deformation + opacity lifecycle
-> G_t
-> render I_t
```

## Stage 5: Training Strategy

Training rule:

- input can only include keyframe Gaussians
- intermediate frames are used only as supervision

Training phases:

- freeze keyframe Gaussian generator, train dynamic predictor
- train with variable GOP gaps: 2, 4, 6, 8, 12, 16
- add anchor quantization noise
- fine-tune with real compressed keyframe Gaussians

Losses:

- RGB reconstruction loss
- SSIM / DSSIM
- LPIPS
- temporal consistency
- depth consistency
- forward-backward consistency
- Gaussian motion smoothness
- opacity lifecycle regularization

## Main Comparisons

Compare against:

- FCGS per-frame
- D-FCGS
- FCGS-I + D-FCGS-P
- CWGS
- StreamSplat reconstruction upper bound

Use a unified x-axis:

```text
average transmitted Gaussian size per frame
```

Use y-axis metrics:

- PSNR
- SSIM
- LPIPS
- temporal consistency
- optional depth / novel-view quality

## Main Claim

KeyStreamSplat transmits only sparse compressed keyframe Gaussian anchors and reconstructs non-keyframes through decoder-side dynamic 3D Gaussian prediction, enabling extremely low transmitted Gaussian size while preserving a dynamic 3DGS representation.
