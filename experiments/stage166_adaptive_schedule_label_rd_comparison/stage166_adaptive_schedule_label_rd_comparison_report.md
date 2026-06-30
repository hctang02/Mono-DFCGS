# Stage166 Adaptive Schedule Label/RD Comparison

## Scope

This is a pre-render label/RD proxy. It compares schedule metadata, keyframe count, and sampled Stage158 residual labels; it does not rerender the adaptive schedules.
Promoted rows only mean that the sampled target index becomes a keyframe under that schedule; this is not a substitute for rendered uniform-gap quality evaluation.

## Schedule Comparison

| schedule | keys | key ratio | metadata bytes | main anchor proxy | promoted rows | hard coverage | payload coverage | avoided payload | total proxy MiB/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 292 | 0.146073 | 1 | 0.097625 | 0/120 | 0.000000 | 0.000000 | 0 | 0.300453 |
| stage165_adaptive | 358 | 0.179090 | 327 | 0.120431 | 70/120 | 0.733333 | 0.819444 | 16241740 | 0.194182 |
| uniform_gap4 | 536 | 0.268134 | 1 | 0.181938 | 8/120 | 0.166667 | 0.069444 | 1792122 | 0.370524 |

## Adaptive Takeaway

- Adaptive keyframes: `358`, between uniform gap8 `292` and uniform gap4 `536`.
- Adaptive metadata: `2610` bits / `327` bytes.
- Sampled hard-row coverage: `22` / `30`.
- Sampled high-payload coverage: `59` / `72`.
- Sampled residual payload avoided versus treating all sampled targets as middle frames: `16241740` bytes.

## Smoke Candidates

| rank | sequence | reason | score | hard promoted/missed | payload promoted | extra keys | avoided payload |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | motocross-jump | hard_promoted;payload_heavy_promoted;many_extra_keyframes | 27 | 4/0 | 4 | 3 | 1006234 |
| 2 | cows | hard_promoted;many_extra_keyframes | 24 | 4/0 | 2 | 4 | 881313 |
| 3 | camel | hard_promoted;payload_heavy_promoted;many_extra_keyframes | 24 | 3/0 | 4 | 4 | 931636 |
| 4 | breakdance | hard_false_negative | 20 | 0/4 | 0 | 0 | 0 |
| 5 | dance-twirl | hard_false_negative;hard_promoted;many_extra_keyframes | 18 | 2/1 | 1 | 3 | 652400 |
| 6 | scooter-black | hard_promoted;payload_heavy_promoted;many_extra_keyframes | 18 | 2/0 | 3 | 4 | 934479 |
| 7 | india | hard_promoted;payload_heavy_promoted;many_extra_keyframes | 17 | 2/0 | 3 | 3 | 685389 |
| 8 | shooting | hard_promoted;payload_heavy_promoted;many_extra_keyframes | 16 | 1/0 | 4 | 4 | 954419 |
| 9 | car-roundabout | hard_promoted;payload_heavy_promoted;many_extra_keyframes | 16 | 1/0 | 4 | 4 | 889252 |
| 10 | bike-packing | hard_false_negative;hard_promoted | 15 | 1/2 | 0 | 1 | 211478 |

## Remaining Risk

Hard-label false negatives remain in:
- `breakdance`: `4` hard rows not promoted
- `bike-packing`: `2` hard rows not promoted
- `dance-twirl`: `1` hard rows not promoted
- `dogs-jump`: `1` hard rows not promoted

## Decision

- Decision: `promising_for_small_rendered_smoke`.
- Run a small rendered smoke before claiming final RD, because inserted keyframes change subsequent interpolation intervals and this proxy does not rerender them.
- Decoder-side contract is unchanged: transmit the adaptive schedule metadata; do not require RGB/motion feature extraction at the decoder.
