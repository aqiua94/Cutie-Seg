# CReFF 480/Seq8 Decision Frame

Created: 2026-05-31 UTC

## Current Run

- Method: CReFF-only
- Train resolution: 480x480 square crop after short-side resize
- Sequence length: 8
- Batch: 4
- Grad accumulation: 8
- Effective batch: 32
- LR: 1e-4
- AMP: off
- GOP: 12
- Trainable modules: CReFF only
- Frozen modules: encoder and decoder
- Checkpoint dir: `output/creff_davis_creff_only_s480_l8_b4a8`
- Train log: `logs/creff_davis_creff_only_s480_l8_b4a8.nohup.log`

## Reference Numbers

| Reference | Config | Status | Purpose |
|---|---|---|---|
| Upper bound | `cutie-base-mega.pth`, no-CReFF, GOP1, same compressed root | Done: J 0.8243, F 0.9016, J&F 0.8630 | True ceiling for the local compressed-frame evaluator |
| Lower bound | `cutie-base-mega.pth`, no-CReFF, GOP12, same compressed root | Running/queued | Quantify LR/compressed pipeline drop without CReFF |

## Step 500 Decision Tree

### J&F >= 0.80

Interpretation: the 128px train/eval scale gap was the main bottleneck, and the CReFF-only route works.

Action:
- Continue training to step 2000-3000.
- Start GOP sweep after the main setting stabilizes.
- Add per-sequence analysis against upper/lower references.

### 0.70 <= J&F < 0.80

Interpretation: the main bottleneck is improved, but a second-order gap remains.

Action:
- Start Stage 1: freeze decoder, train encoder adaptation on 480 LR input for 10-15 epochs.
- Then run Stage 2 again with CReFF-only.
- Keep CReFF-only as the main Stage 2 route.

### 0.60 <= J&F < 0.70

Interpretation: train resolution was not the only problem.

Action:
- Check whether resized HEVC MV still has visible quantization/block noise at 480.
- Try larger CReFF attention radius: 9 or 11 instead of 7.
- Inspect whether `feat_distill` meaningfully decreases and whether gradients reach CReFF.

### J&F <= 0.59

Interpretation: severe abnormality, effectively not better than the old 128px run.

Action:
- Stop training.
- Dump training batch loss, gradient, and feature tensors.
- Verify CReFF path is active and the checkpoint contains changed CReFF weights.

## Training Process Checks

Every 50 steps:

```bash
grep -E "step|loss|grad" logs/creff_davis_creff_only_s480_l8_b4a8.nohup.log | tail -20
```

Expected:
- `total_loss` should trend down by the first few hundred steps, though per-clip variance is high.
- `grad_norm` should not stay below `1e-4`.
- `grad_norm` should not stay above `100`.
- If loss is flat through step 300, do not wait for step 500 before investigating.

## Limitation Note For Part 1

The image-warp sign check showed that `+mv` is correct, but the absolute reconstruction error is still non-trivial:

- `drift-turn` t=7: `MSE(+mv)=0.023235`
- `scooter-gray` t=4: `MSE(+mv)=0.014479`

This supports a Part 1 limitation: even when MV sign, unit, scaling, and cumulative convention are correct, HEVC block MV remains noisy for fast-motion sequences. This motivates Part 2's VOS-aware MV.
