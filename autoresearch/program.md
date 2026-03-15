# Foresight Autoresearch Program

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Autonomous AI agent experiments to push past the Phase 3 wall.

## Context

Foresight explores whether AI can "imagine the future" — generate video predictions and use them to improve reasoning. We proved:
- ✅ VLM latents can be decoded (Gate 1)
- ✅ Small adapters bridge VLM → video generator (Gate 2)
- ✅ "Video Predicts → VLM Describes" pipeline works — VLM retains 93% action understanding on generated video (Gate 3)
- ❌ VLM cannot directly predict future states — 7 architectures failed, none beat copying the input frame (C3)

**The wall:** VLM prediction vs copy metric is at -4.5%. We need >0%.

## Your Mission

You are an autonomous research agent. Your job is to iterate on the prediction architecture to beat the copy baseline. Each experiment should:

1. **Modify** the prediction approach (architecture, training, loss function, conditioning)
2. **Train** for a fixed 5-minute budget on SSv2 data
3. **Evaluate** against the copy baseline using cosine similarity (cos_sim) and pixel L1
4. **Record** results in a structured log
5. **Keep or discard** based on whether prediction > copy on ANY metric
6. **Repeat**

## The Gate

**PASS condition:** prediction_metric > copy_metric on cos_sim OR pixel_L1 for future frame prediction.

Current baselines to beat:
- cos_sim: copy = 0.975, best model = 0.941 (gap: -0.034)
- pixel_L1: copy = 0.070, best model = 0.209 (gap: -0.139, lower is better)

## What You Can Modify

The prediction head / adapter architecture. Fair game:
- Temporal attention mechanisms
- Conditioning strategies (how many frames, which frames)
- Loss functions (contrastive, pixel, perceptual, temporal)
- Training regime (learning rate, warmup, curriculum)
- Architecture (transformer layers, attention patterns, skip connections)
- Input representation (raw pixels, VLM features, hybrid)
- Multi-scale prediction (predict at different temporal horizons)
- Auxiliary tasks (predict optical flow, depth, segmentation alongside future frames)

## What You Cannot Modify

- The evaluation metric definitions
- The SSv2 dataset or dataloader
- The base VLM (Qwen2.5-VL-7B) weights (but you can add adapters/heads)
- The video generator (LTX-Video) weights
- The 5-minute training budget

## Experiment Log Format

After each experiment, append to `autoresearch/experiments.jsonl`:
```json
{
  "id": "AR-001",
  "timestamp": "2026-03-15T22:00:00Z",
  "hypothesis": "What you tried and why",
  "architecture": "Brief description of changes",
  "cos_sim_model": 0.000,
  "cos_sim_copy": 0.000,
  "pixel_l1_model": 0.000,
  "pixel_l1_copy": 0.000,
  "beats_copy": false,
  "kept": false,
  "notes": "What you learned"
}
```

## Research Directions to Explore

Prioritized by likelihood of breaking through the wall:

1. **Auxiliary prediction targets** — Don't predict raw pixels. Predict optical flow, scene graphs, or action labels first. The VLM understands semantics, not pixels.

2. **Temporal curriculum** — Start by predicting 1 frame ahead, then 2, then 5. Current experiments jumped straight to multi-frame.

3. **Contrastive future prediction** — Don't generate the future frame. Instead, learn to distinguish "real next frame" from "wrong next frame" in embedding space. Avoids the pixel prediction problem entirely.

4. **VLM as verifier, not predictor** — Generate N candidate futures with LTX-Video (different prompts/seeds), then have VLM rank which is most plausible. This leverages the E3.8 finding (VLM understands generated video at 93%) without requiring VLM to predict.

5. **Hallucination as feature** — (See: Black Cygnets thesis) Instead of suppressing VLM confabulation, use it as a candidate future generator. Filter with perceptual metrics + physical plausibility checks.

6. **Adapter ensemble** — Multiple small adapters (10M each, per E3.8 finding) specialized for different prediction aspects, combined at inference.

## Philosophy

From Karpathy: "You are programming the program.md, not the code."
From Foresight: "Use each model for what it's trained for."
From today's SXSW insight: "Hallucinations are baby black swans."

The goal isn't to make VLM predict the future directly (we proved that fails). The goal is to find the *architecture* that translates VLM understanding into future-state representation — likely through intermediary representations, not raw pixels.
