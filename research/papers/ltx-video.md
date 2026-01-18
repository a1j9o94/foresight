# LTX-Video: Realtime Video Latent Diffusion

**Authors:** Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, Ofir Bibi (Lightricks)
**Year:** 2024
**Venue:** arXiv preprint
**Links:** [Paper](https://arxiv.org/abs/2501.00103) | [Code](https://github.com/Lightricks/LTX-Video) | [Model](https://huggingface.co/Lightricks/LTX-Video)

## Summary

LTX-Video is the first DiT-based video generation model capable of faster-than-real-time generation. It produces 5 seconds of 24fps video at 768x512 resolution in just 2 seconds on an H100 GPU. The key innovation is a holistic approach to latent diffusion that tightly integrates the Video-VAE and denoising transformer, achieving an unprecedented 1:192 compression ratio while maintaining visual quality.

The model achieves this efficiency through two primary innovations: (1) an extremely high compression VAE with 32x32x8 spatiotemporal downsampling and 128 latent channels, and (2) a "denoising decoder" that performs the final diffusion step during VAE decoding, recovering high-frequency details in pixel space rather than latent space.

## Key Technical Insights

- **Denoising Decoder**: The VAE decoder is conditioned on diffusion timestep and performs the final denoising step alongside latent-to-pixel conversion. This recovers fine details lost to high compression without requiring a separate upsampling module.

- **Extreme Compression**: 1:192 compression ratio (vs typical 1:96) enabled by moving patchification from transformer input to VAE input. This results in 8192 pixels per token (4x typical ratio).

- **128-Channel Latents**: Unlike competing models with 16 channels, LTX-Video uses 128 latent channels to preserve information despite aggressive spatial compression.

- **Per-Token Timesteps**: For image conditioning, diffusion timesteps are defined per-token rather than globally, enabling conditioning on any frame (not just the first).

## Architecture/Method

```
Input Video/Text
      |
      v
+------------------+
|   Text Encoder   |  (T5-XXL, frozen)
|   (T5-XXL)       |
+------------------+
      |
      v (cross-attention)
+------------------+     +------------------+
|   Video VAE      |     |    DiT Model     |
|   Encoder        | --> |   (1.9B params)  |
| (32x32x8 comp)   |     |   28 blocks      |
| (128 channels)   |     |   dim=2048       |
+------------------+     +------------------+
      ^                          |
      |                          v
      |                  +------------------+
      |                  | Denoising VAE    |
      +------------------| Decoder          |
                         | (timestep-cond)  |
                         +------------------+
                                 |
                                 v
                           Output Video
```

### VAE Architecture

| Component | Specification |
|-----------|---------------|
| Encoder | Causal 3D convolutions |
| Spatial compression | 32x |
| Temporal compression | 8x |
| Latent channels | 128 |
| Total compression | 1:192 |

### Transformer Architecture

| Component | Specification |
|-----------|---------------|
| Parameters | 1.9B |
| Hidden dim | 2048 |
| Blocks | 28 |
| Attention | Self + Cross (not self-only) |
| Positional encoding | RoPE with exponential frequency spacing |
| Normalization | QK-RMSNorm |

### Denoising Decoder

The decoder receives noisy latents z_t and timestep t, outputting clean pixels:

```
x_0 = D(z_t, t) = D((1-t)*z_0 + t*epsilon, t)
```

- Trained with noise levels in [0, 0.2] (final diffusion timestep range)
- Uses adaptive normalization layers for timestep conditioning
- Multi-layer noise injection for high-frequency detail diversity

## Results

### Speed Benchmarks

| Resolution | Frames | Time (H100) | FPS |
|------------|--------|-------------|-----|
| 768x512 | 121 (5s@24fps) | 2 seconds | >60 generated/sec |
| 1216x704 | - | Real-time | 30 FPS |

### Quality Metrics (from paper)

| Metric | LTX-Video | Notes |
|--------|-----------|-------|
| User preference | 67% | vs non-denoising decoder |
| Motion quality | Significant improvement | High-motion scenes benefit most |

### Model Variants (Current)

| Variant | Parameters | Use Case |
|---------|------------|----------|
| 2B (original) | 1.9B | Fast prototyping |
| 13B dev | 13B | Highest quality |
| 13B distilled | 13B | 15x faster, 8 steps |
| 2B distilled | 2B | Lightweight |

## Relevance to Foresight

### Direct Relevance

LTX-Video is the primary video decoder candidate for Foresight due to:

1. **Speed**: Real-time generation enables <2 second video prediction, meeting our 3-second total latency target
2. **LoRA Support**: Official trainer supports LoRA fine-tuning, perfect for training our conditioning adapter
3. **Latent Interface**: The model accepts latent inputs, enabling direct conditioning from VLM outputs
4. **Diffusers Integration**: Native HuggingFace support simplifies integration

### Key Integration Points

**Conditioning via Latents:**
```python
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

# Our conditioning adapter would project VLM latents here
condition = LTXVideoCondition(video=encoded_latents, frame_index=0)
```

**LoRA Training Path:**
- Use [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) for conditioning adapter
- Train projection layer: VLM latent space -> LTX latent space
- IC-LoRA adapters demonstrate custom conditioning is feasible

**Latent Space Compatibility:**
- LTX uses 128-channel latents at 32x32x8 compression
- Our conditioning adapter must project VLM hidden states (2048-dim for Qwen2.5-VL-7B) to this space
- Per-token timestep conditioning enables frame-level control

### Critical Consideration: Latent Alignment

The denoising decoder expects latents at specific noise levels. For VLM conditioning:
- Option A: Train adapter to produce "clean" latents (t=0.05 for decoder)
- Option B: Train adapter to produce intermediate latents and run partial diffusion
- Option C: Use image conditioning path with projected VLM features

### Contrast with Alternatives

| Model | Speed | Quality | Conditioning Flexibility |
|-------|-------|---------|-------------------------|
| LTX-Video | Real-time | Good | High (per-token timesteps) |
| HunyuanVideo | 75s/clip | Excellent | Standard |
| CogVideoX | ~30s/clip | Very good | Standard |

LTX-Video's speed makes it ideal for iterative prediction-verification loops.

## Open Questions

1. **Latent Alignment**: How do we align VLM latent representations with LTX's 128-channel latent space? The high channel count (128 vs typical 16) may help or complicate alignment.

2. **Denoising Decoder Interaction**: If we condition with VLM latents, do we bypass the denoising decoder benefits? May need to inject noise and let decoder denoise.

3. **Temporal Coherence**: LTX uses causal encoding (first frame separate). How does this affect multi-step prediction where we condition on predicted frames?

4. **Quality vs Speed Tradeoff**: Should we use 2B (faster) or 13B-distilled (better quality) for prediction? Verification step may compensate for 2B quality.

5. **IC-LoRA Feasibility**: Can we train an IC-LoRA that accepts VLM hidden states instead of depth/pose maps?

## Code/Implementation Notes

### Installation
```bash
pip install diffusers transformers accelerate
huggingface-cli download Lightricks/LTX-Video
```

### Basic Inference
```python
from diffusers import LTXPipeline
import torch

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

video = pipe(
    prompt="A cat walking across a table",
    num_frames=121,
    height=512,
    width=768,
).frames[0]
```

### Latent-Level Access
```python
# For Foresight integration, we need latent-level control:
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline

pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.8-dev",
    torch_dtype=torch.bfloat16
)

# Access VAE for encoding/decoding
vae = pipe.vae

# Encode conditioning image to latent
with torch.no_grad():
    latents = vae.encode(image_tensor).latent_dist.sample()
```

### LoRA Training
The [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) supports:
- LoRA fine-tuning (efficient, our likely path)
- Full fine-tuning (resource intensive)
- IC-LoRA training (for custom conditioning)

## Citation

```bibtex
@article{hacohen2024ltxvideo,
  title={LTX-Video: Realtime Video Latent Diffusion},
  author={HaCohen, Yoav and Chiprut, Nisan and Brazowski, Benny and Shalem, Daniel and Moshe, Dudu and Richardson, Eitan and Levin, Eran and Shiran, Guy and Zabari, Nir and Gordon, Ori and Panet, Poriya and Weissbuch, Sapir and Kulikov, Victor and Bitterman, Yaki and Melumian, Zeev and Bibi, Ofir},
  journal={arXiv preprint arXiv:2501.00103},
  year={2024}
}
```

## Review Status

- [x] Read abstract
- [x] Read full paper
- [x] Reviewed code
- [x] Summarized key insights
- [x] Connected to hypothesis

---

## Appendix: Training Details from Paper

### VAE Training Losses
- MSE pixel reconstruction
- Video-DWT (L1 on discrete wavelet transform)
- Perceptual (LPIPS)
- Reconstruction-GAN (novel: discriminator sees both real and fake concatenated)

### Transformer Training
- Optimizer: AdamW
- Multi-resolution training (diverse resolution/duration combinations)
- Stochastic token dropping (0-20%) for consistent token counts
- Image data mixed with video for concept diversity

### Key Hyperparameters
- RoPE: Exponential frequency spacing (outperforms inverse-exponential)
- Decoder timestep: t=0.05 at inference (not t=0.0)
- Noise injection: Multi-layer in decoder for detail diversity
