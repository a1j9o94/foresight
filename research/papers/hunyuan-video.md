# HunyuanVideo: A Systematic Framework For Large Video Generative Models

**Authors:** Weijie Kong, Qi Tian, Zijian Zhang, et al. (52 authors from Tencent Hunyuan)
**Year:** 2024
**Venue:** arXiv preprint
**Links:** [Paper](https://arxiv.org/abs/2412.03603) | [Code](https://github.com/Tencent-Hunyuan/HunyuanVideo) | [Hugging Face](https://huggingface.co/tencent/HunyuanVideo)

## Summary

HunyuanVideo is the largest open-source video generation model at 13 billion parameters, achieving performance comparable to or surpassing closed-source models like Runway Gen-3 and Luma 1.6. The paper presents a systematic framework covering data curation, architecture design, progressive scaling, and distributed training infrastructure.

The key innovation is a "Dual-stream to Single-stream" Transformer architecture that processes video and text modalities independently before fusing them, combined with an MLLM-based text encoder (rather than CLIP/T5) for better instruction following. The model uses Flow Matching for training and a 3D Causal VAE for efficient spatiotemporal compression.

## Key Technical Insights

- **MLLM Text Encoder**: Uses a decoder-only multimodal LLM instead of CLIP/T5, providing better image-text alignment and complex reasoning. A bidirectional token refiner addresses the limitation of causal attention for diffusion conditioning.
- **Dual-to-Single Stream**: Video and text tokens are processed independently in early layers (learning modality-specific modulations), then concatenated for full-attention fusion in later layers.
- **Efficient Scaling**: Discovered that naive scaling of Flow Matching transformers is inefficient. Their scaling laws reduce computational requirements by up to 5x while maintaining quality.
- **3D VAE with CausalConv3D**: Achieves superior reconstruction (PSNR 33.14 vs 31.73 for CogVideoX) with 4x temporal, 8x spatial, and 16-channel compression.

## Architecture/Method

```
Input Video/Image
       |
       v
+------------------+
| 3D Causal VAE    | --> Latent: (T/4+1) x 16 x (H/8) x (W/8)
| (ct=4, cs=8, C=16)|
+------------------+
       |
       v
+------------------+     +------------------+
| Video Tokens     |     | Text Tokens      |
+------------------+     | (MLLM + Refiner) |
       |                 +------------------+
       |                        |
       v                        v
+--------------------------------------+
| DUAL-STREAM PHASE                    |
| Independent Transformer blocks       |
| (Modality-specific modulation)       |
+--------------------------------------+
       |
       v
+--------------------------------------+
| SINGLE-STREAM PHASE                  |
| Concatenated tokens + Full Attention |
| (Multimodal fusion)                  |
+--------------------------------------+
       |
       v
+------------------+
| 3D VAE Decoder   |
+------------------+
       |
       v
Output Video
```

**Flow Matching Training**: Velocity prediction loss L = E[||v_t - u_t||^2], where the model predicts flow velocity for linear interpolation from noise to data.

**3D RoPE**: Extends Rotary Position Embedding to 3D with channel partitioning (d_t=16, d_h=56, d_w=56) for temporal, height, and width coordinates.

**Progressive Training**:
1. 256px image pre-training
2. Mixed-scale (256px + 512px) image training
3. Low-resolution short video
4. Low-resolution long video
5. High-resolution long video with ~1M fine-tuning samples

## Results

Professional evaluation (1,533 prompts, 60 evaluators):

| Metric | HunyuanVideo | Best Baseline |
|--------|--------------|---------------|
| Motion Quality | 66.5% | 61.7% |
| Text Alignment | 61.8% | 62.6% |
| Visual Quality | 95.7% | 97.7% |
| Overall Rank #1 | 41.3% | 37.7% |

VAE Reconstruction (ImageNet 256x256 PSNR):

| Model | PSNR |
|-------|------|
| HunyuanVideo VAE | 33.14 |
| CogVideoX-1.5 | 31.73 |
| FLUX-VAE | 32.70 |

## Compute Requirements

**Original HunyuanVideo (13B)**:
- VRAM: 24-48GB recommended (can run on 24GB with offloading)
- Inference: ~75 seconds for 5-second 720p video on RTX 4090

**HunyuanVideo 1.5 (8.3B, optimized)**:
- VRAM: 14GB minimum with offloading; 8GB possible with GGUF quantization
- Inference: ~75 seconds on RTX 4090 (with step distillation)
- SSTA mechanism: 1.87x speedup vs FlashAttention-3
- MagCache: Additional 1.7x speedup

**Guidance Distillation**: ~1.9x acceleration by distilling classifier-free guidance

## Relevance to Foresight

**Direct relevance - When to use HunyuanVideo over LTX-Video:**

| Criterion | HunyuanVideo | LTX-Video |
|-----------|--------------|-----------|
| Generation Speed | 3-4x real-time | 6-7x real-time |
| Quality | Higher (cinematic) | Good (lower detail) |
| Parameters | 13B (v1) / 8.3B (v1.5) | ~3.2B |
| VRAM (practical) | 14-24GB | 8-12GB |
| Best Use Case | Final quality, multi-person scenes | Fast iteration, prototyping |

**Recommendation for Foresight**:
- **LTX-Video**: Use during development, rapid experimentation, and real-time reasoning loops where <2 second latency is critical
- **HunyuanVideo 1.5**: Use for production quality predictions, verification tasks where accuracy matters more than speed, and scenarios with complex multi-object dynamics

**Inspiration for our approach:**
- The MLLM text encoder approach aligns well with our Qwen2.5-VL backbone - we could leverage similar conditioning strategies
- Their dual-stream fusion could inform our conditioning adapter design (VLM latents -> video decoder)
- Flow Matching training is directly applicable to our GLP objective

**Contrast with Foresight**:
- HunyuanVideo optimizes for creative content generation; we optimize for predictive accuracy
- They focus on text-to-video; we focus on state prediction from visual context
- Our verification loop (predicted vs actual) is novel relative to their approach

## Open Questions

- How does HunyuanVideo's motion quality advantage translate to physical prediction tasks?
- Can the step-distilled version maintain prediction accuracy for subtle physical dynamics?
- Is the MLLM encoder's "instruction following" beneficial for conditional prediction, or overkill?
- Trade-off between HunyuanVideo's quality and LTX's speed for different verification scenarios?

## Code/Implementation Notes

- Official code: [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- HunyuanVideo 1.5 (recommended): [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- Diffusers integration: `from diffusers import HunyuanVideoPipeline`
- ComfyUI wrapper available for low-VRAM setups (8GB with temporal tiling)
- LoRA training code available for customization
- GGUF quantization available for consumer GPUs

**For Foresight integration**:
- Use HunyuanVideo 1.5 for production (8.3B, better efficiency)
- Consider step-distilled variant for speed-critical paths
- LoRA fine-tuning aligns with our training strategy (~10-50M trainable params)

## Citation

```bibtex
@article{kong2024hunyuanvideo,
  title={HunyuanVideo: A Systematic Framework For Large Video Generative Models},
  author={Kong, Weijie and Tian, Qi and Zhang, Zijian and Min, Rox and others},
  journal={arXiv preprint arXiv:2412.03603},
  year={2024}
}
```

## Review Status

- [x] Read abstract
- [x] Read full paper
- [x] Reviewed code repository
- [x] Summarized key insights
- [x] Connected to Foresight hypothesis
