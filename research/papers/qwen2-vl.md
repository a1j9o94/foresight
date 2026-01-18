# Qwen2-VL / Qwen2.5-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

**Authors:** Qwen Team, Alibaba Group
**Year:** 2024 (Qwen2-VL), 2025 (Qwen2.5-VL)
**Venue:** arXiv
**Links:** [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191) | [Qwen2.5-VL Paper](https://arxiv.org/abs/2502.13923) | [Code](https://github.com/QwenLM/Qwen2-VL) | [Qwen2-VL Blog](https://qwenlm.github.io/blog/qwen2-vl/) | [Qwen2.5-VL Blog](https://qwenlm.github.io/blog/qwen2.5-vl/)

## Summary

Qwen2-VL and its successor Qwen2.5-VL are state-of-the-art vision-language models from Alibaba's Qwen team that introduce two key innovations: **Naive Dynamic Resolution** for processing images/videos at any resolution, and **Multimodal Rotary Position Embedding (M-RoPE)** for unified handling of 1D text, 2D images, and 3D video positional information.

The models combine a 675M parameter Vision Transformer (ViT) with Qwen2/Qwen2.5 language models (2B/3B, 7B/8B, 72B variants). Unlike traditional VLMs that resize inputs to fixed resolutions, Qwen2-VL dynamically converts images into variable-length visual token sequences, preserving native resolution information. The 72B variant achieves performance comparable to GPT-4o and Claude 3.5 Sonnet on multimodal benchmarks.

## Key Technical Insights

- **Naive Dynamic Resolution**: Images are processed at their native resolution and converted to variable-length token sequences (4-16,384 tokens per image), eliminating information loss from forced resizing. A 224x224 image becomes ~66 tokens after 2x2 merging.

- **M-RoPE (Multimodal Rotary Position Embedding)**: Decomposes rotary embeddings into three components (temporal, height, width) to simultaneously encode text (1D), image (2D), and video (3D) positional information within the same sequence.

- **Efficient Token Compression**: After ViT encoding, an MLP layer compresses adjacent 2x2 visual tokens into a single token, reducing computational load while preserving spatial relationships.

- **Frozen ViT Across Scales**: The same 675M parameter ViT is used across all LLM sizes (2B-72B), ensuring consistent visual encoding quality regardless of language model scale.

## Architecture/Method

### Overall Architecture

```
Input (Image/Video)
       |
       v
+------------------+
|  Vision Encoder  |  <-- 675M param ViT with 2D-RoPE (Qwen2-VL)
|  (ViT)           |      or Window Attention + SwiGLU + RMSNorm (Qwen2.5-VL)
+------------------+
       |
       v  [1536-dim embeddings per patch]
+------------------+
|  2x2 Token       |  <-- MLP compresses 4 adjacent tokens -> 1 token
|  Merger (MLP)    |
+------------------+
       |
       v  [<vision_start>..visual_tokens..<vision_end>]
+------------------+
|  Qwen2 LLM       |  <-- 2B/7B/72B parameters
|  Backbone        |      with M-RoPE position encoding
+------------------+
       |
       v
   Text Output
```

### Vision Encoder Details

| Component | Qwen2-VL | Qwen2.5-VL |
|-----------|----------|------------|
| Parameters | ~675M | ~675M |
| Patch Size | 14 | 14 |
| Temporal Patch Size | 2 | 2 |
| Merge Size | 2x2 | 2x2 |
| Position Encoding | 2D-RoPE | 2D-RoPE |
| Attention | Full Attention | 4 Full + Window Attention (8x8 max) |
| Normalization | LayerNorm | RMSNorm |
| Activation | GELU | SwiGLU |
| Output Embedding Dim | 1536 | 1536 |

### Token Flow

1. **Image Input**: Variable resolution (e.g., 448x672)
2. **Patch Embedding**: Divided into 14x14 patches with 2D-RoPE
3. **ViT Processing**: 675M param transformer, outputs 1536-dim per patch
4. **Token Merging**: MLP compresses 2x2 adjacent patches -> 1 token
5. **LLM Integration**: Visual tokens wrapped in `<vision_start>...<vision_end>` special tokens

### M-RoPE Mechanism

M-RoPE decomposes rotary position embeddings into three components:
- **Temporal**: Constant for images, increments per frame for video
- **Height**: Varies with vertical position in image/frame
- **Width**: Varies with horizontal position in image/frame

For text, all three components share identical position IDs. This unified approach enables:
- Single model for text + image + video
- Sequence length extrapolation (smaller position IDs enable longer context)
- Native understanding of spatial relationships

### Dynamic Resolution Processing

```python
# Token calculation example
patch_size = 14
temporal_patch_size = 2
merge_size = 2

# For a 224x224 image:
# Patches: (224/14) x (224/14) = 16 x 16 = 256 patches
# After 2x2 merge: 256 / 4 = 64 tokens
# Plus start/end tokens: 66 total tokens

# Configurable range:
min_pixels = 256 * 28 * 28   # ~256 tokens minimum
max_pixels = 1280 * 28 * 28  # ~1280 tokens maximum
```

## Extracting Intermediate Latents

### Method 1: Direct Vision Encoder Access

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Process image
inputs = processor.image_processor(images=img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)
grid_thw = inputs["image_grid_thw"].to(device)

# Extract visual embeddings directly from ViT
with torch.no_grad():
    vision_outputs = model.visual(pixel_values, grid_thw)
    visual_embeds = vision_outputs  # Shape: [num_tokens, 1536]
```

### Method 2: Hidden States from Full Model

```python
# Get hidden states from all layers
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True
)

# Access hidden states
# hidden_states[0]: embedding layer output
# hidden_states[1:-1]: intermediate layer outputs
# hidden_states[-1]: final layer output
all_hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
```

### Method 3: Attention Weights

```python
outputs = model(
    **inputs,
    output_attentions=True,
    return_dict=True
)

# Access attention patterns
attention_weights = outputs.attentions  # Tuple of (batch, heads, seq_len, seq_len)
```

### Available Representations

| Extraction Point | Shape | Use Case |
|-----------------|-------|----------|
| ViT output (pre-merge) | [N_patches, 1536] | Fine-grained spatial features |
| ViT output (post-merge) | [N_tokens, 1536] | Compressed visual tokens |
| LLM hidden states | [batch, seq_len, hidden_size] | Multimodal fused features |
| Attention weights | [batch, heads, seq, seq] | Cross-modal attention patterns |

## Results

### Key Quantitative Results (Qwen2-VL-72B)

| Benchmark | Qwen2-VL-72B | GPT-4o | Previous SoTA |
|-----------|--------------|--------|---------------|
| DocVQA | **96.5** | 92.8 | 94.1 |
| InfoVQA | 84.5 | - | - |
| OCRBench | **877** | 736 | - |
| ChartQA | 91.7 | 85.7 | - |
| MathVista | 70.5 | 63.8 | - |

### Video Understanding

| Benchmark | Qwen2-VL-72B | Notes |
|-----------|--------------|-------|
| MVBench | Competitive | General video understanding |
| EgoSchema | Strong | Egocentric video reasoning |
| Video-MME | Strong | Long video comprehension |

## Relevance to Foresight

### Direct Relevance

1. **Visual Latent Extraction**: The 1536-dim visual embeddings from the ViT can potentially serve as conditioning signals for video decoders. We can extract:
   - Pre-merge patch embeddings (high spatial resolution)
   - Post-merge tokens (compressed but semantically rich)
   - LLM intermediate hidden states (multimodal fused)

2. **Video Understanding**: Native video support with temporal M-RoPE encoding provides frame-aware representations that preserve temporal relationships - useful for predicting future frames.

3. **Dynamic Resolution**: The variable token count matches our need for processing diverse video resolutions without information loss.

### Inspiration

1. **M-RoPE for Video Generation**: The decomposed positional encoding (temporal + spatial) could inform how we encode conditioning information for video diffusion models.

2. **Token Merging Strategy**: The 2x2 spatial merging could be adapted for our conditioning adapter to reduce computational load while preserving semantic content.

3. **Frozen ViT Approach**: Keeping the vision encoder frozen while training adapters aligns with our GLP architecture philosophy.

### Contrast with Foresight Approach

| Aspect | Qwen2-VL | Foresight |
|--------|----------|-----------|
| Output | Text | Video pixels |
| Visual encoding | ViT -> LLM tokens | ViT -> video decoder latents |
| Training | Next token prediction | Reconstruction + alignment |
| Verification | N/A | LPIPS vs actual outcomes |

### Key Integration Questions

1. **Latent Compatibility**: Do the 1536-dim ViT embeddings contain sufficient spatial information to condition LTX-Video/HunyuanVideo, or do we need the pre-merge representations?

2. **Temporal Encoding**: Can we leverage M-RoPE's temporal component to help the video decoder understand the time dimension of predictions?

3. **Query Token Design**: Should our learned query tokens operate in the post-merge token space (smaller, more semantic) or pre-merge patch space (larger, more spatial)?

## Open Questions

- What is the exact architecture of the ViT (number of layers, attention heads)?
- How does the MLP merger preserve spatial relationships during 2x2 compression?
- Can we fine-tune just the vision encoder for better video prediction conditioning?
- What is the computational cost of extracting intermediate representations during inference?
- How do the Window Attention patterns in Qwen2.5-VL affect the spatial coherence of extracted features?

## Code/Implementation Notes

### Installation

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8  # For video support
pip install flash-attn --no-build-isolation  # Recommended for efficiency
```

### Model Loading

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Standard loading
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# With Flash Attention 2 (recommended)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

### Key Classes

- `Qwen2VLForConditionalGeneration`: Main model class
- `Qwen2VLModel`: Base model without LM head
- `Qwen2VLTextModel`: Text-only backbone
- `Qwen2VLImageProcessor`: Image preprocessing
- `Qwen2VLVideoProcessor`: Video preprocessing
- `Qwen2VLProcessor`: Combined processor

## Citation

```bibtex
@article{wang2024qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{qwen2025qwen25vl,
  title={Qwen2.5-VL Technical Report},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}
```

## Review Status

- [x] Read abstract
- [x] Read full paper
- [x] Reviewed code/documentation
- [x] Summarized key insights
- [x] Connected to hypothesis
