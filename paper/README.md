# Foresight Paper

This directory contains the research paper documenting the Foresight experiments.

## Title

**Foresight: Can Video Prediction Ground Language Model Reasoning? A Negative Result and Benchmark Proposal**

## Abstract

We investigate whether vision-language models (VLMs) can benefit from generating pixel-level video predictions as part of their reasoning process. Through systematic experimentation, we find that:

1. VLMs cannot predict future states in their latent space
2. Video models can generate plausible continuations that VLMs understand (93% retention)
3. However, pixel-level verification (LPIPS) does not correlate with semantic correctness
4. Verification loops do not enable effective self-correction

We release our experimental framework as a benchmark for evaluating future video-language systems.

## Compilation

Requires a LaTeX distribution (MacTeX, TeX Live, or MiKTeX).

```bash
# Install MacTeX on macOS
brew install --cask mactex

# Compile (run twice for references)
pdflatex foresight.tex
pdflatex foresight.tex

# Or use latexmk
latexmk -pdf foresight.tex
```

Alternatively, upload `foresight.tex` to [Overleaf](https://overleaf.com).

## Key Citations

- Zhang et al. (2018) - LPIPS perceptual metric
- Goyal et al. (2017) - Something-Something v2 dataset
- Bardes et al. (2024) - V-JEPA latent video prediction
- Wang et al. (2024) - Qwen2-VL vision-language model
- Oquab et al. (2024) - DINOv2 visual features
