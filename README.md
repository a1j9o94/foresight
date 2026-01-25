# Foresight

**Can AI benefit from "imagining" the future before making decisions?**

This research project explored whether AI systems could improve their reasoning by generating video predictions and checking them against reality—like how humans often visualize outcomes before acting.

**Result: Not yet.** Current models aren't capable of this, but we've documented what works, what doesn't, and propose tracking this as a benchmark for future AI systems.

---

## The Idea (Plain English)

When you're about to pour coffee, you might briefly imagine the liquid filling the cup. If you imagined it overflowing, you'd pour less. This "mental simulation" helps you make better decisions.

We asked: **Can AI do something similar?**

The plan was:
1. Show the AI an image (a cup, a ball, etc.)
2. Ask "What happens if I push this?"
3. Have it generate a video prediction of the outcome
4. Compare that prediction to what actually happens
5. Use the difference to improve future predictions

If this worked, AI systems could catch their own mistakes by noticing when their predictions look wrong—just like you'd notice if your mental image of pouring coffee showed it going sideways instead of down.

---

## What We Found

### Summary Table

| Phase | Question | Result | Key Finding |
|-------|----------|--------|-------------|
| **1. Reconstruction** | Can we decode images from AI's internal representation? | ✅ Passed | Hybrid approach (DINOv2 + VLM) preserves spatial info |
| **2. Bridging** | Can we connect the language model to a video generator? | ✅ Passed | Small 10M adapter works better than large 100M one |
| **3. Prediction** | Can the AI predict what happens next? | ❌ Failed | 7 architectures tested—none beat just copying the input |
| **4. Verification** | Does comparing predictions to reality help? | ❌ Failed | Perceptual similarity doesn't indicate correctness |

### Detailed Results

| Metric | What It Measures | Achieved | Needed | Status |
|--------|------------------|----------|--------|--------|
| Spatial IoU | Position accuracy | 0.837 | > 0.60 | ✅ |
| LPIPS | Visual quality | 0.162 | < 0.35 | ✅ |
| Prediction vs Copy | Can it predict better than copying? | -4.5% | > 0% | ❌ |
| LPIPS-Correctness correlation | Does visual error indicate wrong answer? | 0.106 | > 0.30 | ❌ |
| Self-correction rate | Can it fix mistakes with feedback? | 7.4% | > 15% | ❌ |

### The Key Failures

**1. VLMs Can't Predict the Future**

We tried 7 different approaches to make the language model predict what happens next in a video:
- Single frame input
- Multiple frames input
- Temporal transformers
- Contrastive learning
- Pixel-level feedback
- Fine-tuning the model

All of them performed worse than simply copying the current frame as the "prediction." The language model understands what's in an image, but it cannot predict what will change.

**2. Visual Similarity ≠ Semantic Correctness**

Even when we used a video model to generate predictions (which looked reasonable), comparing them to reality using perceptual metrics (LPIPS) didn't help. Surprisingly, **wrong predictions often looked MORE similar to reality than correct ones**.

This means you can't use "does it look right?" to catch mistakes—the visual appearance doesn't indicate whether the prediction is semantically correct.

---

## What Did Work

Despite the negative results, we made useful discoveries:

| Finding | Why It Matters |
|---------|----------------|
| **Hybrid encoder** (DINOv2 + VLM) preserves spatial information | Solves the problem of VLMs losing position data |
| **VLMs understand generated video** (93% retention) | Video models generate content VLMs can reason about |
| **Small adapters work** (10M beats 100M) | Efficient bridging between models is possible |
| **Video Predicts → VLM Describes** works | Use each model for what it's good at |

---

## Benchmark Proposal: VideoReason

We're releasing this as a benchmark to track when this approach becomes viable. As video models improve, these capabilities may emerge.

**Tasks to track:**
1. Future frame prediction accuracy
2. Action understanding in generated vs real video
3. Verification metric correlation with correctness
4. Self-correction success rate

**Why track this?** Video generation is improving rapidly. The capabilities we found lacking in 2026 may emerge in future systems. A standardized benchmark helps identify when "visual imagination" becomes useful for AI reasoning.

---

## Prerequisites

To reproduce the experiments, you'll need accounts with these services:

| Service | Purpose | Sign Up |
|---------|---------|---------|
| **Modal** | GPU compute for experiments (A100s) | [modal.com](https://modal.com) |
| **Hugging Face** | Model downloads (Qwen2.5-VL, LTX-Video) | [huggingface.co](https://huggingface.co) |
| **Weights & Biases** | Experiment tracking and logging | [wandb.ai](https://wandb.ai) |

### Dataset: Something-Something v2

The experiments use the **Something-Something v2** dataset for action prediction. This must be downloaded manually:

1. Go to [Qualcomm AI Datasets](https://developer.qualcomm.com/software/ai-datasets/something-something)
2. Request access and download the dataset
3. Extract to `data/something-something-v2/`

The dataset contains ~220K videos of humans performing 174 different actions (pushing, pulling, dropping, etc.).

---

## Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/a1j9o94/foresight.git
cd foresight
uv sync

# 2. Copy environment template
cp .env.example .env
# Edit .env with your API keys (WANDB_API_KEY, HF_TOKEN)

# 3. Configure Modal secrets
modal secret create wandb-api-key WANDB_API_KEY=<your-wandb-key>
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>

# 4. Download models (first time only, ~20GB)
uv run modal run infra/modal/app.py::download_models

# 5. Verify setup
uv run modal run infra/modal/app.py::smoke_test
```

## Quick Start

```bash
# Run demo locally (shows the working parts)
cd demo/backend && uvicorn main:app --reload --port 8000
cd demo/frontend && bun run dev
# Open http://localhost:3000

# Run experiments on Modal GPUs
uv run modal run infra/modal/app.py::run_experiment --experiment-id <id>

# Test experiment harness without GPU
uv run modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency --stub-mode
```

## Live Demo

- **Frontend:** https://foresight-demo-kappa.vercel.app
- **Backend:** https://foresight-demo.fly.dev

---

## Project Structure

```
foresight/
├── paper/              # Research paper (LaTeX)
├── research/           # Experiment results & findings
│   ├── FINDINGS.md     # Summary of all results
│   └── experiments/    # Per-experiment details
├── infra/modal/        # GPU experiment infrastructure
│   └── handlers/       # Experiment implementations
├── demo/               # Live demo (React + FastAPI)
├── packages/           # Modular code packages
└── configs/            # Model/training configs
```

## Tools & Models Used

| Component | Tool | Notes |
|-----------|------|-------|
| Vision-Language Model | [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Frozen, used for encoding |
| Visual Encoder | [DINOv2-ViT-L](https://huggingface.co/facebook/dinov2-large) | Spatial feature extraction |
| Video Generation | [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | Real-time video synthesis |
| Perceptual Metric | [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | Learned perceptual similarity |
| GPU Compute | [Modal](https://modal.com) | A100-80GB for experiments |
| Experiment Tracking | [Weights & Biases](https://wandb.ai) | Metrics and artifacts |
| Package Manager | [uv](https://github.com/astral-sh/uv) | Fast Python packaging |
| Frontend | React + TypeScript + Bun | Demo UI |
| Backend | FastAPI | Demo API |

## Documentation

- **[Research Findings](./research/FINDINGS.md)** - Detailed experiment results
- **[Paper](./paper/)** - Full writeup with citations
- **[CLAUDE.md](./CLAUDE.md)** - Development guide

## Citation

If you use this work, please cite:

```bibtex
@misc{obleton2026foresight,
  title={Foresight: Can Video Prediction Ground Language Model Reasoning? A Negative Result and Benchmark Proposal},
  author={Adrian Obleton},
  year={2026},
  url={https://github.com/a1j9o94/foresight},
  note={Research prototype and benchmark}
}
```

## License

Research prototype - released for academic use. See paper for full methodology.
