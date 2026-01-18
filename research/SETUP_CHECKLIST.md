# Pre-Experiment Setup Checklist

Setup tasks that must be completed **once** before spinning out experiment agents.

## Status

- [ ] **W&B Configuration** - Project created, API key available
- [ ] **Environment** - uv sync works, all dependencies install
- [ ] **Models Downloaded** - Qwen2.5-VL, LTX-Video cached locally
- [ ] **Datasets** - At least one dataset accessible for initial experiments
- [ ] **Hardware Verified** - GPU available, VRAM confirmed
- [ ] **Experiment Directories** - Created with proper structure

---

## 1. W&B Configuration

### Create Project
```bash
# Login (do this interactively)
wandb login

# Verify
wandb status
```

### Set Project Defaults
Create `configs/wandb/default.yaml`:
```yaml
project: foresight
entity: <your-wandb-username-or-team>
tags:
  - research
  - glp
settings:
  console: "wrap"
  quiet: false
```

### Environment Variables
Add to `.env` (git-ignored):
```bash
WANDB_API_KEY=<your-key>
WANDB_PROJECT=foresight
WANDB_ENTITY=<your-entity>
```

---

## 2. Environment Setup

### Verify Python Environment
```bash
# Should create .venv and install all packages
uv sync

# Verify imports
.venv/bin/python -c "import foresight_core; print('OK')"
```

### Install Additional Dependencies
Some experiments need extra packages:
```bash
uv add torch torchvision torchaudio
uv add transformers diffusers accelerate
uv add wandb lpips
uv add matplotlib seaborn  # for visualizations
uv add scikit-learn        # for t-SNE, probes
uv add jupyter             # for analysis notebooks
```

---

## 3. Model Downloads

### Qwen2.5-VL (Required for all experiments)
```bash
# ~15GB download
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# Verify
python -c "from transformers import Qwen2VLForConditionalGeneration; print('OK')"
```

### LTX-Video (Required for C2+)
```bash
# ~4GB download
huggingface-cli download Lightricks/LTX-Video

# Verify
python -c "from diffusers import LTXPipeline; print('OK')"
```

### Cache Location
Models download to `~/.cache/huggingface/`. Ensure sufficient disk space (~25GB).

To use a different cache location:
```bash
export HF_HOME=/path/to/cache
```

---

## 4. Dataset Setup

### Minimum Viable: Something-Something v2 Subset

For initial experiments, we need a small subset (~1000 videos).

```bash
# Create data directory (git-ignored)
mkdir -p data/something-something-v2

# Option A: Download from Hugging Face
# (Requires agreeing to terms at https://huggingface.co/datasets/HuggingFaceM4/something-something-v2)

# Option B: Manual download from https://developer.qualcomm.com/software/ai-datasets/something-something
# Extract to data/something-something-v2/
```

### Verify Dataset
```python
# Should list video files
import os
videos = os.listdir("data/something-something-v2/videos")
print(f"Found {len(videos)} videos")
```

### For Later: Full Datasets
- **COIN**: https://coin-dataset.github.io/
- **CrossTask**: https://github.com/DmZhukov/CrossTask

---

## 5. Hardware Verification

### Check GPU
```bash
nvidia-smi
```

### Verify VRAM
```python
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {gb:.1f} GB")
else:
    print("No GPU available")
```

### Minimum Requirements
| Experiment | Min VRAM | Recommended |
|------------|----------|-------------|
| C1, Q1, Q2 | 24GB | 40GB |
| C2, Q3 | 32GB | 40GB |
| C3, C4, Q4, Q5 | 40GB | 80GB |

---

## 6. Experiment Directory Structure

Create directories for each experiment:

```bash
# Create experiment result directories
for exp in c1-vlm-latent-sufficiency c2-adapter-bridging c3-future-prediction c4-pixel-verification q1-latent-alignment q2-information-preservation q3-temporal-coherence q4-training-data q5-prediction-horizon; do
    mkdir -p "research/experiments/${exp}/artifacts"
    cp research/validation/templates/results.yaml.template "research/experiments/${exp}/results.yaml"
done
```

---

## 7. Verify Everything Works

### Quick Smoke Test
```python
# smoke_test.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

print("Loading Qwen2.5-VL...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print(f"Model loaded on {model.device}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Quick inference test
from PIL import Image
import requests
url = "https://picsum.photos/224/224"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text="Describe this image.", images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
print("\n✓ Smoke test passed!")
```

---

## Post-Setup

Once all items are checked:

1. Update this file marking items complete
2. Commit the setup state
3. Sub-agents can now start experiments

### Starting Phase 1 Experiments

These can run in parallel (no dependencies):
- **C1**: `research/experiments/c1-vlm-latent-sufficiency.md`
- **Q1**: `research/experiments/q1-latent-alignment.md`
- **Q2**: `research/experiments/q2-information-preservation.md`
