"""Modal app for Foresight GPU experiments.

Usage:
    # Run a specific experiment
    modal run infra/modal/app.py::run_experiment --experiment-id c1-vlm-latent-sufficiency

    # Deploy as persistent endpoint (for longer experiments)
    modal deploy infra/modal/app.py

    # Run smoke test
    modal run infra/modal/app.py::smoke_test
"""

import os
from pathlib import Path

import modal

# Define the Modal app
app = modal.App("foresight")

# GPU image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # Required for cv2
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.45.0",  # Required for Qwen2_5_VLForConditionalGeneration
        "diffusers>=0.27.0",
        "accelerate>=0.27.0",
        # Experiment tracking
        "wandb>=0.15.0",
        # Metrics
        "lpips>=0.1.4",
        "scipy",
        # Utilities
        "pyyaml",
        "pydantic>=2.0.0",
        "numpy",
        "pillow",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "opencv-python-headless",  # For edge detection in Q2.5
        # Flash attention (optional, for speed)
        # "flash-attn",  # Uncomment if needed, requires specific CUDA version
    )
    .pip_install(
        # Install from GitHub for latest
        "qwen-vl-utils",
        "sentencepiece",  # Required for T5 tokenizer in LTX-Video
    )
    # Add the runner module to the image
    .add_local_dir(
        Path(__file__).parent / "runner",
        remote_path="/root/runner",
    )
    # Add the handlers module to the image
    .add_local_dir(
        Path(__file__).parent / "handlers",
        remote_path="/root/handlers",
    )
    # Add research_plan.yaml for experiment configuration
    .add_local_file(
        Path(__file__).parent.parent.parent / "research" / "research_plan.yaml",
        remote_path="/research/research_plan.yaml",
    )
)

# Volume for persisting results and caching models
results_volume = modal.Volume.from_name("foresight-results", create_if_missing=True)
model_cache = modal.Volume.from_name("foresight-model-cache", create_if_missing=True)
datasets_volume = modal.Volume.from_name("foresight-datasets", create_if_missing=True)

# Secrets for W&B
wandb_secret = modal.Secret.from_name("wandb-api-key", required_keys=["WANDB_API_KEY"])


@app.function(
    image=gpu_image,
    gpu="A100",  # or "A10G" for cheaper, "H100" for faster
    timeout=3600 * 4,  # 4 hours max
    volumes={
        "/results": results_volume,
        "/model-cache": model_cache,
    },
    secrets=[wandb_secret],
)
def smoke_test():
    """Quick test to verify GPU setup works."""
    import torch

    print("=" * 60)
    print("FORESIGHT SMOKE TEST")
    print("=" * 60)

    # Check GPU
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check imports
    print("\nChecking imports...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("  - transformers: OK")
    from diffusers import DiffusionPipeline

    print("  - diffusers: OK")
    import wandb

    print("  - wandb: OK")
    import lpips

    print("  - lpips: OK")

    # Quick model load test (just config, not weights)
    print("\nChecking model configs...")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    print(f"  - Qwen2.5-VL config: OK (hidden_size={config.hidden_size})")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)

    return {"status": "ok", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=3600 * 8,  # 8 hours max for experiments
    volumes={
        "/results": results_volume,
        "/model-cache": model_cache,
        "/datasets": datasets_volume,
    },
    secrets=[wandb_secret],
)
def run_experiment(experiment_id: str, sub_experiment: str | None = None, stub_mode: bool = False):
    """Run a Foresight experiment on GPU.

    Args:
        experiment_id: Experiment ID (e.g., 'c1-vlm-latent-sufficiency')
        sub_experiment: Optional specific sub-experiment (e.g., 'e1_2')
        stub_mode: If True, run with stub handlers to test the harness
    """
    # Set up environment - cache all models to persistent volume
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["TORCH_HOME"] = "/model-cache/torch"  # DINOv2 uses torch.hub

    # Import runner (needs to be inside function for Modal serialization)
    import sys
    sys.path.insert(0, "/root")

    from runner import ExperimentRunner, create_stub_handlers

    print(f"Starting experiment: {experiment_id}")
    if sub_experiment:
        print(f"Sub-experiment: {sub_experiment}")
    if stub_mode:
        print("Running in STUB MODE (testing harness only)")

    # Create runner
    runner = ExperimentRunner(
        experiment_id=experiment_id,
        results_dir="/results",
        wandb_project="foresight",
    )

    # Initialize W&B
    runner.init_wandb()

    # Register handlers
    if stub_mode:
        # Use stub handlers for testing
        for sub_exp_id, handler in create_stub_handlers(experiment_id).items():
            runner.register_handler(sub_exp_id, handler)
    else:
        # Import and register real handlers
        # TODO: Implement real handlers in separate modules
        handlers = _get_experiment_handlers(experiment_id)
        for sub_exp_id, handler in handlers.items():
            runner.register_handler(sub_exp_id, handler)

    # Run experiments
    if sub_experiment:
        results = {"single": runner.run_sub_experiment(sub_experiment)}
    else:
        results = runner.run_all()

    # Cleanup
    runner.finish()

    # Commit volume changes
    results_volume.commit()

    return results


def _get_experiment_handlers(experiment_id: str) -> dict:
    """Get experiment-specific handlers.

    Loads handlers from the handlers/ package based on experiment ID.
    Falls back to stub handlers if none are implemented.
    """
    try:
        from handlers import get_handlers_for_experiment

        handlers = get_handlers_for_experiment(experiment_id)
        if handlers:
            print(f"Loaded {len(handlers)} handlers for {experiment_id}")
            print(f"  Available: {list(handlers.keys())}")
            return handlers
        else:
            print(f"[WARN] No handlers found for {experiment_id}")
    except ImportError as e:
        print(f"[WARN] Could not load handlers: {e}")

    print("       Falling back to stub handlers")
    from runner import create_stub_handlers

    return create_stub_handlers(experiment_id)


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=3600 * 2,
    volumes={
        "/results": results_volume,
        "/model-cache": model_cache,
    },
    secrets=[wandb_secret],
)
def download_models():
    """Pre-download models to cache volume."""
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    import torch

    print("Downloading Qwen2.5-VL-7B-Instruct...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    print(f"  - Qwen2.5-VL loaded, {model.num_parameters() / 1e9:.1f}B params")

    # Clean up to free memory before loading next model
    del model
    del processor
    torch.cuda.empty_cache()

    print("\nDownloading LTX-Video...")
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
    )
    print("  - LTX-Video loaded")

    # Clean up
    del pipe
    torch.cuda.empty_cache()

    # Commit cache
    model_cache.commit()

    print("\nModels cached successfully!")
    return {"status": "ok"}


@app.function(
    image=gpu_image,
    gpu="A100",
    timeout=3600 * 2,
    volumes={
        "/datasets": datasets_volume,
        "/model-cache": model_cache,
    },
)
def download_datasets():
    """Download datasets to the persistent volume.

    Downloads:
    - Something-Something v2 subset (~5GB) - for real experiments
    - Generates synthetic test data - for quick debugging
    """
    from pathlib import Path

    print("=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)

    datasets_dir = Path("/datasets")
    datasets_dir.mkdir(exist_ok=True)

    # =========================================================================
    # 1. Generate synthetic test data
    # =========================================================================
    print("\n[1/2] Generating synthetic test data...")

    synthetic_dir = datasets_dir / "synthetic"
    synthetic_dir.mkdir(exist_ok=True)

    from PIL import Image, ImageDraw
    import numpy as np

    # Generate 100 synthetic test images
    categories = [
        ("red_circle", (255, 0, 0), "circle"),
        ("blue_square", (0, 0, 255), "square"),
        ("green_triangle", (0, 255, 0), "triangle"),
        ("yellow_star", (255, 255, 0), "star"),
    ]

    for cat_name, color, shape in categories:
        cat_dir = synthetic_dir / cat_name
        cat_dir.mkdir(exist_ok=True)

        for i in range(25):
            img = Image.new("RGB", (224, 224), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            cx = np.random.randint(50, 174)
            cy = np.random.randint(50, 174)
            size = np.random.randint(30, 60)

            if shape == "circle":
                draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
            elif shape == "square":
                draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
            elif shape == "triangle":
                points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                draw.polygon(points, fill=color)
            elif shape == "star":
                # Simple 5-pointed star
                outer_r = size
                inner_r = size // 2
                points = []
                for j in range(10):
                    r = outer_r if j % 2 == 0 else inner_r
                    angle = np.pi / 2 + j * np.pi / 5
                    points.append((cx + int(r * np.cos(angle)), cy - int(r * np.sin(angle))))
                draw.polygon(points, fill=color)

            img.save(cat_dir / f"{i:03d}.png")

    print(f"  Created {4 * 25} synthetic images in {synthetic_dir}")

    # =========================================================================
    # 2. Download Something-Something v2 subset
    # =========================================================================
    print("\n[2/2] Downloading Something-Something v2 subset...")

    ssv2_dir = datasets_dir / "something-something-v2"
    ssv2_dir.mkdir(exist_ok=True)

    # NOTE: Something-Something v2 requires agreeing to terms
    # For now, create a placeholder and instructions
    readme = ssv2_dir / "README.md"
    readme.write_text(
        """# Something-Something v2 Dataset

This dataset requires manual download due to licensing.

## Steps:
1. Go to https://developer.qualcomm.com/software/ai-datasets/something-something
2. Agree to terms and download
3. Upload to this directory using Modal volume commands

## Alternative: Use Hugging Face
```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/something-something-v2", split="validation[:1000]")
```

Note: This also requires agreeing to terms at:
https://huggingface.co/datasets/HuggingFaceM4/something-something-v2
"""
    )
    print(f"  Created instructions at {readme}")
    print("  NOTE: SSv2 requires manual download - see README")

    # Commit changes
    datasets_volume.commit()

    print("\n" + "=" * 60)
    print("DATASET DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nSynthetic data: {synthetic_dir}")
    print(f"SSv2 instructions: {ssv2_dir}")

    return {"status": "ok", "synthetic_images": 100}


@app.function(
    image=gpu_image,
    volumes={"/results": results_volume},
)
def list_results():
    """List all experiment results."""
    results_dir = Path("/results")
    experiments = []

    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                import json

                with open(results_file) as f:
                    data = json.load(f)
                experiments.append(
                    {
                        "experiment_id": exp_dir.name,
                        "status": data.get("status", "unknown"),
                    }
                )
            else:
                experiments.append({"experiment_id": exp_dir.name, "status": "no results"})

    return experiments


@app.local_entrypoint()
def main(experiment_id: str = "", action: str = "smoke"):
    """Local entrypoint for running experiments.

    Args:
        experiment_id: Experiment to run
        action: 'smoke', 'download', 'run', or 'list'
    """
    if action == "smoke":
        result = smoke_test.remote()
        print(f"Result: {result}")
    elif action == "download":
        result = download_models.remote()
        print(f"Result: {result}")
    elif action == "list":
        results = list_results.remote()
        print("Experiments:")
        for r in results:
            print(f"  - {r['experiment_id']}: {r['status']}")
    elif action == "run":
        if not experiment_id:
            print("Error: --experiment-id required for 'run' action")
            return
        result = run_experiment.remote(experiment_id)
        print(f"Result: {result}")
    else:
        print(f"Unknown action: {action}")
