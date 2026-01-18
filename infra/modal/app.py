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
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.40.0",
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
        # Flash attention (optional, for speed)
        # "flash-attn",  # Uncomment if needed, requires specific CUDA version
    )
    .pip_install(
        # Install from GitHub for latest
        "qwen-vl-utils",
    )
)

# Volume for persisting results and caching models
results_volume = modal.Volume.from_name("foresight-results", create_if_missing=True)
model_cache = modal.Volume.from_name("foresight-model-cache", create_if_missing=True)

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
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
    },
    secrets=[wandb_secret],
)
def run_experiment(experiment_id: str, sub_experiment: str | None = None):
    """Run a Foresight experiment on GPU.

    Args:
        experiment_id: Experiment ID (e.g., 'c1-vlm-latent-sufficiency')
        sub_experiment: Optional specific sub-experiment (e.g., 'e1_2')
    """
    import json
    import yaml

    print(f"Starting experiment: {experiment_id}")
    if sub_experiment:
        print(f"Sub-experiment: {sub_experiment}")

    # Set up environment
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    # Create results directory
    results_dir = Path(f"/results/{experiment_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "artifacts").mkdir(exist_ok=True)

    # Initialize W&B
    import wandb

    wandb.init(
        project="foresight",
        name=f"{experiment_id}_{sub_experiment or 'full'}",
        tags=[experiment_id.split("-")[0], experiment_id],
        config={
            "experiment_id": experiment_id,
            "sub_experiment": sub_experiment,
        },
    )

    # Load experiment plan (would be passed in or fetched)
    # For now, just run a placeholder
    print(f"Results will be saved to: {results_dir}")

    # Placeholder for actual experiment logic
    # Each experiment would have its own runner module
    results = {
        "experiment_id": experiment_id,
        "status": "placeholder",
        "message": "Implement experiment runner for this experiment",
    }

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    wandb.finish()

    # Commit volume changes
    results_volume.commit()

    return results


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

    print("Downloading Qwen2.5-VL-7B-Instruct...")
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    print(f"  - Qwen2.5-VL loaded, {model.num_parameters() / 1e9:.1f}B params")

    print("\nDownloading LTX-Video...")
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype="auto",
    )
    print("  - LTX-Video loaded")

    # Commit cache
    model_cache.commit()

    print("\nModels cached successfully!")
    return {"status": "ok"}


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
