#!/bin/bash
# Sync experiment results from Modal volume to local research/experiments directory
# Run this after experiments complete on Modal to get latest results locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EXPERIMENTS=(
  "c1-vlm-latent-sufficiency"
  "q1-latent-alignment"
  "q2-information-preservation"
  "c2-adapter-bridging"
  "q3-temporal-coherence"
  "c3-future-prediction"
  "q4-training-data"
  "q5-prediction-horizon"
  "c4-pixel-verification"
)

echo "Syncing results from Modal volume 'foresight-results'..."

cd "$PROJECT_ROOT"

for exp in "${EXPERIMENTS[@]}"; do
  exp_dir="research/experiments/${exp}"

  # Create directory if it doesn't exist
  mkdir -p "$exp_dir"

  # Try to sync results.yaml (--force to overwrite existing)
  if uv run modal volume get foresight-results "${exp}/results.yaml" "$exp_dir/" --force 2>/dev/null; then
    echo "  [OK] $exp/results.yaml"
  else
    echo "  [--] $exp (no results yet)"
  fi

  # Try to sync artifacts directory (--force to overwrite existing)
  if uv run modal volume get foresight-results "${exp}/artifacts" "$exp_dir/" -r --force 2>/dev/null; then
    echo "  [OK] $exp/artifacts"
  fi
done

echo ""
echo "Sync complete!"
echo "Run 'cd dashboard && npm run check-status' to update the dashboard."
