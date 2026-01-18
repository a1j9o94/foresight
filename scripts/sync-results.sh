#!/bin/bash
# Sync experiment results from Modal volume to local research/experiments directory
# Run this after experiments complete on Modal to get latest results locally
#
# NOTE: Experiment IDs are loaded from research/research_plan.yaml
# (single source of truth for all experiment configuration)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load experiment IDs from research_plan.yaml (single source of truth)
RESEARCH_PLAN="$PROJECT_ROOT/research/research_plan.yaml"

if [ ! -f "$RESEARCH_PLAN" ]; then
  echo "Error: research_plan.yaml not found at $RESEARCH_PLAN"
  exit 1
fi

# Extract experiment IDs using Python (since we already have uv/python in the project)
EXPERIMENTS=($(uv run python -c "
import yaml
with open('$RESEARCH_PLAN') as f:
    plan = yaml.safe_load(f)
for exp_id in plan.get('experiments', {}).keys():
    print(exp_id)
"))

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
