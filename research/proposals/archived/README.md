# Archived Proposals

These pivot proposals were evaluated but **not pursued**. They are preserved for future reference.

## Decision (2026-01-18)

After Gate 1 revealed spatial information loss in VLM embeddings, we evaluated 4 pivot options:

| Proposal | Decision | Reason |
|----------|----------|--------|
| **Pivot 1: Pre-merge ViT** | Rejected | Q2 showed pre-merge IoU was only 0.101 - spatial loss occurs before merger |
| **Pivot 2: Hybrid Encoder** | **ACCEPTED** | Proven components (DINOv2), lowest cost, directly addresses problem |
| **Pivot 3: Spatial Enhancement** | Rejected | High risk (30-40% success) - can't recover destroyed information |
| **Pivot 4: Alternative VLM** | Rejected | Risk that all VLMs have similar limitations; higher cost than hybrid |

## Why Hybrid Encoder Won

1. **Addresses root cause** - DINOv2 is proven for spatial tasks (SOTA on dense prediction)
2. **Lowest cost** - ~$580 vs $1,600-$25,000 for alternatives
3. **Preserves existing work** - Keeps Qwen2.5-VL for semantics/temporal reasoning
4. **Proven architecture** - Cross-attention fusion is well-understood (BLIP-2, Q-Former)

## Reopening Criteria

Consider revisiting these proposals if:
- Hybrid encoder fails to achieve Spatial IoU > 0.6
- DINOv2 proves insufficient for our specific use case
- New VLMs emerge with better spatial preservation
