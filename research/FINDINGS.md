# Research Findings

High-level summary of Foresight experiment results. For detailed findings, see individual experiment FINDINGS.md files.

---

## Project Status

**Current Phase:** Phase 1 - Foundation
**Overall Progress:** Gate 1 in progress (1/3 experiments complete)

---

## Phase 1: Can VLM latents support video reconstruction?

### [C1: VLM Latent Sufficiency](experiments/c1-vlm-latent-sufficiency/FINDINGS.md) ✅ PASSED

**Question:** Can Qwen2.5-VL's internal representation be decoded back into video?

**Answer:** Yes. LPIPS=0.236 (target <0.35), SSIM=0.946 (target >0.75).

**Key Insight:** Post-merge latents work better than pre-merge (surprising). Middle layers (layer 14) are optimal. A ~20M parameter adapter is sufficient.

**Caveat:** Spatial precision slightly below target (IoU=0.567 vs 0.6 target).

→ [Full details](experiments/c1-vlm-latent-sufficiency/FINDINGS.md)

---

### [Q1: Latent Space Alignment](experiments/q1-latent-alignment/FINDINGS.md) 🔄 RUNNING

**Question:** Do VLM and video decoder latent spaces have compatible structure?

**Answer:** Pending

→ [Full details](experiments/q1-latent-alignment/FINDINGS.md)

---

### [Q2: Information Preservation](experiments/q2-information-preservation/FINDINGS.md) 🔄 RUNNING

**Question:** Where does spatial information get lost in the VLM pipeline?

**Answer:** Pending

→ [Full details](experiments/q2-information-preservation/FINDINGS.md)

---

## Decision Gates

### Gate 1: Reconstruction
**Status:** 🔄 In Progress

| Experiment | Status | Result |
|------------|--------|--------|
| C1 - VLM Latent Sufficiency | ✅ Complete | **PASSED** |
| Q1 - Latent Alignment | 🔄 Running | Pending |
| Q2 - Information Preservation | 🔄 Running | Pending |

**Unlocks:** Phase 2 (Adapter Training)

---

## Key Learnings So Far

### What's Working

1. **Core hypothesis validated** - VLM latents contain enough visual information for reconstruction
2. **Small adapters work** - ~20M params is sufficient to bridge representations
3. **Token merging helps** - Post-merge representations are better, not worse

### Open Questions

1. **Spatial precision** - Can we improve IoU from 0.567 to >0.6?
2. **Real video** - Will results hold on Something-Something v2?
3. **Multi-object scenes** - How does complexity affect reconstruction?

### Surprising Findings

1. **Post-merge > Pre-merge** - The 2x2 token merger aggregates information helpfully
2. **Middle layers best** - Layer 14 beats both early and late layers for reconstruction

---

## Changelog

| Date | Update |
|------|--------|
| 2026-01-18 | C1 completed - core hypothesis validated |
| 2026-01-18 | Q1, Q2 experiments started |
| 2026-01-18 | Created experiment-level FINDINGS.md structure |
