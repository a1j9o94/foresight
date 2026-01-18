# Experiment Plans

Detailed research plans for validating the Foresight hypothesis. Each plan is designed to be executed independently by a researcher, with clear success/failure criteria.

## Overview

```
                        ┌─────────────────────────────────────┐
                        │     PRIMARY HYPOTHESIS              │
                        │  Pixel grounding improves reasoning │
                        └──────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
   ┌─────────┐                  ┌─────────────┐                ┌──────────┐
   │ CLAIMS  │                  │  OPEN       │                │ END-TO-  │
   │ (C1-C4) │                  │  QUESTIONS  │                │ END      │
   │         │                  │  (Q1-Q5)    │                │ EVAL     │
   └─────────┘                  └─────────────┘                └──────────┘
```

## Claims (Must Validate)

These are the core claims that must hold for the hypothesis to be true. Each has go/no-go criteria.

| ID | Claim | Risk | Timeline | Plan |
|----|-------|------|----------|------|
| **C1** | VLM latents contain sufficient information | Medium | 18 days | [c1-vlm-latent-sufficiency.md](c1-vlm-latent-sufficiency.md) |
| **C2** | Small adapter can bridge latent spaces | High | 18 days | [c2-adapter-bridging.md](c2-adapter-bridging.md) |
| **C3** | VLM can predict future states | Medium | 4 weeks | [c3-future-prediction.md](c3-future-prediction.md) |
| **C4** | Pixel verification improves accuracy | Low | 28 days | [c4-pixel-verification.md](c4-pixel-verification.md) |

**Dependency chain:** C1 → C2 → C3 → C4

## Open Questions (Need Answers)

These inform architectural decisions but don't necessarily block progress.

| ID | Question | Risk | Timeline | Plan |
|----|----------|------|----------|------|
| **Q1** | Latent space alignment | High | 10-14 days | [q1-latent-alignment.md](q1-latent-alignment.md) |
| **Q2** | Information preservation through VLM | Medium | 5-7 days | [q2-information-preservation.md](q2-information-preservation.md) |
| **Q3** | Temporal coherence | Medium | 2-3 weeks | [q3-temporal-coherence.md](q3-temporal-coherence.md) |
| **Q4** | Training data requirements | Low-Med | 5 weeks | [q4-training-data.md](q4-training-data.md) |
| **Q5** | Right prediction horizon | Low | 14-24 days | [q5-prediction-horizon.md](q5-prediction-horizon.md) |

**Q1 and Q2 can run in parallel with C1** - they inform C2's adapter design.

## Recommended Execution Order

### Phase 1: Foundation (Weeks 1-3)
Run in parallel:
- **C1**: VLM latent sufficiency (critical path)
- **Q1**: Latent space alignment (informs C2)
- **Q2**: Information preservation (informs C2)

**Gate:** If C1 fails (LPIPS > 0.45), pivot before investing in C2-C4.

### Phase 2: Bridging (Weeks 4-6)
- **C2**: Adapter bridging (blocked on C1)
- **Q3**: Temporal coherence (can start once C2 has initial adapter)

**Gate:** If C2 fails (no convergence), explore pivots before C3.

### Phase 3: Prediction (Weeks 7-10)
- **C3**: Future prediction (blocked on C2)
- **Q4**: Training data (can run alongside C3)
- **Q5**: Prediction horizon (can run alongside C3)

**Gate:** If C3 fails (predictions no better than random), core hypothesis is challenged.

### Phase 4: Verification (Weeks 11-14)
- **C4**: Pixel verification (blocked on C3)
- End-to-end evaluation

**Gate:** If C4 fails, pixel grounding hypothesis is falsified (but system may still be useful).

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB (RTX 4090) | 40-80GB (A100) |
| Storage | 500GB | 2TB |
| GPU Hours | 500 | 1000+ |
| Calendar Time | 10 weeks | 14 weeks |

## Success Dashboard

Track progress with this checklist:

### Claims
- [ ] **C1**: LPIPS < 0.35, Spatial IoU > 0.6
- [ ] **C2**: 10M adapter achieves >80% of 100M quality
- [ ] **C3**: Predicted latents closer to correct future than random (p < 0.01)
- [ ] **C4**: Verification loop improves accuracy by >10%

### Questions
- [ ] **Q1**: Linear probe R² determined, adapter architecture recommended
- [ ] **Q2**: Optimal extraction point identified
- [ ] **Q3**: Conditioning guidelines established
- [ ] **Q4**: Minimum viable dataset size determined
- [ ] **Q5**: Optimal prediction horizon for each task type

### End-to-End
- [ ] Full system beats text-only baseline on action prediction
- [ ] Pixel prediction beats latent-only prediction
- [ ] System runs in <10 seconds per prediction

## Related Documents

- [Core Hypothesis](../hypotheses/core-hypothesis.md)
- [Paper Index](../papers/index.md)
- [PRD](../../PRD.md)
