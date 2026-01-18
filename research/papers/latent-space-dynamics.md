# Learning Latent Dynamics for Planning from Pixels (PlaNet)

**Authors:** Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson
**Year:** 2019
**Venue:** ICML 2019
**Links:** [Paper](https://arxiv.org/abs/1811.04551) | [Code](https://github.com/google-research/planet) | [Project](https://planetrl.github.io/)

## Summary

This foundational paper introduces PlaNet (Deep Planning Network), a purely model-based agent that learns environment dynamics from images and selects actions through fast online planning in latent space. The key innovation is that instead of learning to predict future pixels directly (which is computationally expensive and prone to error accumulation), the model learns a compact latent representation where dynamics are easier to model and planning can occur efficiently.

The approach addresses a long-standing challenge: learning dynamics models accurate enough for planning in image-based domains. PlaNet uses a latent dynamics model with both deterministic and stochastic transition components, trained with a novel multi-step variational inference objective called "latent overshooting." This enables the agent to solve continuous control tasks with contact dynamics, partial observability, and sparse rewards using substantially fewer environment interactions than model-free methods.

## Key Technical Insights

- **Hybrid state representation**: Combining deterministic (RNN hidden states) and stochastic (Gaussian latent variables) components is crucial. The deterministic path prevents information loss across timesteps, while the stochastic path captures inherent environment uncertainty.

- **Latent overshooting**: Instead of only training on single-step predictions, this objective optimizes multi-step predictions at all distances. Critically, it operates entirely in latent space without requiring image reconstruction at each step, making it fast and effective as a regularizer.

- **Planning efficiency**: Because rewards are modeled as functions of latent states, the planner operates purely in latent space without generating images. This enables fast evaluation of thousands of action sequences across multiple trajectories.

- **50x sample efficiency**: The model-based approach requires approximately 50x less environment interaction than comparable model-free methods (A3C, D4PG) while achieving similar or better final performance.

## Architecture/Method

```
                     Observations (Images)
                            |
                            v
                    +---------------+
                    |   Encoder     |  (CNN: image -> latent)
                    +---------------+
                            |
                            v
                    +---------------+
                    |  Latent State |  s_t = [h_t, z_t]
                    | (deterministic|  h_t: RNN hidden state
                    |  + stochastic)|  z_t: Gaussian sample
                    +---------------+
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
    +-------------+  +-------------+  +-------------+
    | Transition  |  |   Reward    |  | Observation |
    |   Model     |  |   Model     |  |  Predictor  |
    | p(s'|s,a)   |  |  p(r|s)     |  |  p(x|s)     |
    +-------------+  +-------------+  +-------------+
            |               |
            v               v
    Future States     Predicted Rewards
            |               |
            +-------+-------+
                    |
                    v
            +---------------+
            |    Planner    |  (CEM: Cross-Entropy Method)
            | (in latent    |  Searches action sequences
            |  space only)  |  without decoding to pixels
            +---------------+
                    |
                    v
              Best Action
```

The Recurrent State-Space Model (RSSM) combines:
1. **Deterministic state model**: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`
2. **Stochastic state model**: `z_t ~ p(z_t | h_t)` (prior) or `z_t ~ q(z_t | h_t, x_t)` (posterior)
3. **Observation model**: `x_t ~ p(x_t | h_t, z_t)`
4. **Reward model**: `r_t ~ p(r_t | h_t, z_t)`

## Results

Key quantitative results on continuous control benchmarks:

| Task | PlaNet (1000 eps) | D4PG (1000 eps) | A3C (1000 eps) |
|------|-------------------|-----------------|----------------|
| Cheetah Run | 662 | 478 | 305 |
| Walker Walk | 918 | 613 | 277 |
| Finger Spin | 561 | 818 | 27 |
| Cartpole Balance | 1000 | 998 | 982 |

- **Data efficiency**: Achieves strong performance in ~50x fewer episodes than model-free baselines
- **Planning horizon**: Effective planning over 50+ step horizons
- **Latent dimension**: Compact 200-dimensional latent state sufficient for complex tasks

## Relevance to Foresight

This paper is foundational for understanding why latent space prediction is superior to pixel-level prediction for reasoning and planning:

- **Direct relevance**: Foresight's GLP architecture shares the core insight that reasoning should happen in latent space, not pixel space. The conditioning adapter in Foresight serves a similar role to PlaNet's encoder - projecting observations into a space where dynamics can be efficiently modeled.

- **Inspiration**:
  - **Multi-step prediction**: Latent overshooting's approach to training on multi-step predictions could inform Foresight's training objective for longer reasoning chains.
  - **Hybrid representations**: The deterministic + stochastic architecture could be adapted for Foresight's latent representations.
  - **Planning without decoding**: Foresight could potentially perform multiple "imagination" steps in latent space before decoding to video, improving efficiency.

- **Contrast**:
  - Foresight decodes to pixels for **verification** - we actually want to see predictions to compare against reality. PlaNet avoids pixel generation entirely.
  - Foresight uses pretrained components (VLM, video decoder) rather than learning everything end-to-end.
  - Our focus is on *generative* prediction (creating videos a human can evaluate) rather than just planning.

## Connection to Chain-of-Thought Reasoning

The latent space dynamics paradigm has direct implications for visual chain-of-thought:

1. **Efficient multi-step reasoning**: Just as PlaNet can roll out 50+ steps in latent space efficiently, a visual reasoning system could "think" through multiple future states before committing to pixel-level outputs.

2. **Continuous latent thought**: Recent work (e.g., Meta's COCONUT - Chain of Continuous Thought) extends this idea to language models, allowing reasoning in continuous latent space rather than discrete tokens. Foresight could leverage similar ideas for visual reasoning.

3. **Imagination as planning**: The "dream to control" paradigm from subsequent Dreamer work shows that imagining future states (latent imagination) enables better decision-making. Foresight extends this by making imagination visible through pixel generation.

## Related Work

This paper spawned a highly influential line of research:

| Paper | Year | Key Addition |
|-------|------|--------------|
| [Dreamer](https://arxiv.org/abs/1912.01603) | 2020 | Learning behaviors by latent imagination |
| [DreamerV2](https://arxiv.org/abs/2010.02193) | 2021 | Discrete latent representations |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | 2023 | Scaling to 150+ diverse domains |
| [JEPA/V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) | 2022-24 | Joint embedding prediction without generation |

## Open Questions

- How to balance latent space efficiency with the need for interpretable (pixel-level) outputs for human verification?
- Can latent overshooting be adapted for video diffusion models to improve temporal consistency?
- What is the optimal trade-off between latent imagination depth and pixel decoding frequency for reasoning tasks?
- How does the stochastic component (modeling uncertainty) relate to representing multiple possible futures in Foresight?

## Code/Implementation Notes

- Official TensorFlow implementation: [google-research/planet](https://github.com/google-research/planet)
- PyTorch reimplementations available in the Dreamer repositories
- RSSM architecture has become a standard component in model-based RL, extensively documented
- The model trains on a single GPU in reasonable time, making it accessible for experimentation

## Citation

```bibtex
@inproceedings{hafner2019learning,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle={International Conference on Machine Learning},
  pages={2555--2565},
  year={2019},
  organization={PMLR}
}
```

## Review Status

- [x] Read abstract
- [x] Read full paper
- [x] Reviewed code
- [x] Summarized key insights
- [x] Connected to hypothesis

---

## Supplementary: Latent vs Pixel Prediction for Reasoning

### Why Latent Dynamics Enable Better Reasoning

The core argument for latent space dynamics, articulated across PlaNet, Dreamer, and JEPA, is that:

1. **Abstraction**: Latent representations capture semantic information (object permanence, physics, temporal flow) while ignoring irrelevant pixel-level details (texture noise, lighting variations).

2. **Efficiency**: Planning/reasoning in a 200-dimensional latent space is orders of magnitude faster than in a 256x256x3 pixel space.

3. **Generalization**: Learned latent dynamics transfer better across visual variations than pixel-level models.

4. **Multi-step stability**: Errors compound slower in latent space than pixel space, enabling longer reasoning chains.

### The Foresight Synthesis

Foresight aims to combine the benefits of both approaches:
- **Reasoning** in latent space (via VLM + query tokens)
- **Verification** in pixel space (via video decoder)

This enables:
1. Efficient multi-step reasoning (latent)
2. Human-interpretable predictions (pixel)
3. Objective evaluation against reality (pixel comparison)
4. Grounded imagination that can be verified (the GLP advantage)

### Key Insight for Visual Chain-of-Thought

The latent space dynamics literature suggests that effective visual chain-of-thought should:
1. Perform most "thinking" in compressed latent space
2. Decode to pixels only at key checkpoints for verification
3. Use the decoded predictions to ground and correct the latent reasoning
4. Iterate between imagination (latent) and verification (pixel) for robust multi-step reasoning
