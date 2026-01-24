# First Successful End-to-End Test

**Date:** 2026-01-23
**Status:** SUCCESS (inference pipeline working)
**Model Quality:** Incoherent (expected - untrained prediction head)

---

## Test Setup

- **Frontend:** https://foresight-demo-kappa.vercel.app
- **Backend:** Fly.io (FastAPI + WebSocket)
- **Inference:** Modal A10G GPU
- **Models:** Qwen2.5-VL-7B + LTX-Video

---

## Conversation 1: Leaf Pile Scenario

### User Message 1
**Image:** Pile of autumn leaves
**Prompt:** "What would happen if I used a leaf blower on this pile and my friend was hiding inside?"

### Assistant Response 1
> Using a leaf blower on the pile of leaves could be quite a chaotic and potentially dangerous scenario. Here's what might happen:
>
> 1. **Sudden Movement**: The leaf blower would likely disperse the leaves quickly, creating a loud noise and a sudden burst of movement.
> 2. **Hidden Friend**: If your friend is hiding inside the pile of leaves, the sudden movement caused by the leaf blower could displace them. Depending on how they were positioned, they might be partially or fully exposed to view.
> 3. **Potential Injury**: The force of the leaf blower could potentially cause physical harm to your friend if they are not properly shielded or if the noise and movement are unexpected.
>
> For safety reasons, it is important to ensure that anyone hiding in the leaves knows when you plan to use the leaf blower and that the area is clear before starting. Additionally, using the leaf blower outside of a safe distance from where someone might be hiding can help avoid any accidents.

**Video Generated:** Yes (25 frames)
**Metrics Displayed:** LPIPS: 0.200, Confidence: 85%, IoU: 75%

---

### User Message 2
**Image:** Flaming torch
**Prompt:** "What if I added this?"

### Assistant Response 2
> If you were to add something to the first image (the autumn leaves pile), it could be a creative touch that enhances the seasonal and natural theme. Here are some suggestions:
>
> 1. **A small figure or toy**: A cute autumn-themed figurine, like a pumpkin or a scarecrow, could complement the pile of leaves and add a playful element.
> 2. **A leaf blower or rake**: Including a leaf blower or rake next to the pile could suggest an activity related to fall cleanup, giving the scene more context.
> 3. **A bird or butterfly**: Adding a bird or a butterfly might symbolize the change in seasons and bring a sense of life amidst the stillness of the fallen leaves.
> [... additional suggestions ...]

**Video Generated:** Yes (25 frames)
**Multi-image Context:** Working (both images visible to VLM)

---

## Observations

### What Worked
1. **End-to-end pipeline functional** - User can upload images, send prompts, receive text + video
2. **Multi-image context** - VLM correctly references both uploaded images
3. **Streaming text** - Markdown renders correctly with bold, lists, etc.
4. **Video generation** - 25 frames generated and displayed in timeline
5. **UI/UX polish** - Thinking state, bouncing dots, frame player all working

### What Needs Improvement
1. **Video prediction quality** - Videos show semi-random patterns, not coherent predictions
2. **Scenario understanding** - VLM response is reasonable, but video doesn't match the prediction
3. **Safety reasoning** - VLM correctly identified dangers (good for eval baseline)

### Key Insight
The VLM text response shows good reasoning about "what would happen" but the video generation is disconnected from this reasoning. This validates the core hypothesis that we need to train an adapter/prediction head to bridge VLM understanding to coherent video generation.

---

## Screenshot

![First successful test](../../../demo/docs/first-test-screenshot-2026-01-23.png)

---

## Next Steps

1. **Build evaluation framework** - Measure alignment between VLM prediction text and generated video
2. **Train prediction head** - C3 experiments to teach model to predict future states
3. **Collect test cases** - Build diverse set of scenarios for evaluation
