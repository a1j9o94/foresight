# Foresight Demo UI Design

**Status:** Draft
**Last Updated:** 2026-01-20
**Author:** Design Document for Demo Interface

---

## Overview

The Foresight demo UI is a chat-based interface that showcases the system's ability to generate visual predictions as part of its reasoning process. The core concept: users chat with an AI that can "see" predicted future states, displayed alongside the conversation.

### Design Goals

1. **Intuitive Chat Experience** - Familiar chat interface with minimal learning curve
2. **Visual Reasoning Display** - Show the model's "thoughts" as video predictions in real-time
3. **Research Demonstration** - Clearly communicate the novel capabilities of GLP architecture
4. **Performance Transparency** - Display timing, confidence, and quality metrics

---

## Architecture Decision: Gradio

**Recommendation: Gradio** (over React or Streamlit)

### Rationale

| Consideration | Gradio | React | Streamlit |
|--------------|--------|-------|-----------|
| ML/Video integration | Native | Custom work | Moderate |
| Development speed | Fast | Slow | Fast |
| Real-time streaming | Built-in | Custom | Limited |
| Python backend | Seamless | API needed | Native |
| HuggingFace ecosystem | Perfect | N/A | Good |
| Custom components | Good | Excellent | Limited |
| Production-ready | Yes | Yes | No |

**Key factors:**
1. Project already lists `gradio` in CLAUDE.md commands
2. Native support for video/image components
3. Streaming responses via generators
4. Easy integration with PyTorch/HuggingFace models
5. Built-in handling of file uploads, chat interfaces

---

## UI Layout

### Wireframe

```
+------------------------------------------------------------------+
|                         FORESIGHT DEMO                           |
|  "AI that sees the future"                              [?] [cog]|
+------------------------------------------------------------------+
|                                                                  |
|  +---------------------------+   +-----------------------------+ |
|  |       CHAT PANEL          |   |     THOUGHTS PANEL          | |
|  |       (60% width)         |   |       (40% width)           | |
|  |                           |   |                              |
|  | [User Avatar]             |   |  CURRENT PREDICTION          |
|  | What will happen if I     |   |  +------------------------+  |
|  | pour water into this cup? |   |  |                        |  |
|  |                           |   |  |   [Video Preview]      |  |
|  | [AI Avatar]               |   |  |                        |  |
|  | Based on what I see,      |   |  +------------------------+  |
|  | the water will fill the   |   |  Confidence: 0.87            |
|  | cup and might overflow... |   |  LPIPS: 0.18 | Latency: 2.1s |
|  |                           |   |                              |
|  |                           |   |  PREDICTION TIMELINE         |
|  |                           |   |  +------------------------+  |
|  |                           |   |  | t=0  | t=1  | t=2  |...|  |
|  |                           |   |  | [f1] | [f2] | [f3] |   |  |
|  |                           |   |  +------------------------+  |
|  |                           |   |                              |
|  | +----------------------+  |   |  BEFORE / AFTER              |
|  | | Type your message... |  |   |  +----------+ +----------+   |
|  | +----------------------+  |   |  | Input    | | Predicted|   |
|  | [Upload] [Send]           |   |  | Frame    | | Outcome  |   |
|  +---------------------------+   |  +----------+ +----------+   |
|                                  +-----------------------------+ |
+------------------------------------------------------------------+
|  Model: Qwen2.5-VL-7B + LTX-Video | VRAM: 38.2GB | Status: Ready |
+------------------------------------------------------------------+
```

### Components

#### 1. Header Bar
- Application title and tagline
- Help button (?) - shows tutorial/explanation
- Settings gear - model selection, quality options

#### 2. Chat Panel (Left, 60%)
- Standard chat interface with message history
- User messages with avatar
- AI responses with avatar
- Text input box with placeholder text
- Buttons:
  - Upload image/video (for visual context)
  - Send message
  - Clear conversation

#### 3. Thoughts Panel (Right, 40%)

##### a. Current Prediction
- Primary video display (largest element)
- Auto-plays the latest prediction
- Confidence score badge
- Quality metrics (LPIPS, generation time)

##### b. Prediction Timeline
- Horizontal strip of thumbnail frames
- Clickable to expand individual frames
- Shows temporal progression (t=0, t=1, t=2, etc.)
- Timestamps correspond to predicted future moments

##### c. Before/After Comparison
- Side-by-side display
- Left: Input frame (what the model saw)
- Right: Predicted outcome frame
- Optional: Diff overlay toggle

#### 4. Status Bar (Bottom)
- Currently loaded model
- VRAM usage
- System status (Ready/Processing/Error)
- FPS/latency indicator

---

## User Flow

### Flow 1: Basic Chat with Prediction

```
1. User opens demo interface
   - Models load (show loading state)
   - Status: "Ready"

2. User uploads an image (optional)
   - Image appears in chat
   - Thumbnail shown in input context

3. User types question about future state
   - e.g., "What will happen if I push this ball?"

4. System processes:
   a. VLM encodes visual context
   b. VLM generates text response (streams to chat)
   c. SIMULTANEOUSLY: Query tokens predict future state
   d. Video decoder generates prediction frames
   e. Prediction appears in Thoughts panel

5. User sees:
   - Text response streaming in chat
   - Video prediction appearing in Thoughts panel
   - Metrics updating (latency, confidence)
```

### Flow 2: Multi-Turn Prediction Refinement

```
1. Initial prediction displayed

2. User asks follow-up:
   - "What if I push it harder?"

3. System generates new prediction:
   - Previous prediction moves to timeline history
   - New prediction appears in main view
   - Comparison mode shows change

4. User can scroll timeline to compare predictions
```

### Flow 3: Verification Mode

```
1. User uploads "before" video/image

2. User describes action taken

3. System predicts outcome

4. User uploads "after" video/image (actual outcome)

5. System displays:
   - Predicted vs Actual comparison
   - LPIPS score between prediction and reality
   - Analysis of prediction accuracy
```

---

## Technical Architecture

### Component Diagram

```
+------------------+     +-------------------+     +------------------+
|   Gradio UI      |     |  Backend Server   |     |  Model Pipeline  |
|                  |     |                   |     |                  |
| - Chat Interface |<--->| - WebSocket       |<--->| - Qwen2.5-VL     |
| - Video Display  |     | - Session Mgmt    |     | - DINOv2 (P2)    |
| - Timeline       |     | - Request Queue   |     | - Adapter        |
| - Metrics        |     | - Caching         |     | - LTX-Video      |
+------------------+     +-------------------+     +------------------+
                                 |
                         +-------v-------+
                         |   Storage     |
                         | - Sessions    |
                         | - Artifacts   |
                         | - Cache       |
                         +---------------+
```

### Backend API Endpoints

```python
# Gradio interface functions (not REST endpoints)

def chat_predict(message: str, history: list, image: Optional[Image]) -> Generator:
    """
    Main chat function with streaming.

    Yields:
    - text_chunk: Streaming text response
    - video_frames: Predicted video (when ready)
    - metrics: Timing and quality metrics
    """
    pass

def get_prediction_timeline(session_id: str) -> list[PredictionFrame]:
    """Get all predictions from current session."""
    pass

def compare_prediction(predicted: Video, actual: Video) -> ComparisonResult:
    """Compare prediction against actual outcome."""
    pass
```

### Integration with Foresight Pipeline

```python
# Integration points with foresight_inference package

from foresight_inference import ForesightPipeline
from foresight_models import (
    QwenVLEncoder,
    DINOv2Encoder,     # P2 hybrid
    FusionAdapter,
    LTXVideoDecoder,
)
from foresight_eval import compute_lpips, spatial_iou

class DemoPipeline:
    """Wrapper for demo-specific functionality."""

    def __init__(self, config: DemoConfig):
        # Load models
        self.vlm = QwenVLEncoder(config.vlm_path)
        self.dino = DINOv2Encoder(config.dino_path)  # P2 hybrid
        self.adapter = FusionAdapter(config.adapter_path)
        self.decoder = LTXVideoDecoder(config.decoder_path)

    async def predict_future(
        self,
        image: Image,
        prompt: str,
        num_frames: int = 30,
    ) -> PredictionResult:
        """
        Main prediction pipeline.

        Returns:
            PredictionResult with:
            - video: Generated video frames
            - latents: Intermediate representations
            - metrics: Timing, confidence
            - text_response: VLM text output
        """
        # 1. Encode visual input
        vlm_latents = self.vlm.encode(image)
        dino_latents = self.dino.encode(image)  # P2 hybrid

        # 2. Generate text response (streamed)
        text_stream = self.vlm.generate_text(prompt, image)

        # 3. Extract query token predictions
        query_latents = self.vlm.get_query_predictions()

        # 4. Fuse latents (P2 hybrid approach)
        fused = self.adapter.fuse(vlm_latents, dino_latents, query_latents)

        # 5. Decode to video
        video = self.decoder.generate(fused, num_frames)

        return PredictionResult(
            video=video,
            text_stream=text_stream,
            metrics=self._compute_metrics(),
        )
```

---

## Thoughts Visualization Details

### Real-Time Streaming

The "Thoughts" panel updates progressively:

1. **Immediate** (< 100ms): Show "Thinking..." animation
2. **VLM Encoding** (~500ms): Show encoded latent heatmap
3. **Frame Generation** (~2s): Progressive video reveal
4. **Completion**: Show full video with metrics

### Latent Visualization

```
+----------------------------------+
|  INTERNAL REPRESENTATION         |
|  +----------------------------+  |
|  |  [Attention Heatmap]       |  |
|  |  Highlighting focus areas  |  |
|  +----------------------------+  |
|                                  |
|  Query Tokens: 32/32 active      |
|  Confidence: 0.87                |
+----------------------------------+
```

Options for visualizing the model's internal state:
- Attention heatmaps over input image
- Query token activation patterns
- Latent space trajectory (t-SNE/UMAP projection)
- Layer-by-layer feature maps

### Prediction Timeline UI

```
+--------------------------------------------------+
| TIMELINE                              [< >] [Play]|
| +------+ +------+ +------+ +------+ +------+     |
| | t=0  | | t=1  | | t=2  | | t=3  | | t=4  |     |
| |[img] | |[img] | |[img] | |[img] | |[img] |     |
| +------+ +------+ +------+ +------+ +------+     |
|    ^                                              |
|    |-- current frame indicator                    |
+--------------------------------------------------+
```

Features:
- Horizontal scroll for long timelines
- Click frame to view full-size
- Play/pause auto-advance
- Frame interpolation slider

### Before/After Comparison

```
+---------------------+---------------------+
|      INPUT          |     PREDICTION      |
|  "What I saw"       |  "What I predict"   |
| +-----------------+ | +-----------------+ |
| |                 | | |                 | |
| |   [Image]       | | |   [Video/Img]   | |
| |                 | | |                 | |
| +-----------------+ | +-----------------+ |
|                     |                     |
| [Overlay Diff]  [Swap]  [Full Screen]     |
+-------------------------------------------+
```

Toggle modes:
- Side-by-side (default)
- Overlay with slider
- Diff map (pixel differences)
- Flicker comparison

---

## Configuration Options

### Demo Settings Panel

```yaml
# demo/config.yaml

model:
  vlm: "Qwen/Qwen2.5-VL-7B-Instruct"
  video_decoder: "Lightricks/LTX-Video"
  hybrid_encoder: true  # P2: DINOv2 spatial features
  dino_model: "facebook/dinov2-vitl14"

generation:
  num_frames: 30
  fps: 15
  resolution: [512, 512]
  guidance_scale: 7.5

ui:
  theme: "soft"  # soft, dark, light
  show_metrics: true
  show_latents: false  # Advanced mode
  auto_play_predictions: true

performance:
  batch_size: 1
  use_fp16: true
  cache_encodings: true
  max_history: 10  # predictions to keep
```

### User-Adjustable Settings

Exposed in UI settings panel:
- Video quality (speed vs quality tradeoff)
- Number of prediction frames
- Show/hide technical metrics
- Dark/light mode
- Advanced mode (show latents, attention)

---

## Error States

### Model Loading Failed

```
+------------------------------------------+
|  [Warning Icon]                          |
|                                          |
|  Unable to load models                   |
|                                          |
|  Error: CUDA out of memory               |
|  Required: 40GB VRAM                     |
|  Available: 24GB                         |
|                                          |
|  [Try Reduced Mode] [View Requirements]  |
+------------------------------------------+
```

### Prediction Failed

```
+------------------------------------------+
|  Prediction could not be generated       |
|                                          |
|  Reason: Input image too complex         |
|                                          |
|  Suggestions:                            |
|  - Try a simpler scene                   |
|  - Reduce resolution                     |
|  - Check GPU memory                      |
|                                          |
|  [Retry] [Report Issue]                  |
+------------------------------------------+
```

---

## Responsive Design

### Desktop (> 1200px)
- Full side-by-side layout as shown in wireframe

### Tablet (768px - 1200px)
- Chat panel: 55%
- Thoughts panel: 45%
- Compact metrics display

### Mobile (< 768px)
- Tabbed interface: Chat | Thoughts
- Swipe to switch between panels
- Video preview in chat as thumbnail

---

## Accessibility

- Keyboard navigation for all controls
- Screen reader labels for video content
- High contrast mode option
- Adjustable text sizes
- Alt text generation for predictions

---

## Performance Considerations

### Target Metrics (from CLAUDE.md)

| Operation | Target | Method |
|-----------|--------|--------|
| Video generation | < 2s per 5s clip | LTX-Video real-time |
| VLM reasoning | < 1s | Quantization, caching |
| Total step | < 3s | Parallel processing |

### Optimization Strategies

1. **Streaming**: Show partial results immediately
2. **Caching**: Cache repeated encodings
3. **Quantization**: Use FP16/INT8 where possible
4. **Prefetch**: Encode images while user types
5. **CDN**: Serve static assets from CDN

---

## File Structure

```
demo/
├── DESIGN.md           # This file
├── app.py              # Main Gradio application
├── config.yaml         # Demo configuration
├── components/
│   ├── __init__.py
│   ├── chat.py         # Chat interface component
│   ├── thoughts.py     # Thoughts panel component
│   ├── timeline.py     # Prediction timeline
│   └── comparison.py   # Before/after comparison
├── pipeline/
│   ├── __init__.py
│   └── demo_pipeline.py  # Wraps foresight_inference
├── static/
│   ├── css/
│   │   └── custom.css
│   └── assets/
│       ├── logo.png
│       └── placeholder.mp4
└── tests/
    ├── test_components.py
    └── test_pipeline.py
```

---

## Implementation Phases

### Phase 1: Basic Scaffold (This PR)
- Project structure
- Basic Gradio app with placeholder UI
- Configuration loading
- Mock pipeline responses

### Phase 2: Chat Integration
- Connect to VLM for text generation
- Streaming text responses
- Image upload and display

### Phase 3: Prediction Display
- Integrate video decoder
- Thoughts panel with video display
- Basic metrics display

### Phase 4: Advanced Features
- Timeline navigation
- Before/after comparison
- Latent visualization
- Settings panel

### Phase 5: Polish
- Error handling
- Mobile responsive
- Performance optimization
- Documentation

---

## Dependencies

```python
# demo/requirements.txt (or add to package)

gradio>=4.0.0
torch>=2.0.0
transformers>=4.36.0
diffusers>=0.25.0
accelerate>=0.25.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0

# Internal packages
foresight-inference
foresight-models
foresight-eval
```

---

## Open Questions

1. **WebSocket vs HTTP**: Should predictions stream via WebSocket for lower latency?
2. **Multi-GPU**: How to handle multi-GPU setups for larger models?
3. **Session Persistence**: Save/restore conversation sessions?
4. **Export**: Allow users to download predictions as video files?
5. **Batch Mode**: Process multiple images at once for comparison?

---

## Appendix: Example Prompts

Good demonstration prompts:
- "What will happen if I pour water into this glass?"
- "Predict what this room will look like in 5 seconds if I turn off the light"
- "What would happen if I dropped this ball?"
- "Show me what happens next in this scene"
- "Predict the outcome of pushing this domino"

These work well because:
- Clear physical action
- Observable change
- Short time horizon
- Single object focus

---

## References

- [Gradio Documentation](https://gradio.app/docs/)
- [LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Research findings: `research/FINDINGS.md`
- Agent guide: `research/AGENT_GUIDE.md`
