# Inference Pipeline Design

**Priority: Speed of response** - This is a proof of concept that needs to quickly demonstrate the core idea.

---

## TL;DR

Stream text reasoning immediately while video generates in background. Show first frame fast, fill in video progressively. Target: user sees something meaningful within 1-2 seconds.

---

## Existing Infrastructure

The codebase already has:
- **Backend**: FastAPI with WebSocket streaming (`demo/backend/`)
- **Frontend**: React + TypeScript with `useWebSocket` hook, `VideoPlayer`, `ThoughtsPanel`
- **Modal**: A100 GPU functions with model caching

We're extending, not building from scratch.

---

## Pipeline Architecture

```
Input Image
    │
    ├──────────────────┬────────────────────┐
    │                  │                    │
    ▼                  ▼                    │
┌─────────────┐  ┌──────────────┐           │
│ DINOv2-ViT-L│  │ Qwen2.5-VL   │───────────┼──► Text Stream (immediate)
│   (~1GB)    │  │   (~14GB)    │           │    "I see a cup on the table..."
└─────────────┘  └──────────────┘           │
    │                  │                    │
    ▼                  ▼                    │
┌───────────────────────────────┐           │
│   Cross-Attention Fusion      │           │
│         (~10M params)         │           │
└───────────────────────────────┘           │
    │                                       │
    ▼                                       │
┌───────────────────────────────┐           │
│   Conditioning Adapter        │           │
│       (~10-50M params)        │           │
└───────────────────────────────┘           │
    │                                       │
    ▼                                       │
┌───────────────────────────────┐           │
│      LTX-Video (~12GB)        │───────────┴──► Video Stream (progressive)
│      Real-time 30fps          │               Frame 1 fast, then fill
└───────────────────────────────┘

Total VRAM: ~38GB (fits A100-40GB)
```

---

## Speed-First Design Decisions

### 1. Text First, Video Follows

```
t=0.0s  User submits image
t=0.3s  First text token streams ("I see...")
t=0.5s  Text continues streaming
t=1.0s  First video frame ready, display immediately
t=1.5s  More frames stream in
t=2.5s  Full video complete
```

The VLM can start generating text while video is still being created.

### 2. Progressive Video Display

Don't wait for all 30 frames. Show frame 1 immediately, then:
- Option A: Show frames as they generate (choppy but fast feedback)
- Option B: Show frame 1 static, then play full video when ready (cleaner)

**Recommendation:** Option B for PoC - simpler, still fast feedback.

### 3. Resolution/Quality Tradeoffs

| Setting | Latency | Quality |
|---------|---------|---------|
| 256x256, 15 frames | ~0.8s | Low (PoC demo) |
| 512x512, 30 frames | ~2.0s | Medium (default) |
| 768x768, 60 frames | ~4.0s | High (optional) |

**Default to 512x512, 30 frames** - good balance for demos.

---

## Streaming Protocol

### WebSocket Messages (extend existing)

```typescript
// Text stream (already exists)
{ type: "text_chunk", content: "I see a cup..." }

// Video stream (new)
{ type: "video_frame", frame_index: 0, total_frames: 30, data: "<base64 jpeg>" }
{ type: "video_complete", video_url: "/api/videos/abc123.mp4" }

// Combined state
{ type: "thinking_start" }  // Pipeline started
{ type: "thinking_complete", text: "...", video_url: "..." }
```

### Why WebSocket + Base64 (not binary)?

- Simpler to implement
- Existing frontend already handles it
- 33% overhead acceptable for PoC
- Can optimize to binary later if needed

---

## API Endpoints

```
POST /api/predict
  - Input: { image: base64, prompt?: string }
  - Returns: { session_id }
  - Initiates WebSocket stream

WS /api/ws/{session_id}
  - Streams text_chunk, video_frame, video_complete

GET /api/videos/{video_id}.mp4
  - Returns assembled video file
```

---

## Frontend Components

### New: `ThinkingStream.tsx`

```tsx
// Displays both text and video streams side by side
<div className="thinking-stream">
  <div className="text-stream">
    {/* Token-by-token text display */}
    <StreamingText tokens={textTokens} />
  </div>
  <div className="video-stream">
    {/* First frame → full video */}
    <ProgressiveVideo
      firstFrame={firstFrame}
      videoUrl={videoUrl}
      isComplete={isComplete}
    />
  </div>
</div>
```

### Modify: `useChat.ts`

Add video frame handling to existing WebSocket hook.

---

## Implementation Plan

### Phase 1: Minimal Viable Demo (3-5 days)

1. **Backend**: Add video frame streaming to existing WebSocket
2. **Frontend**: Add `ProgressiveVideo` component
3. **Mock Pipeline**: Generate placeholder frames to test streaming
4. **Goal**: See text + video streaming work end-to-end

### Phase 2: Real Models (1 week)

1. **Modal Function**: Load hybrid encoder + LTX-Video
2. **Pipeline Class**: Wire up real inference
3. **Goal**: Real predictions from real models

### Phase 3: Polish (ongoing)

- Latency optimization
- Error handling
- Multiple resolution options
- Session history

---

## Key Decisions Needed

### Must Decide Now

1. **Video display strategy**:
   - A) Show frames as they generate (choppy)
   - B) Show first frame, then full video (cleaner) ← Recommended

2. **Default resolution**: 512x512 @ 30 frames @ 15fps = 2 second video

### Can Decide Later

- S3 vs local storage (local for PoC)
- Caching strategy
- Multi-user scaling
- Adapter versioning

---

## Files to Create/Modify

```
demo/backend/
├── services/
│   └── inference.py      # Add ForesightPipeline class
├── routers/
│   └── predict.py        # Add video streaming endpoints
└── main.py               # Wire up new routes

demo/frontend/src/
├── components/
│   ├── ProgressiveVideo.tsx   # New: first frame → video
│   └── ThinkingStream.tsx     # New: combined text+video view
├── hooks/
│   └── useChat.ts             # Modify: handle video_frame messages
└── types/
    └── index.ts               # Add video message types

infra/modal/
└── inference/
    └── pipeline.py            # New: Modal function for real inference
```

---

## Open Questions

1. **Adapter injection point**: How exactly does the adapter inject into LTX-Video? (Need to check C2 experiment results)

2. **Text generation timing**: Should VLM describe what it "plans to show" before generating, or narrate as video generates?

3. **Error UX**: If video fails but text succeeds, show text only? Or retry?
