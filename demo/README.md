# Foresight Demo

Interactive demo UI for the Foresight research prototype. This demo showcases the system's ability to generate visual predictions alongside conversational AI.

## Architecture

- **Frontend**: React + TypeScript + Vite (built with Bun)
- **Backend**: FastAPI with WebSocket support
- **Design**: Two-panel layout (60% chat, 40% thoughts visualization)

## Quick Start

### Prerequisites

- [Bun](https://bun.sh/) (v1.0+) for frontend
- Python 3.11+ with `uv` for backend

### Development

1. **Start the backend** (mock mode, no GPU required):

```bash
cd demo/backend
FORESIGHT_MOCK_MODE=true uvicorn main:app --reload --port 8000
```

2. **Start the frontend**:

```bash
cd demo/frontend
bun install
bun run dev
```

3. Open http://localhost:3000

### Production Build

```bash
# Build frontend
cd demo/frontend
bun run build

# Run backend (serves built frontend)
cd demo/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
demo/
├── README.md           # This file
├── DESIGN.md           # Design specification
├── config.yaml         # Configuration
├── frontend/           # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── ThoughtsPanel.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── MetricsDisplay.tsx
│   │   │   └── Timeline.tsx
│   │   ├── hooks/
│   │   │   ├── useChat.ts
│   │   │   └── useWebSocket.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   └── styles/
│   │       └── globals.css
│   └── index.html
└── backend/            # FastAPI backend
    ├── main.py
    ├── config.py
    ├── api/
    │   ├── routes.py
    │   └── websocket.py
    ├── models/
    │   └── schemas.py
    └── services/
        └── inference.py
```

## API Endpoints

### REST

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | System status |
| GET | `/api/health` | Health check |
| GET | `/api/config` | Current configuration |
| POST | `/api/chat` | Chat (non-streaming) |
| POST | `/api/upload` | Upload image |

### WebSocket

Connect to `/ws/chat` for real-time streaming:

**Client sends:**
```json
{
  "message": "What will happen if...",
  "messageId": "unique-id",
  "imageBase64": "..." // optional
}
```

**Server streams:**
```json
{"type": "text_chunk", "data": {"messageId": "...", "chunk": "...", "done": false}}
{"type": "prediction_start", "data": {"predictionId": "..."}}
{"type": "prediction_progress", "data": {"predictionId": "...", "progress": 50}}
{"type": "prediction_complete", "data": {"predictionId": "...", "videoUrl": "...", "metrics": {...}}}
```

## Configuration

Environment variables (prefix `FORESIGHT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FORESIGHT_MOCK_MODE` | `true` | Use mock inference |
| `FORESIGHT_HOST` | `0.0.0.0` | Server host |
| `FORESIGHT_PORT` | `8000` | Server port |
| `FORESIGHT_DEBUG` | `false` | Debug mode |
| `FORESIGHT_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM model |
| `FORESIGHT_VIDEO_DECODER` | `Lightricks/LTX-Video` | Video decoder |
| `FORESIGHT_USE_HYBRID_ENCODER` | `true` | Enable DINOv2 (P2) |

## Mock Mode

By default, the demo runs in mock mode which:
- Generates placeholder video frames
- Simulates realistic latencies
- Produces random (but plausible) metrics
- **Does not require a GPU**

Set `FORESIGHT_MOCK_MODE=false` for real inference (requires ~40GB VRAM).

## Design Reference

See [DESIGN.md](./DESIGN.md) for:
- UI wireframes and layout
- User flows
- Component specifications
- Performance targets
