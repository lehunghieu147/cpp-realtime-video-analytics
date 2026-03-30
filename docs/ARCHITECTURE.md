# Architecture

## Thread Design

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Capture      │     │ Frame Queue  │     │ Inference Worker  │
│ Thread       ├────►│ (bounded, 8) ├────►│ #0               │
│              │     │ drop-oldest  │     │ (own ORT session) │
│ reads USB    │     │              │  ┌──┤                   │
│ camera at    │     └──────────────┘  │  └───────┬──────────┘
│ native FPS   │                      │           │
└──────────────┘     ┌──────────────┐  │  ┌───────▼──────────┐
                     │ Result Queue │◄─┘  │ Inference Worker  │
                     │ (bounded,16) │◄────┤ #1               │
                     │              │     │ (own ORT session) │
                     └──────┬───────┘     └──────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         cv::imshow    SSE Server    MJPEG Stream
         (local)       /events       /video
                       (JSON)        (frames)
                            │             │
                            ▼             ▼
                       Browser Dashboard (port 9001)
```

### Thread Responsibilities

| Thread | Owner | Purpose |
|--------|-------|---------|
| Capture | `Pipeline::captureLoop()` | Read frames from camera/video, push to frame queue |
| Inference Worker N | `Pipeline::inferenceWorker(N)` | Pop frame, run YOLOv8 detection, push result |
| HTTP Server | `SseServer::httpThread_` | Serve static files, SSE events, MJPEG stream |
| Main | `main()` | Consume results, draw overlays, broadcast to SSE |

### Why Session-per-Worker

ONNX Runtime sessions are thread-safe for inference, but concurrent `Run()` calls on a single session serialize internally. Using one session per worker avoids contention and gives true parallelism.

## Data Flow

```
Camera Frame (cv::Mat BGR 640x480)
    │
    ▼ captureLoop()
FrameData { frameId, captureTime, frame }
    │
    ▼ push to BoundedQueue<FrameData>
    │
    ▼ inferenceWorker()
    │
    ├── preProcess(frame) → tensor (1x3x640x640 float32, normalized)
    │   └── resize → BGR→RGB → HWC→CHW → normalize [0,1]
    │
    ├── session->Run(tensor) → output tensor (1x84x8400)
    │
    ├── postProcess(output) → vector<Detection>
    │   └── transpose → confidence filter → NMS
    │
    ▼
AnalyticsResult { frameId, frame, detections, latencyMs }
    │
    ▼ push to BoundedQueue<AnalyticsResult>
    │
    ▼ main loop consumes
    │
    ├── drawDetections() → bounding boxes on frame
    ├── drawOverlay() → FPS/latency HUD
    ├── cv::imshow() → local display
    ├── serializeResult() → JSON
    ├── sseServer.broadcast(json) → SSE clients
    └── sseServer.updateFrame(frame) → MJPEG clients
```

## Buffer Management

### BoundedQueue

Thread-safe queue with configurable max size. When full, **drops the oldest item** to make room. This ensures:
- Memory stays bounded regardless of producer/consumer speed mismatch
- The system always processes the most recent frames (low latency over completeness)
- No producer blocking — capture thread never stalls

```cpp
// Producer side (never blocks)
queue.push(item);  // drops oldest if full

// Consumer side (blocks until item available or stopped)
auto item = queue.pop();       // blocking
auto item = queue.tryPopFor(timeout);  // timed wait
```

### Queue Sizing

| Queue | Default Size | Rationale |
|-------|-------------|-----------|
| Frame Queue | 8 | Small buffer between capture and inference |
| Result Queue | 16 | Larger to absorb inference timing jitter |
| Benchmark Mode | maxFrames+10 | No dropping — process every frame for accurate metrics |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Camera disconnect | 30 consecutive read failures → pipeline stops |
| Video file EOF | Capture thread exits, queue stops, inference drains remaining frames |
| Model load failure | Pipeline refuses to start, returns error |
| SIGINT/SIGTERM | Global flag set, all loops exit gracefully |

## HTTP Server Routes

| Route | Type | Description |
|-------|------|-------------|
| `/` | Static | Dashboard HTML page |
| `/events` | SSE | JSON detection results stream |
| `/video` | MJPEG | Live video frame stream |
| `/app.js`, `/app.css` | Static | Dashboard assets |
