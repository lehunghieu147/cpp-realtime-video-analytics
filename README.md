# Real-time AI Video Analytics Pipeline

A production-grade C++ pipeline that captures USB camera frames, runs YOLOv8 object detection via ONNX Runtime, and streams results to a browser dashboard.

## Architecture

```
USB Camera ──► Capture Thread ──► Frame Queue ──► Inference Workers (thread pool)
                                                        │
                                                  Result Queue
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                     SSE Server    MJPEG Stream   OpenCV Display
                                     (JSON)        (/video)       (local window)
                                          │             │
                                          ▼             ▼
                                      Browser Dashboard (http://localhost:9001)
```

**Key design decisions:**
- Thread-per-concern: capture, inference, and HTTP server run independently
- Session-per-worker: each inference thread owns its ONNX Runtime session (thread-safe)
- Bounded queue with drop-oldest: prevents memory growth under load
- SSE + MJPEG: JSON results via Server-Sent Events, video via Motion JPEG

## Quick Start

### Docker

```bash
docker build -t video-analytics .
docker run --rm --device /dev/video0 -p 9001:9001 video-analytics
```

### Manual Build

```bash
# Prerequisites: Ubuntu 22.04+, CMake 3.20+, GCC 11+
sudo apt install libopencv-dev

# Install ONNX Runtime 1.17.0
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar xzf onnxruntime-linux-x64-1.17.0.tgz
sudo cp onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
sudo ldconfig

# Download YOLOv8n ONNX model
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
mkdir -p models && mv yolov8n.onnx models/

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# Run
build/src/video_analytics --device 0
```

Open http://localhost:9001 for the dashboard.

## Usage

```
video_analytics [OPTIONS]                  Live camera mode
video_analytics --benchmark OPTIONS        Benchmark mode

General options:
  --device N       Camera device index (default: 0)
  --model PATH     ONNX model path (default: models/yolov8n.onnx)
  --port N         Dashboard port (default: 9001)
  --bind ADDR      Bind address (default: 127.0.0.1)
  --workers N      Inference worker threads (default: 2)

Benchmark options:
  --input FILE     Input video file (required)
  --frames N       Number of frames (default: 300)
  --output FILE    Write report to file
```

### Examples

```bash
# Live mode with camera 2
build/src/video_analytics --device 2

# Live mode on all interfaces, port 8080, 4 workers
build/src/video_analytics --bind 0.0.0.0 --port 8080 --workers 4

# Benchmark mode
build/src/video_analytics --benchmark --input test.avi --frames 500 --output report.md
```

## Benchmarks

**Hardware:** Intel CPU, 2 inference workers, YOLOv8n (640x640)

| Metric | Value |
|--------|-------|
| Throughput | ~25-33 FPS |
| Inference latency (per frame) | ~35 ms |
| Model size | 12.3 MB (FP32) |

**FP16 quantization** available via `scripts/quantize-fp16.py` — reduces model to 6.2 MB (50%). Requires `onnxruntime-gpu` with CUDA provider for runtime support.

## Project Structure

```
src/
├── capture/          Video capture (OpenCV backend)
│   ├── video-capture.hpp    IVideoCapture interface
│   ├── opencv-capture.hpp   OpenCV implementation
│   └── opencv-capture.cpp
├── inference/        YOLOv8 ONNX Runtime inference
│   ├── inference-engine.hpp Engine class + config
│   ├── inference-engine.cpp ONNX session management
│   ├── pre-process.hpp/cpp  Frame → tensor (resize, normalize, HWC→CHW)
│   ├── post-process.hpp/cpp Tensor → detections (NMS, confidence filter)
│   └── detection.hpp        Detection struct + COCO class names
├── pipeline/         Thread-safe capture→inference pipeline
│   ├── pipeline.hpp/cpp     Orchestrates threads + queues
│   ├── bounded-queue.hpp    Lock-free bounded queue (drop-oldest)
│   └── frame-data.hpp       FrameData + AnalyticsResult structs
├── server/           HTTP server (SSE + MJPEG)
│   ├── sse-server.hpp/cpp   cpp-httplib based server
│   └── result-serializer.hpp JSON serialization
├── benchmark/        Latency tracking + report generation
│   ├── latency-tracker.hpp  Percentile stats (p50/p95/p99)
│   └── benchmark-report.hpp Markdown report generator
├── utils/
│   └── fps-tracker.hpp      Rolling FPS calculator
└── main.cpp          CLI entry point
```

## Testing

```bash
build/tests/test_pipeline         # BoundedQueue, FrameData, threading
build/tests/test_inference        # ONNX model loading, detection
build/tests/test_serializer       # JSON serialization
build/tests/test_latency_tracker  # Percentile math, report format
```

Build with sanitizers:
```bash
cmake -B build -DENABLE_TSAN=ON   # ThreadSanitizer
cmake -B build -DENABLE_ASAN=ON   # AddressSanitizer
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — thread design, data flow, buffer strategy
- [Development](docs/DEVELOPMENT.md) — build guide, prerequisites, profiling

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | >= 4.6 | Video capture, image processing |
| ONNX Runtime | >= 1.17 | YOLOv8 inference (C++ API) |
| nlohmann/json | 3.11.3 | JSON serialization (fetched by CMake) |
| cpp-httplib | 0.18.7 | HTTP/SSE/MJPEG server (fetched by CMake) |

## License

MIT
