#include "pipeline/pipeline.hpp"
#include "server/sse-server.hpp"
#include "server/result-serializer.hpp"
#include "inference/detection.hpp"
#include "utils/fps-tracker.hpp"

#include <csignal>
#include <iostream>
#include <opencv2/opencv.hpp>

// Global shutdown flag for SIGINT handling
static std::atomic<bool> gShutdown{false};

void signalHandler(int) { gShutdown = true; }

// Draw detection bounding boxes and labels on frame
void drawDetections(cv::Mat& frame,
                    const std::vector<Detection>& detections) {
  for (const auto& det : detections) {
    cv::Rect box(static_cast<int>(det.bbox.x), static_cast<int>(det.bbox.y),
                 static_cast<int>(det.bbox.width),
                 static_cast<int>(det.bbox.height));

    cv::Scalar color(
        static_cast<int>(det.classId * 50) % 256,
        static_cast<int>(det.classId * 80 + 100) % 256,
        static_cast<int>(det.classId * 120 + 50) % 256);

    cv::rectangle(frame, box, color, 2);

    std::string label = std::string(getClassName(det.classId)) + " " +
                        cv::format("%.2f", det.confidence);

    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(frame, cv::Point(box.x, box.y - textSize.height - 6),
                  cv::Point(box.x + textSize.width + 4, box.y), color, -1);
    cv::putText(frame, label, cv::Point(box.x + 2, box.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

// Draw pipeline stats overlay
void drawOverlay(cv::Mat& frame, double fps, double latencyMs,
                 int detectionCount, uint64_t frameId) {
  cv::rectangle(frame, cv::Point(5, 5), cv::Point(280, 85),
                cv::Scalar(0, 0, 0), -1);
  cv::rectangle(frame, cv::Point(5, 5), cv::Point(280, 85),
                cv::Scalar(0, 255, 0), 1);

  cv::Scalar green(0, 255, 0);
  cv::putText(frame, cv::format("FPS: %.1f | Latency: %.0fms", fps, latencyMs),
              cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55, green, 1);
  cv::putText(frame, cv::format("Detections: %d", detectionCount),
              cv::Point(12, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, green, 1);
  cv::putText(frame,
              cv::format("Frame: %llu",
                         static_cast<unsigned long long>(frameId)),
              cv::Point(12, 72), cv::FONT_HERSHEY_SIMPLEX, 0.55, green, 1);
}

int main() {
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // Configure pipeline
  PipelineConfig config;
  config.captureConfig.deviceIndex = 2;
  config.captureConfig.width = 640;
  config.captureConfig.height = 480;
  config.captureConfig.codec = "MJPG";
  config.inferenceConfig.modelPath = "models/yolov8n.onnx";
  config.inferenceConfig.confidenceThreshold = 0.5f;
  config.inferenceConfig.nmsThreshold = 0.45f;
  config.numWorkers = 2;

  Pipeline pipeline(config);
  pipeline.start();

  if (!pipeline.isRunning()) {
    std::cerr << "Failed to start pipeline\n";
    return -1;
  }

  // Start SSE + MJPEG server
  SseServerConfig sseConfig;
  sseConfig.port = 9001;
  sseConfig.staticDir = "web/";
  SseServer sseServer(sseConfig);
  sseServer.start();

  std::cout << "Pipeline running. Press 'q' or Ctrl+C to stop.\n";
  std::cout << "Dashboard: http://localhost:9001\n";

  FpsTracker fpsTracker;

  // Main display loop — sole consumer of result queue
  // Sends data to both OpenCV window AND browser dashboard
  while (!gShutdown) {
    auto resultOpt =
        pipeline.resultQueue().tryPopFor(std::chrono::milliseconds(10));
    if (!resultOpt) continue;

    auto& result = *resultOpt;
    fpsTracker.update();

    // Draw overlays on frame
    drawDetections(result.frame, result.detections);
    drawOverlay(result.frame, fpsTracker.fps, result.inferenceLatencyMs,
                static_cast<int>(result.detections.size()), result.frameId);

    // Send to OpenCV display window
    cv::imshow("Video Analytics", result.frame);

    // Send to browser dashboard (SSE JSON + MJPEG video)
    auto json = serializeResult(result, fpsTracker.fps);
    sseServer.broadcast(json.dump());
    sseServer.updateFrame(result.frame);

    if (cv::waitKey(1) == 'q') break;
  }

  sseServer.stop();
  pipeline.stop();
  cv::destroyAllWindows();

  return 0;
}
