#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

// Forward declare — defined in sse-server.cpp
struct SseClients;

// SSE + MJPEG server for streaming detection results and video to browsers.
// Main thread calls broadcast() and updateFrame() — no queue dependency.
struct SseServerConfig {
  int port = 9001;
  std::string staticDir = "web/";
};

class SseServer {
public:
  explicit SseServer(const SseServerConfig& config);
  ~SseServer();

  // Start HTTP server (SSE + MJPEG + static files) in background thread
  void start();
  void stop();

  // Called by main thread to push data to connected clients
  void broadcast(const std::string& json);    // → SSE /events
  void updateFrame(const cv::Mat& frame);     // → MJPEG /video

  bool isRunning() const { return running_; }

private:
  SseServerConfig config_;
  std::thread httpThread_;
  std::atomic<bool> running_{false};

  // SSE client tracking (shared with http thread)
  std::shared_ptr<SseClients> sseClients_;

  // Latest JPEG frame for MJPEG streaming
  std::mutex frameMutex_;
  std::condition_variable frameReady_;
  std::vector<unsigned char> jpegBuffer_;
  uint64_t frameSeq_ = 0;
};
