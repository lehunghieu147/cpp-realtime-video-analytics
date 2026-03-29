#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "capture/opencv-capture.hpp"
#include "utils/fps-tracker.hpp"

// ── All state for a single camera ─────────────────────────────────────────────
struct CameraUnit {
  std::unique_ptr<IVideoCapture> capture;
  std::string label;
  cv::VideoWriter writer;
  std::thread thread;
  FpsTracker tracker;
  int frameCount = 0;
  cv::Mat latestFrame;

  // Shared state between grab thread and main thread
  cv::Mat sharedFrame;
  std::mutex mtx;
  std::atomic<bool> hasNew{false};
  std::atomic<bool> alive{true};  // false when camera disconnects
};

// ── Detect available cameras ──────────────────────────────────────────────────
std::vector<int> detectCameras(int maxIndex = 10) {
  std::vector<int> found;
  for (int i = 0; i < maxIndex; i++) {
    cv::VideoCapture test(i);
    if (test.isOpened()) {
      found.push_back(i);
      test.release();
    }
  }
  return found;
}

// ── Grab thread ───────────────────────────────────────────────────────────────
void grabThread(CameraUnit& cam, std::atomic<bool>& running) {
  cv::Mat tmp;
  int failCount = 0;
  const int kMaxFails = 30;  // 30 × 10ms = ~300ms before declaring disconnect

  while (running && cam.alive) {
    if (!cam.capture->read(tmp)) {
      failCount++;
      if (failCount >= kMaxFails) {
        std::cerr << cam.label << ": camera disconnected after "
                  << kMaxFails << " consecutive failures\n";
        cam.alive = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    failCount = 0;  // reset on successful read

    std::lock_guard<std::mutex> lock(cam.mtx);
    cam.sharedFrame = tmp.clone();
    cam.hasNew = true;
  }
}

// ── Draw overlay ──────────────────────────────────────────────────────────────
void drawOverlay(cv::Mat& display, const std::string& label, double fps,
                 int frameCount) {
  cv::Scalar color(0, 255, 0);
  cv::rectangle(display, cv::Point(5, 5), cv::Point(290, 70),
                cv::Scalar(0, 0, 0), -1);
  cv::rectangle(display, cv::Point(5, 5), cv::Point(290, 70), color, 1);

  cv::putText(display, cv::format("%s FPS: %.1f", label.c_str(), fps),
              cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
  cv::putText(display, cv::format("Frame: %d", frameCount),
              cv::Point(12, 58), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

int main() {
  // ── Detect cameras ──────────────────────────────────────────────────────────
  auto indices = detectCameras();
  if (indices.empty()) {
    std::cerr << "No cameras detected\n";
    return -1;
  }
  std::cout << "Detected " << indices.size() << " camera(s)\n";

  // ── Initialize camera units ─────────────────────────────────────────────────
  std::vector<std::unique_ptr<CameraUnit>> cameras;

  for (int idx : indices) {
    auto cam = std::make_unique<CameraUnit>();
    cam->label = "CAM" + std::to_string(idx);

    // Create capture via class — all config in CaptureConfig struct
    CaptureConfig config;
    config.deviceIndex = idx;
    cam->capture = std::make_unique<OpenCvCapture>();
    if (!cam->capture->open(config)) {
      std::cerr << "Skipping camera " << idx << "\n";
      continue;
    }
    std::cout << cam->capture->getInfo() << "\n";

    // Video writer
    double actualFps = cam->capture->getActualFps();
    std::string outPath = "output_cam" + std::to_string(idx) + ".avi";
    cam->writer = cv::VideoWriter(
        outPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        actualFps, cv::Size(config.width, config.height), true);

    if (!cam->writer.isOpened()) {
      std::cerr << "Failed to open writer for camera " << idx << ", skipping\n";
      continue;
    }

    cameras.push_back(std::move(cam));
  }

  if (cameras.empty()) {
    std::cerr << "No cameras could be initialized\n";
    return -1;
  }

  // ── Start grab threads ──────────────────────────────────────────────────────
  std::atomic<bool> running{true};
  for (auto& cam : cameras) {
    cam->thread = std::thread(grabThread, std::ref(*cam), std::ref(running));
  }

  std::cout << "Press 'q' to stop...\n";

  // ── Main loop ───────────────────────────────────────────────────────────────
  while (true) {
    bool anyAlive = false;

    for (auto& cam : cameras) {
      if (!cam->alive) continue;  // skip dead cameras
      anyAlive = true;

      if (cam->hasNew) {
        {
          std::lock_guard<std::mutex> lock(cam->mtx);
          cam->latestFrame = cam->sharedFrame.clone();
          cam->hasNew = false;
        }
        cam->tracker.update();
        cam->frameCount++;
        cam->writer.write(cam->latestFrame);
      }

      if (!cam->latestFrame.empty()) {
        drawOverlay(cam->latestFrame, cam->label, cam->tracker.fps,
                    cam->frameCount);
        cv::imshow(cam->label, cam->latestFrame);
      }
    }

    if (!anyAlive) {
      std::cerr << "All cameras disconnected, exiting\n";
      break;
    }

    if (cv::waitKey(1) == 'q') break;
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────────
  running = false;
  for (auto& cam : cameras) {
    cam->thread.join();
    std::cout << cam->label << ": " << cam->frameCount << " frames saved\n";
    cam->capture->release();
    cam->writer.release();
  }
  cv::destroyAllWindows();
  return 0;
}
