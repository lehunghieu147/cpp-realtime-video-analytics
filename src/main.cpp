#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "capture/opencv-capture.hpp"
#include "utils/fps-tracker.hpp"

// ── All state for a single camera ─────────────────────────────────────────────
// Double buffer design:
//   buffers[0] and buffers[1] are pre-allocated cv::Mat.
//   writeIdx = index grab thread writes into (owned by grab thread).
//   When grab thread finishes a frame → lock, swap writeIdx, set hasNew.
//   Main thread reads from buffers[1 - writeIdx] (the "other" buffer).
//   Result: only 1 clone (cap→buffer) instead of 2. Swap is just an int flip.
struct CameraUnit {
  std::unique_ptr<IVideoCapture> capture;
  std::string label;
  cv::VideoWriter writer;
  std::thread thread;
  FpsTracker tracker;
  int frameCount = 0;
  cv::Mat displayFrame;

  // Double buffer: grab thread writes to buffers[writeIdx],
  // main thread reads from buffers[1 - writeIdx]
  cv::Mat buffers[2];
  int writeIdx = 0;
  std::mutex mtx;
  std::atomic<bool> hasNew{false};
  std::atomic<bool> alive{true};
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
  int failCount = 0;
  const int kMaxFails = 30;

  while (running && cam.alive) {
    // Read directly into the write buffer — no intermediate tmp needed
    cv::Mat& writeBuf = cam.buffers[cam.writeIdx];

    if (!cam.capture->read(writeBuf)) {
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
    failCount = 0;

    // Swap: flip writeIdx so main thread reads the buffer we just filled
    {
      std::lock_guard<std::mutex> lock(cam.mtx);
      cam.writeIdx = 1 - cam.writeIdx;  // 0→1 or 1→0
      cam.hasNew = true;
    }
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
          // Read from the buffer that grab thread is NOT writing to
          cam->displayFrame = cam->buffers[1 - cam->writeIdx];
          cam->hasNew = false;
        }
        cam->tracker.update();
        cam->frameCount++;
        cam->writer.write(cam->displayFrame);
      }

      if (!cam->displayFrame.empty()) {
        drawOverlay(cam->displayFrame, cam->label, cam->tracker.fps,
                    cam->frameCount);
        cv::imshow(cam->label, cam->displayFrame);
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
