#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

// ── FPS tracker ───────────────────────────────────────────────────────────────
struct FpsTracker {
  using Clock = std::chrono::steady_clock;
  Clock::time_point timer = Clock::now();
  int counter = 0;
  double fps = 0.0;

  void update() {
    counter++;
    auto now = Clock::now();
    double elapsed = std::chrono::duration<double>(now - timer).count();
    if (elapsed >= 1.0) {
      fps = counter / elapsed;
      counter = 0;
      timer = now;
    }
  }
};

// ── All state for a single camera ─────────────────────────────────────────────
struct CameraUnit {
  int deviceIndex;
  std::string label;
  cv::VideoCapture cap;
  cv::VideoWriter writer;
  std::thread thread;
  FpsTracker tracker;
  int frameCount = 0;
  cv::Mat latestFrame;

  // Shared state between grab thread and main thread
  cv::Mat sharedFrame;
  std::mutex mtx;
  std::atomic<bool> hasNew{false};
};

// ── Detect available cameras by probing device indices ────────────────────────
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

// ── Grab thread: continuously captures frames from a camera ───────────────────
void grabThread(CameraUnit &cam, std::atomic<bool> &running) {
  cv::Mat tmp;
  while (running) {
    cam.cap >> tmp;
    if (tmp.empty()) {
      //std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    std::lock_guard<std::mutex> lock(cam.mtx);
    cam.sharedFrame = tmp.clone();
    cam.hasNew = true;
  }
}

// ── Camera setup helper ───────────────────────────────────────────────────────
void setupCamera(cv::VideoCapture &cap) {
  cap.set(cv::CAP_PROP_FOURCC,
          cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv::CAP_PROP_FPS, 30);
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
}

// ── Print actual camera properties ────────────────────────────────────────────
void printCameraInfo(const CameraUnit &cam) {
  int fourcc = static_cast<int>(cam.cap.get(cv::CAP_PROP_FOURCC));
  char fmt[5] = {0};
  std::memcpy(fmt, &fourcc, 4);
  std::cout << cam.label << " (index " << cam.deviceIndex << "): "
            << static_cast<int>(cam.cap.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
            << static_cast<int>(cam.cap.get(cv::CAP_PROP_FRAME_HEIGHT)) << " @ "
            << cam.cap.get(cv::CAP_PROP_FPS) << " FPS"
            << " format=" << fmt << "\n";
}

// ── Draw overlay (FPS + frame count) onto display frame ──────────────────────
void drawOverlay(cv::Mat &display, const std::string &label, double fps,
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
  const int W = 640, H = 480;
  std::vector<std::unique_ptr<CameraUnit>> cameras;

  for (int idx : indices) {
    auto cam = std::make_unique<CameraUnit>();
    cam->deviceIndex = idx;
    cam->label = "CAM" + std::to_string(idx);

    cam->cap.open(idx);
    if (!cam->cap.isOpened()) {
      std::cerr << "Failed to open camera " << idx << ", skipping\n";
      continue;
    }

    setupCamera(cam->cap);
    printCameraInfo(*cam);

    double actualFps = cam->cap.get(cv::CAP_PROP_FPS);
    if (actualFps <= 0) actualFps = 30.0;

    std::string outPath = "output_cam" + std::to_string(idx) + ".avi";
    cam->writer = cv::VideoWriter(
        outPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        actualFps, cv::Size(W, H), true);

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
  for (auto &cam : cameras) {
    cam->thread = std::thread(grabThread, std::ref(*cam), std::ref(running));
  }

  std::cout << "Press 'q' to stop...\n";

  // ── Main loop ───────────────────────────────────────────────────────────────
  while (true) {
    for (auto &cam : cameras) {
      // Fetch latest frame if available
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

      // Display
      if (!cam->latestFrame.empty()) {
        drawOverlay(cam->latestFrame, cam->label, cam->tracker.fps,
                    cam->frameCount);
        cv::imshow(cam->label, cam->latestFrame);
      }
    }

    if (cv::waitKey(1) == 'q') break;
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────────
  running = false;
  for (auto &cam : cameras) {
    cam->thread.join();
    std::cout << cam->label << ": " << cam->frameCount << " frames saved\n";
    cam->cap.release();
    cam->writer.release();
  }
  cv::destroyAllWindows();
  return 0;
}