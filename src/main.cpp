#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

struct CameraState {
  cv::Mat frame;
  std::mutex mtx;
  std::atomic<bool> hasNew{false};
};

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

void grabThread(cv::VideoCapture &cap, CameraState &state,
                std::atomic<bool> &running) {
  cv::Mat tmp;
  while (running) {
    cap >> tmp;
    if (tmp.empty())
      continue;

    std::lock_guard<std::mutex> lock(state.mtx);
    state.frame = tmp.clone();
    state.hasNew = true;
  }
}

void setupCamera(cv::VideoCapture &cap) {
  cap.set(cv::CAP_PROP_FOURCC,
          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv::CAP_PROP_FPS, 30);
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
}

void drawOverlay(cv::Mat &display, double fps, int frameCount) {
  cv::Scalar color(0, 255, 0);
  cv::rectangle(display, cv::Point(5, 5), cv::Point(290, 70),
                cv::Scalar(0, 0, 0), -1);
  cv::rectangle(display, cv::Point(5, 5), cv::Point(290, 70), color, 1);

  cv::putText(display, cv::format("FPS: %.1f", fps),
              cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
  cv::putText(display, cv::format("Frame: %d", frameCount),
              cv::Point(12, 58), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

int main() {
  cv::VideoCapture cap0(0);
  if (!cap0.isOpened()) {
    std::cerr << "Cannot open camera\n";
    return -1;
  }

  setupCamera(cap0);

  // Print camera info
  int fourcc = (int)cap0.get(cv::CAP_PROP_FOURCC);
  char fmt[5] = {0};
  memcpy(fmt, &fourcc, 4);
  std::cout << "cam0: "
            << (int)cap0.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << (int)cap0.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
            << cap0.get(cv::CAP_PROP_FPS) << " FPS"
            << " format=" << fmt << "\n";

  // Video writer
  const int W = 640, H = 480;
  cv::VideoWriter writer("output_cam0.avi",
                         cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                         cap0.get(cv::CAP_PROP_FPS), cv::Size(W, H), true);

  if (!writer.isOpened()) {
    std::cerr << "Cannot open VideoWriter\n";
    return -1;
  }

  // Start grab thread
  CameraState state;
  std::atomic<bool> running{true};
  std::thread t0(grabThread, std::ref(cap0), std::ref(state), std::ref(running));

  FpsTracker tracker;
  int frameCount = 0;
  cv::Mat frame;

  std::cout << "Press 'q' to stop...\n";

  // Main loop
  while (true) {
    if (state.hasNew) {
      {
        std::lock_guard<std::mutex> lock(state.mtx);
        frame = state.frame.clone();
        state.hasNew = false;
      }
      tracker.update();
      frameCount++;
      writer.write(frame);
    }

    if (!frame.empty()) {
      cv::Mat display = frame.clone();
      drawOverlay(display, tracker.fps, frameCount);
      cv::imshow("Camera", display);
    }

    if (cv::waitKey(1) == 'q')
      break;
  }

  // Cleanup
  running = false;
  t0.join();

  std::cout << "cam0: " << frameCount << " frames saved\n";
  cap0.release();
  writer.release();
  cv::destroyAllWindows();
  return 0;
}