#pragma once

#include <chrono>

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
