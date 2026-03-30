#pragma once

#include "latency-tracker.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Generate a Markdown benchmark report from collected metrics
struct BenchmarkResults {
  double fpsMean = 0.0;
  uint64_t totalFrames = 0;
  LatencyTracker latency;
  std::string modelName = "yolov8n";
  int inputWidth = 640;
  int inputHeight = 640;
  std::string device = "CPU";
};

inline std::string generateReport(const BenchmarkResults& r) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1);

  ss << "# Benchmark Results\n\n";

  ss << "## Configuration\n";
  ss << "| Setting | Value |\n";
  ss << "|---------|-------|\n";
  ss << "| Model | " << r.modelName << " |\n";
  ss << "| Input Size | " << r.inputWidth << "x" << r.inputHeight << " |\n";
  ss << "| Device | " << r.device << " |\n";
  ss << "| Frames Processed | " << r.totalFrames << " |\n\n";

  ss << "## Performance\n";
  ss << "| Metric | Value |\n";
  ss << "|--------|-------|\n";
  ss << "| FPS (mean) | " << r.fpsMean << " |\n";
  ss << "| Latency p50 | " << r.latency.p50() << " ms |\n";
  ss << "| Latency p95 | " << r.latency.p95() << " ms |\n";
  ss << "| Latency p99 | " << r.latency.p99() << " ms |\n";
  ss << "| Latency max | " << r.latency.max() << " ms |\n";
  ss << "| Latency min | " << r.latency.min() << " ms |\n";
  ss << "| Latency mean | " << r.latency.mean() << " ms |\n";
  ss << "| Latency stddev | " << r.latency.stdDev() << " ms |\n";
  ss << "| Total samples | " << r.latency.count() << " |\n";

  return ss.str();
}

inline bool writeReport(const BenchmarkResults& r,
                         const std::string& outputPath) {
  std::ofstream file(outputPath);
  if (!file) {
    std::cerr << "Failed to write report to: " << outputPath << "\n";
    return false;
  }
  file << generateReport(r);
  std::cout << "Benchmark report written to: " << outputPath << "\n";
  return true;
}
