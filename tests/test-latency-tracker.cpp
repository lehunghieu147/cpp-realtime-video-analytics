#include <iostream>
#include <cmath>
#include <string>

#include "benchmark/latency-tracker.hpp"
#include "benchmark/benchmark-report.hpp"

// Test utilities
namespace test {

enum class TestResult {
  PASS,
  FAIL
};

struct TestStats {
  int total = 0;
  int passed = 0;
  int failed = 0;

  void reset() {
    total = 0;
    passed = 0;
    failed = 0;
  }

  float passRate() const {
    if (total == 0) return 0.0f;
    return static_cast<float>(passed) / total * 100.0f;
  }
};

TestStats stats;

void assert_true(bool condition, const std::string& testName) {
  stats.total++;
  if (condition) {
    std::cout << "  [PASS] " << testName << "\n";
    stats.passed++;
  } else {
    std::cout << "  [FAIL] " << testName << "\n";
    stats.failed++;
  }
}

template <typename T>
void assert_eq(T actual, T expected, const std::string& testName) {
  assert_true(actual == expected, testName);
  if (actual != expected) {
    std::cout << "    Expected: " << expected << ", Got: " << actual << "\n";
  }
}

void assert_near(double actual, double expected, double tolerance,
                 const std::string& testName) {
  bool pass = std::fabs(actual - expected) <= tolerance;
  assert_true(pass, testName);
  if (!pass) {
    std::cout << "    Expected: " << expected << " (±" << tolerance
              << "), Got: " << actual << "\n";
  }
}

}  // namespace test

// Test 1: Empty tracker
void testEmptyTracker() {
  std::cout << "Test 1: Empty tracker\n";
  LatencyTracker tracker;

  test::assert_eq(tracker.count(), static_cast<size_t>(0), "count() == 0");
  test::assert_eq(tracker.p50(), 0.0, "p50() == 0");
  test::assert_eq(tracker.p95(), 0.0, "p95() == 0");
  test::assert_eq(tracker.p99(), 0.0, "p99() == 0");
  test::assert_eq(tracker.max(), 0.0, "max() == 0");
  test::assert_eq(tracker.min(), 0.0, "min() == 0");
  test::assert_eq(tracker.mean(), 0.0, "mean() == 0");
  test::assert_eq(tracker.stdDev(), 0.0, "stdDev() == 0");
  std::cout << "\n";
}

// Test 2: Single sample
void testSingleSample() {
  std::cout << "Test 2: Single sample\n";
  LatencyTracker tracker;
  tracker.record(42.5);

  test::assert_eq(tracker.count(), static_cast<size_t>(1), "count() == 1");
  test::assert_eq(tracker.min(), 42.5, "min() == 42.5");
  test::assert_eq(tracker.max(), 42.5, "max() == 42.5");
  test::assert_eq(tracker.mean(), 42.5, "mean() == 42.5");
  test::assert_eq(tracker.p50(), 42.5, "p50() == 42.5");
  test::assert_eq(tracker.p95(), 42.5, "p95() == 42.5");
  test::assert_eq(tracker.p99(), 42.5, "p99() == 42.5");
  test::assert_eq(tracker.stdDev(), 0.0, "stdDev() == 0");
  std::cout << "\n";
}

// Test 3: Known data set [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
// p50 should be 55.0 (linear interpolation between indices 4 and 5)
// p95 should be 95.5
// p99 should be 99.1
void testKnownData() {
  std::cout << "Test 3: Known data set percentiles\n";
  LatencyTracker tracker;
  double data[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  for (double d : data) {
    tracker.record(d);
  }

  test::assert_eq(tracker.count(), static_cast<size_t>(10), "count() == 10");
  test::assert_eq(tracker.min(), 10.0, "min() == 10");
  test::assert_eq(tracker.max(), 100.0, "max() == 100");

  // p50 at 0.5 * (10-1) = 4.5, interpolate between indices 4 and 5
  // sorted: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
  // index:  0,  1,  2,  3,  4,  5,  6,  7,  8,  9
  // value at 4.5: 50 * 0.5 + 60 * 0.5 = 55.0
  test::assert_near(tracker.p50(), 55.0, 0.01, "p50() ≈ 55.0");

  // p95 at 0.95 * 9 = 8.55, interpolate between indices 8 and 9
  // value at 8.55: 90 * 0.45 + 100 * 0.55 = 40.5 + 55.0 = 95.5
  test::assert_near(tracker.p95(), 95.5, 0.01, "p95() ≈ 95.5");

  // p99 at 0.99 * 9 = 8.91, interpolate between indices 8 and 9
  // value at 8.91: 90 * 0.09 + 100 * 0.91 = 8.1 + 91.0 = 99.1
  test::assert_near(tracker.p99(), 99.1, 0.01, "p99() ≈ 99.1");
  std::cout << "\n";
}

// Test 4: Mean calculation
void testMeanCalculation() {
  std::cout << "Test 4: Mean calculation\n";
  LatencyTracker tracker;
  tracker.record(10.0);
  tracker.record(20.0);
  tracker.record(30.0);

  test::assert_eq(tracker.mean(), 20.0, "mean([10, 20, 30]) == 20.0");
  std::cout << "\n";
}

// Test 5: Standard deviation calculation
// Data: [10, 20, 30, 40, 50]
// Mean = 30
// Deviations: -20, -10, 0, 10, 20
// Squared deviations: 400, 100, 0, 100, 400
// Sum of squared deviations: 1000
// Sample variance (n-1): 1000 / 4 = 250
// Sample stddev: sqrt(250) ≈ 15.811
void testStdDevCalculation() {
  std::cout << "Test 5: Standard deviation calculation\n";
  LatencyTracker tracker;
  double data[] = {10.0, 20.0, 30.0, 40.0, 50.0};
  for (double d : data) {
    tracker.record(d);
  }

  double expectedStdDev = std::sqrt(250.0);
  test::assert_near(tracker.stdDev(), expectedStdDev, 0.01,
                    "stdDev([10,20,30,40,50]) ≈ 15.81");
  std::cout << "\n";
}

// Test 6: Max and Min
void testMaxMin() {
  std::cout << "Test 6: Max and Min\n";
  LatencyTracker tracker;
  double data[] = {42.1, 5.8, 99.2, 12.3, 67.5};
  for (double d : data) {
    tracker.record(d);
  }

  test::assert_eq(tracker.min(), 5.8, "min() == 5.8");
  test::assert_eq(tracker.max(), 99.2, "max() == 99.2");
  std::cout << "\n";
}

// Test 7: Count increments
void testCountIncrement() {
  std::cout << "Test 7: Count increments\n";
  LatencyTracker tracker;

  for (int i = 0; i < 50; i++) {
    tracker.record(10.0 + i);
    test::assert_eq(tracker.count(), static_cast<size_t>(i + 1),
                    "count() after " + std::to_string(i + 1) + " records");
  }
  std::cout << "\n";
}

// Test 8: Reset clears all data
void testReset() {
  std::cout << "Test 8: Reset clears all data\n";
  LatencyTracker tracker;
  tracker.record(10.0);
  tracker.record(20.0);
  tracker.record(30.0);

  test::assert_eq(tracker.count(), static_cast<size_t>(3), "count() == 3 before reset");

  tracker.reset();

  test::assert_eq(tracker.count(), static_cast<size_t>(0), "count() == 0 after reset");
  test::assert_eq(tracker.p50(), 0.0, "p50() == 0 after reset");
  test::assert_eq(tracker.p95(), 0.0, "p95() == 0 after reset");
  test::assert_eq(tracker.p99(), 0.0, "p99() == 0 after reset");
  test::assert_eq(tracker.max(), 0.0, "max() == 0 after reset");
  test::assert_eq(tracker.min(), 0.0, "min() == 0 after reset");
  test::assert_eq(tracker.mean(), 0.0, "mean() == 0 after reset");
  test::assert_eq(tracker.stdDev(), 0.0, "stdDev() == 0 after reset");
  std::cout << "\n";
}

// Test 9: Report generation contains expected fields
void testReportGeneration() {
  std::cout << "Test 9: Report generation\n";
  LatencyTracker tracker;
  double data[] = {10.0, 20.0, 30.0, 40.0, 50.0};
  for (double d : data) {
    tracker.record(d);
  }

  BenchmarkResults results;
  results.fpsMean = 60.5;
  results.totalFrames = 5;
  results.latency = tracker;
  results.modelName = "yolov8n";
  results.inputWidth = 640;
  results.inputHeight = 480;
  results.device = "CPU";

  std::string report = generateReport(results);

  // Check for key sections
  test::assert_true(report.find("# Benchmark Results") != std::string::npos,
                    "Report contains title");
  test::assert_true(report.find("## Configuration") != std::string::npos,
                    "Report contains Configuration section");
  test::assert_true(report.find("## Performance") != std::string::npos,
                    "Report contains Performance section");

  // Check for key metrics
  test::assert_true(report.find("FPS (mean)") != std::string::npos,
                    "Report contains FPS metric");
  test::assert_true(report.find("Latency p50") != std::string::npos,
                    "Report contains p50 metric");
  test::assert_true(report.find("Latency p95") != std::string::npos,
                    "Report contains p95 metric");
  test::assert_true(report.find("Latency p99") != std::string::npos,
                    "Report contains p99 metric");
  test::assert_true(report.find("Latency max") != std::string::npos,
                    "Report contains max metric");
  test::assert_true(report.find("Latency min") != std::string::npos,
                    "Report contains min metric");
  test::assert_true(report.find("Latency mean") != std::string::npos,
                    "Report contains mean metric");
  test::assert_true(report.find("Latency stddev") != std::string::npos,
                    "Report contains stddev metric");
  test::assert_true(report.find("Total samples") != std::string::npos,
                    "Report contains sample count");

  // Check for configuration values
  test::assert_true(report.find("yolov8n") != std::string::npos,
                    "Report contains model name");
  test::assert_true(report.find("640x480") != std::string::npos,
                    "Report contains input dimensions");
  test::assert_true(report.find("CPU") != std::string::npos,
                    "Report contains device info");
  std::cout << "\n";
}

// Test 10: Multiple percentile queries don't affect state
void testMultiplePercentileQueries() {
  std::cout << "Test 10: Multiple percentile queries\n";
  LatencyTracker tracker;
  double data[] = {10, 20, 30, 40, 50};
  for (double d : data) {
    tracker.record(d);
  }

  // Query multiple times
  double p50_1 = tracker.p50();
  double p95_1 = tracker.p95();
  double p99_1 = tracker.p99();

  double p50_2 = tracker.p50();
  double p95_2 = tracker.p95();
  double p99_2 = tracker.p99();

  test::assert_eq(p50_1, p50_2, "p50() consistent across multiple queries");
  test::assert_eq(p95_1, p95_2, "p95() consistent across multiple queries");
  test::assert_eq(p99_1, p99_2, "p99() consistent across multiple queries");
  std::cout << "\n";
}

int main() {
  std::cout << "===== LatencyTracker & BenchmarkReport Tests =====\n\n";

  testEmptyTracker();
  testSingleSample();
  testKnownData();
  testMeanCalculation();
  testStdDevCalculation();
  testMaxMin();
  testCountIncrement();
  testReset();
  testReportGeneration();
  testMultiplePercentileQueries();

  std::cout << "===== Test Summary =====\n";
  std::cout << "Total: " << test::stats.total << "\n";
  std::cout << "Passed: " << test::stats.passed << "\n";
  std::cout << "Failed: " << test::stats.failed << "\n";
  std::cout << "Pass Rate: " << test::stats.passRate() << "%\n";

  return test::stats.failed > 0 ? 1 : 0;
}
