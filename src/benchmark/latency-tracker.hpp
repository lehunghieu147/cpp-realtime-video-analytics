#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

// Records per-frame latencies and computes percentile statistics.
// Usage:
//   LatencyTracker tracker;
//   tracker.record(35.2);  // ms
//   tracker.record(42.1);
//   std::cout << "p50=" << tracker.p50() << " p99=" << tracker.p99();
class LatencyTracker {
public:
  void record(double latencyMs) {
    samples_.push_back(latencyMs);
    sorted_ = false;
  }

  double p50() const { return percentile(0.50); }
  double p95() const { return percentile(0.95); }
  double p99() const { return percentile(0.99); }

  double max() const {
    if (samples_.empty()) return 0.0;
    ensureSorted();
    return sorted_samples_.back();
  }

  double min() const {
    if (samples_.empty()) return 0.0;
    ensureSorted();
    return sorted_samples_.front();
  }

  double mean() const {
    if (samples_.empty()) return 0.0;
    double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
    return sum / static_cast<double>(samples_.size());
  }

  double stdDev() const {
    if (samples_.size() < 2) return 0.0;
    double avg = mean();
    double variance = 0.0;
    for (double s : samples_) {
      double diff = s - avg;
      variance += diff * diff;
    }
    variance /= static_cast<double>(samples_.size() - 1);
    return std::sqrt(variance);
  }

  size_t count() const { return samples_.size(); }

  void reset() {
    samples_.clear();
    sorted_samples_.clear();
    sorted_ = false;
  }

private:
  double percentile(double p) const {
    if (samples_.empty()) return 0.0;
    ensureSorted();

    double rank = p * static_cast<double>(sorted_samples_.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(rank));
    size_t upper = static_cast<size_t>(std::ceil(rank));

    if (lower == upper) return sorted_samples_[lower];

    // Linear interpolation between adjacent samples
    double frac = rank - static_cast<double>(lower);
    return sorted_samples_[lower] * (1.0 - frac) +
           sorted_samples_[upper] * frac;
  }

  void ensureSorted() const {
    if (sorted_) return;
    sorted_samples_ = samples_;
    std::sort(sorted_samples_.begin(), sorted_samples_.end());
    sorted_ = true;
  }

  std::vector<double> samples_;
  mutable std::vector<double> sorted_samples_;
  mutable bool sorted_ = false;
};
