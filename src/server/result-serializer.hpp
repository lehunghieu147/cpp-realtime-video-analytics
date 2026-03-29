#pragma once

#include "pipeline/frame-data.hpp"

#include <nlohmann/json.hpp>
#include <string>

// Serialize AnalyticsResult to compact JSON for SSE streaming
// Example output:
// {"frame_id":42,"detections":[{"class":"person","confidence":0.94,
//   "bbox":{"x":100,"y":50,"w":200,"h":300}}],"latency_ms":35.2,"fps":21.5}
inline nlohmann::json serializeResult(const AnalyticsResult& result,
                                       double currentFps) {
  nlohmann::json detections = nlohmann::json::array();
  for (const auto& det : result.detections) {
    detections.push_back({
        {"class", getClassName(det.classId)},
        {"confidence", std::round(det.confidence * 100) / 100},  // 2 decimals
        {"bbox",
         {{"x", static_cast<int>(det.bbox.x)},
          {"y", static_cast<int>(det.bbox.y)},
          {"w", static_cast<int>(det.bbox.width)},
          {"h", static_cast<int>(det.bbox.height)}}}
    });
  }

  return {
      {"frame_id", result.frameId},
      {"detections", detections},
      {"latency_ms", std::round(result.inferenceLatencyMs * 10) / 10},
      {"fps", std::round(currentFps * 10) / 10}
  };
}
