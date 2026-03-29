#pragma once

#include "detection.hpp"

#include <cstdint>
#include <vector>

// Parse YOLOv8 raw output tensor and apply NMS
// outputData: raw float pointer from ONNX output tensor
// outputShape: expected [1, 84, 8400] for YOLOv8 with 80 COCO classes
// origWidth/origHeight: original frame dimensions for bbox scaling
// inputWidth/inputHeight: model input dimensions (e.g., 640x640)
std::vector<Detection> postprocess(const float* outputData,
                                   const std::vector<int64_t>& outputShape,
                                   float confThreshold, float nmsThreshold,
                                   int origWidth, int origHeight,
                                   int inputWidth, int inputHeight);
