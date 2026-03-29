#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Preprocess a BGR frame for YOLOv8 ONNX inference
// Steps: resize to target → BGR→RGB → normalize [0,1] → HWC→CHW
// Returns float vector ready for ONNX input tensor
std::vector<float> preprocess(const cv::Mat& frame, int targetW, int targetH);
