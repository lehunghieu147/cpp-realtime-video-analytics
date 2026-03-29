#include "pre-process.hpp"

#include <cstring>

std::vector<float> preprocess(const cv::Mat& frame, int targetW, int targetH) {
  // Resize to model input dimensions (typically 640x640)
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_LINEAR);

  // BGR→RGB color conversion
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // Normalize to [0, 1] and convert to float32
  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);

  // HWC→CHW transpose: channel-first layout for ONNX
  // Output layout: [C, H, W] = [3, targetH, targetW]
  const int channels = 3;
  const int area = targetW * targetH;
  std::vector<float> chw(channels * area);

  // Split into separate channels, then copy each into contiguous memory
  std::vector<cv::Mat> channelMats(channels);
  cv::split(normalized, channelMats);

  for (int c = 0; c < channels; c++) {
    std::memcpy(chw.data() + c * area, channelMats[c].data, area * sizeof(float));
  }

  return chw;
}
