#pragma once

#include "video-capture.hpp"

class OpenCvCapture : public IVideoCapture {
public:
  bool open(const CaptureConfig& config) override;
  bool read(cv::Mat& frame) override;
  void release() override;
  bool isOpened() const override;
  double getActualFps() const override;
  std::string getInfo() const override;

private:
  cv::VideoCapture cap_;
  CaptureConfig config_;
};
