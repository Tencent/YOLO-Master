// Domain-specific preprocessing for VisDrone edge inference (C++).
// Mirrors python/preprocess.py exactly so Python and C++ backends are consistent.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo {

constexpr int kLetterboxFill = 114;  // standard YOLO pad grey

struct LetterboxInfo {
  float ratio;
  float pad_w;
  float pad_h;
  int orig_w;
  int orig_h;
};

// Aspect-ratio-preserving resize + pad to (imgsz x imgsz).
// Returns the padded BGR image; geometric params are written to `info`.
inline cv::Mat letterbox(const cv::Mat& bgr, int imgsz, LetterboxInfo& info) {
  const int h = bgr.rows, w = bgr.cols;
  const float ratio = std::min(float(imgsz) / h, float(imgsz) / w);
  const int new_w = static_cast<int>(std::round(w * ratio));
  const int new_h = static_cast<int>(std::round(h * ratio));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  const int pad_w = imgsz - new_w;
  const int pad_h = imgsz - new_h;
  const int top = pad_h / 2;
  const int left = pad_w / 2;
  cv::Mat padded(imgsz, imgsz, bgr.type(), cv::Scalar(kLetterboxFill, kLetterboxFill, kLetterboxFill));
  resized.copyTo(padded(cv::Rect(left, top, new_w, new_h)));

  info.ratio = ratio;
  info.pad_w = static_cast<float>(left);
  info.pad_h = static_cast<float>(top);
  info.orig_w = w;
  info.orig_h = h;
  return padded;
}

// BGR uint8 (imgsz x imgsz) -> CHW RGB float32 [0,1], row-major.
inline std::vector<float> hwc_to_nchw(const cv::Mat& padded_bgr) {
  cv::Mat rgb;
  cv::cvtColor(padded_bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat f;
  rgb.convertTo(f, CV_32F, 1.0 / 255.0);
  // HWC -> CHW
  std::vector<float> out(3 * f.rows * f.cols);
  const int H = f.rows, W = f.cols;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      const float* row = f.ptr<float>(y);
      for (int x = 0; x < W; ++x) {
        out[c * H * W + y * W + x] = row[x * 3 + c];
      }
    }
  }
  return out;
}

// Map boxes from letterboxed image space back into original image space.
inline void unscale_box(float* xyxy, const LetterboxInfo& info) {
  xyxy[0] = (xyxy[0] - info.pad_w) / info.ratio;
  xyxy[1] = (xyxy[1] - info.pad_h) / info.ratio;
  xyxy[2] = (xyxy[2] - info.pad_w) / info.ratio;
  xyxy[3] = (xyxy[3] - info.pad_h) / info.ratio;
  xyxy[0] = std::max(0.0f, std::min(xyxy[0], float(info.orig_w)));
  xyxy[1] = std::max(0.0f, std::min(xyxy[1], float(info.orig_h)));
  xyxy[2] = std::max(0.0f, std::min(xyxy[2], float(info.orig_w)));
  xyxy[3] = std::max(0.0f, std::min(xyxy[3], float(info.orig_h)));
}

}  // namespace yolo
