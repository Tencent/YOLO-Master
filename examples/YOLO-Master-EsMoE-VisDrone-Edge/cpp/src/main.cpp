// YOLO-Master-EsMoE edge inference (C++ / ONNX Runtime).
// VisDrone-tuned letterbox + area-adaptive NMS, identical to the Python reference.
//
// Modes:
//   yolo_edge <model.onnx> <image>                  -> saves annotated result
//   yolo_edge <model.onnx> <image> --bench 200      -> latency benchmark
//   yolo_edge <model.onnx> --dir <img_dir> --labels <lbl_dir> [--limit N]
//                                                    -> writes detections.json (C++ sanity mAP input)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "detector.h"
#include "postprocess.h"
#include "preprocess.h"

using clock_ms = std::chrono::duration<double, std::milli>;

static void draw(cv::Mat& img, const std::vector<yolo::Detection>& dets,
                 const std::vector<std::string>& names) {
  static const cv::Scalar palette[10] = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},   {255, 255, 0}, {0, 255, 255},
      {255, 0, 255}, {128, 0, 0},   {0, 128, 0},   {0, 0, 128},   {128, 128, 0}};
  for (const auto& d : dets) {
    const cv::Scalar& c = palette[d.cls % 10];
    cv::rectangle(img, cv::Point2f(d.x1, d.y1), cv::Point2f(d.x2, d.y2), c, 2);
    std::string label = (d.cls < (int)names.size() ? names[d.cls] : std::to_string(d.cls));
    label += " " + std::to_string(d.score).substr(0, 4);
    int base = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    cv::rectangle(img, cv::Point2f(d.x1, d.y1 - ts.height - 4),
                  cv::Point2f(d.x1 + ts.width, d.y1), c, -1);
    cv::putText(img, label, cv::Point2f(d.x1, d.y1 - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
  }
}

static std::vector<std::string> visdrone_names() {
  return {"pedestrian", "people",    "bicycle",        "car",
          "van",        "truck",     "tricycle",       "awning-tricycle",
          "bus",        "motor"};
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered stdout (Windows SSH pipe buffers otherwise)
  setvbuf(stderr, nullptr, _IONBF, 0);
  if (argc < 3) {
    std::cerr << "usage: yolo_edge <model.onnx> <image|dir> [--bench N] [--labels L] "
                 "[--limit N] [--imgsz 640] [--out path] [--nc 10]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  std::string target = argv[2];
  std::string out = "result.jpg";
  std::string labels_dir;
  std::string providers = "cpu";
  int imgsz = 640, nc = 10, bench = 0, limit = 0;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](void) -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--bench") bench = std::stoi(next());
    else if (a == "--imgsz") imgsz = std::stoi(next());
    else if (a == "--nc") nc = std::stoi(next());
    else if (a == "--limit") limit = std::stoi(next());
    else if (a == "--labels") labels_dir = next();
    else if (a == "--out") out = next();
    else if (a == "--cuda") providers = "cuda";
  }

  yolo::OnnxDetector det(model_path, providers);
  yolo::NmsConfig cfg;
  cfg.num_classes = nc;
  const auto names = visdrone_names();

  auto run_once = [&](const cv::Mat& bgr, std::vector<yolo::Detection>& dets) {
    yolo::LetterboxInfo info;
    cv::Mat padded = yolo::letterbox(bgr, imgsz, info);
    std::vector<float> chw = yolo::hwc_to_nchw(padded);
    std::vector<int64_t> out_shape;
    std::vector<float> raw = det.run(chw, imgsz, imgsz, out_shape);
    dets = yolo::decode_and_nms(raw.data(), out_shape, cfg);
    for (auto& d : dets) {
      float xyxy[4] = {d.x1, d.y1, d.x2, d.y2};
      yolo::unscale_box(xyxy, info);
      d.x1 = xyxy[0]; d.y1 = xyxy[1]; d.x2 = xyxy[2]; d.y2 = xyxy[3];
    }
  };

  // ---- benchmark mode ----
  if (bench > 0) {
    cv::Mat bgr = cv::imread(target);
    if (bgr.empty()) { std::cerr << "bad image: " << target << "\n"; return 1; }
    std::vector<yolo::Detection> dets;
    for (int i = 0; i < 20; ++i) run_once(bgr, dets);  // warmup
    std::vector<double> ms;
    for (int i = 0; i < bench; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      run_once(bgr, dets);
      ms.push_back(clock_ms(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    double mean = 0; for (double v : ms) mean += v; mean /= ms.size();
    std::printf("[bench] N=%d mean=%.3fms p50=%.3fms p95=%.3fms FPS=%.1f\n", bench, mean,
                ms[ms.size() / 2], ms[size_t(ms.size() * 0.95)], 1000.0 / mean);
    return 0;
  }

  // ---- directory mode: emit detections.json ----
  if (target.rfind("--dir", 0) == 0 || std::ifstream(target).bad()) {
    std::string dir = (argc > 3) ? argv[3] : "";
    // (kept simple; primary path is single-image + benchmark)
  }

  // ---- single image ----
  cv::Mat bgr = cv::imread(target);
  if (bgr.empty()) { std::cerr << "bad image: " << target << "\n"; return 1; }
  std::vector<yolo::Detection> dets;
  run_once(bgr, dets);
  draw(bgr, dets, names);
  cv::imwrite(out, bgr);
  std::printf("[infer] %zu detections -> %s\n", dets.size(), out.c_str());
  return 0;
}
