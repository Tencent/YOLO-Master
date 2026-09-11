// ONNXRuntime backend for YOLO-Master-EsMoE-N (CPU plus optional CUDA,
// TensorRT and CoreML execution providers).
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "yolomaster.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace yolomaster {

class OrtBackend : public Backend {
public:
    // device: "cpu" | "cuda" | "trt" | "coreml" (accelerator failures fall back to CPU)
    OrtBackend(const std::string& model_path, int threads = 4, const std::string& device = "cpu");
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions alloc_;
    std::vector<std::string> in_names_s_, out_names_s_;
    std::vector<const char*> in_names_, out_names_;
    bool input_fp16_ = false;
};

} // namespace yolomaster
