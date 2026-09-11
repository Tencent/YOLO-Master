// MNN backend for YOLO-Master-EsMoE-N (Alibaba MNN; CPU plus optional
// OpenCL/Vulkan/CUDA forwards, depending on the SDK build).
// Mirrors the ncnn/ORT backends: model loads in the ctor, infer() reuses the shared
// letterbox + decode. Outputs are normalized to the channel-major
// [1, features, anchors] contract shared by ORT and NCNN.
// The runner requires float32 model input/output tensors; MNN quantized graphs
// remain usable when their public input/output tensors stay float32.
#pragma once
#include "yolomaster.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

namespace yolomaster {

class MnnBackend : public Backend {
public:
    // forward: "cpu" (default), "opencl", "vulkan", or "cuda" (build-dependent)
    MnnBackend(const std::string& model_path, int threads = 4, const std::string& forward = "cpu");
    ~MnnBackend() override;
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    std::shared_ptr<MNN::Interpreter> interp_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor*  input_    = nullptr;   // owned by the session
    MNN::Tensor*  output_   = nullptr;   // owned by the session
    int threads_;
};

} // namespace yolomaster
