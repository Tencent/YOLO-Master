// MNN backend for YOLO-Master-EsMoE-N (Alibaba MNN; CPU now, CUDA optional later).
// Mirrors the ncnn/ORT backends: model loads in the ctor, infer() reuses the shared
// letterbox + decode. Output is channel-major [1, 4+nc, anchors] (same contract as ORT/ncnn).
#pragma once
#include "yolomaster.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

namespace yolomaster {

class MnnBackend : public Backend {
public:
    // forward: "cpu" (default) | "cuda" (requires an MNN built with CUDA)
    MnnBackend(const std::string& model_path, int threads = 4, const std::string& forward = "cpu",
               Precision precision = Precision::Auto);
    ~MnnBackend() override;
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    std::shared_ptr<MNN::Interpreter> interp_;
    MNN::Session* session_ = nullptr;
    bool end2end_ = false;   // NMS-free [1,num_det,6] output (yolo26 lineage)
    MNN::Tensor*  input_    = nullptr;   // owned by the session
    MNN::Tensor*  output_   = nullptr;   // owned by the session
    int threads_;
};

} // namespace yolomaster
