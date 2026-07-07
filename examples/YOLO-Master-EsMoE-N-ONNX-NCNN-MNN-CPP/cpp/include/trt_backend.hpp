// TensorRT backend for YOLO-Master-EsMoE-N — GPU inference from a prebuilt .engine.
// Loads an engine built on-device by trtexec (jetson/10_trt_bench.sh) and runs it on CUDA.
#pragma once
#include "yolomaster.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace yolomaster {

class TrtBackend : public Backend {
public:
    explicit TrtBackend(const std::string& engine_path);
    ~TrtBackend() override;
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    cudaStream_t stream_ = nullptr;
    void* d_in_  = nullptr;
    void* d_out_ = nullptr;
    std::string in_name_, out_name_;
    int in_sz_ = 0;                        // input H (== W)
    int feat_dim_ = 0, num_anchors_ = 0;   // output [1, feat_dim, num_anchors]
    std::vector<float> h_out_;
};

} // namespace yolomaster
