#include "trt_backend.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

struct TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
    }
};
static TrtLogger g_logger;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); } while (0)

TrtBackend::TrtBackend(const std::string& engine_path) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_)
        throw std::runtime_error("failed to deserialize engine (built for a different GPU arch / TRT version?)");
    ctx_.reset(engine_->createExecutionContext());
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // discover I/O tensors (TensorRT 10 named-tensor API)
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* nm = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(nm);
        if (engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT) {
            in_name_ = nm; in_sz_ = dims.d[2];                    // [1,3,H,W]
        } else {
            out_name_ = nm; feat_dim_ = dims.d[1]; num_anchors_ = dims.d[2];  // [1,feat,anchors]
        }
    }
    if (in_sz_ <= 0 || feat_dim_ <= 0 || num_anchors_ <= 0)
        throw std::runtime_error("unexpected engine I/O shape");
    fixed_imgsz = in_sz_;
    active_ep = "TRT-CUDA";

    CUDA_CHECK(cudaMalloc(&d_in_,  size_t(3) * in_sz_ * in_sz_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_, size_t(feat_dim_) * num_anchors_ * sizeof(float)));
    h_out_.resize(size_t(feat_dim_) * num_anchors_);
    ctx_->setTensorAddress(in_name_.c_str(),  d_in_);
    ctx_->setTensorAddress(out_name_.c_str(), d_out_);
}

TrtBackend::~TrtBackend() {
    if (d_in_)   cudaFree(d_in_);
    if (d_out_)  cudaFree(d_out_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Detection> TrtBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = letterbox(bgr, in_sz_, lb);              // in_sz_ x in_sz_, BGR
    const int sz = in_sz_, hw = sz * sz;
    std::vector<float> in(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;                  // BGR -> RGB /255, NCHW
            const int idx = y * sz + x;
            in[idx]        = px[2] * (1.0f / 255);
            in[hw + idx]   = px[1] * (1.0f / 255);
            in[2 * hw + idx] = px[0] * (1.0f / 255);
        }
    }
    pre_ms = ms_since(t0);

    auto t1 = clk::now();
    CUDA_CHECK(cudaMemcpyAsync(d_in_, in.data(), in.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("TRT enqueueV3 failed");
    CUDA_CHECK(cudaMemcpyAsync(h_out_.data(), d_out_, h_out_.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    infer_ms = ms_since(t1);

    auto t2 = clk::now();
    auto dets = decode(h_out_.data(), feat_dim_, num_anchors_, cfg, lb);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
