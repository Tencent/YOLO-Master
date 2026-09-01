// TensorRT backend for YOLO-Master - GPU inference from a prebuilt .engine.
// Loads an engine built on-device by trtexec (jetson/10_trt_bench.sh) and runs it on CUDA.
// Detection engines have one static output [1,feat,anchors] (the transposed
// [1,anchors,feat] form is also accepted); segmentation engines add one proto
// output [1,nm,mh,mw]. Input/output tensors may be FP32 or FP16. Auxiliary or
// dynamic-shape tensors are rejected with an actionable error because a native
// engine must be built for the target input size. Class names / imgsz come from an
// optional metadata.yaml sidecar (engines embed no metadata):
// <engine-minus-ext>.metadata.yaml, or metadata.yaml next to the engine.
#pragma once
#include "yolomaster.hpp"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yolomaster {

// The native runner intentionally targets TensorRT 10's named-I/O API
// (getNbIOTensors/setTensorAddress/enqueueV3).  TensorRT 8 exposes a different
// binding API and is rejected by the CMake configure step with an actionable
// message; use the ONNX Runtime TensorRT execution provider when a target is
// pinned to a TensorRT 8/JetPack release.
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
#error "YOLO-Master native TensorRT backend requires TensorRT 10.x; use ORT + TensorRT EP for TensorRT 8.x"
#else
template <typename T>
struct TrtDeleter {
    void operator()(T* value) const noexcept { delete value; }
};
#endif

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

class TrtBackend : public Backend {
public:
    explicit TrtBackend(const std::string& engine_path);
    ~TrtBackend() override;
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> ctx_;
    cudaStream_t stream_ = nullptr;
    void* d_in_    = nullptr;
    void* d_out_   = nullptr;
    void* d_proto_ = nullptr;              // seg engines only
    std::string in_name_, out_name_, proto_name_;
    int in_sz_ = 0;                        // input H (== W)
    int out_dim0_ = 0, out_dim1_ = 0;      // detection output axes as exported
    int feat_dim_ = 0, num_anchors_ = 0;   // normalized [features, anchors]
    int pc_ = 0, ph_ = 0, pw_ = 0;         // proto output [1, pc, ph, pw] (0 = detection engine)
    bool input_fp16_ = false;
    bool output_fp16_ = false;
    bool proto_fp16_ = false;
    bool has_objectness_ = false;
    int mask_channels_ = 0;
    std::vector<float> h_out_, h_proto_;
    std::vector<uint16_t> h_out16_, h_proto16_;
};

} // namespace yolomaster
