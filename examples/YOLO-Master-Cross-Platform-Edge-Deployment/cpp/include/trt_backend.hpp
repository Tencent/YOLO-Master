// TensorRT backend for YOLO-Master - GPU inference from a prebuilt .engine.
// Loads an engine built on-device by trtexec (jetson/10_trt_bench.sh) and runs it on CUDA.
// Detection engines have one output [1,feat,anchors]; segmentation engines add a proto
// output [1,nm,mh,mw] (both discovered by rank). Class names / imgsz come from an
// optional metadata.yaml sidecar (engines embed no metadata):
// <engine-minus-ext>.metadata.yaml, or metadata.yaml next to the engine.
#pragma once
#include "yolomaster.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace yolomaster {

// Engine-build options used when the model path is an .onnx (needs nvonnxparser at build
// time, USE_TRT_ONNXPARSER). The engine is cached under cache_dir keyed by the ONNX content,
// the GPU name, the TensorRT version and the precision, so a warm start deserializes only.
struct TrtOptions {
    bool fp16 = false;               // builder FP16 flag (fp32 otherwise; INT8 engines are prebuilt .engine)
    std::string cache_dir;           // "" = next to the .onnx
    int workspace_mb = 2048;
    bool verbose = false;
    // Preprocess on the GPU (raw uint8 H2D + one kernel into the input tensor) when the runtime was
    // built with USE_CUDA_PREPROC; false = the CPU preprocess_nchw path (parity runs, old timing tables).
    bool gpu_preproc = true;
    // Capture the per-frame stream work (params copy, kernel, enqueueV3, D2H) into a CUDA graph after
    // the first forward and replay it. Falls back to the plain path if capture fails (ep_note says so).
    bool cuda_graph = false;
};

class TrtBackend : public Backend {
public:
    explicit TrtBackend(const std::string& engine_path);
    // .engine -> load; .onnx -> build (or reuse cached) engine per `opt`, then load.
    TrtBackend(const std::string& model_path, const TrtOptions& opt);
    // Build an engine file from an ONNX model. Returns the engine path; throws on failure.
    static std::string build_engine(const std::string& onnx_path, const std::string& engine_path,
                                    const TrtOptions& opt);
    // Cache path for (onnx, opt) on this GPU: <cache_dir>/<stem>-<sha1[:12]>-<gpu>-trt<ver>-<fp16|fp32>.engine
    static std::string cached_engine_path(const std::string& onnx_path, const TrtOptions& opt);
    std::string engine_path;         // engine actually loaded
    ~TrtBackend() override;
    // The forward_raw contract (decode=false = bench-only forward, candidates cleared); infer() is
    // the base class's forward_raw + nms_and_cap.
    void forward_raw(const cv::Mat& bgr, const Config& cfg, bool decode = true) override;
    const char* runtime_name() const override { return "trt"; }
    std::string device_name() const override;
    bool gpu_preproc() const { return gpu_preproc_; }
    bool cuda_graph_active() const { return graph_ok_; }

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    cudaStream_t stream_ = nullptr;
    void* d_in_    = nullptr;
    void* d_out_   = nullptr;
    void* d_proto_ = nullptr;              // seg engines only
    std::string in_name_, out_name_, proto_name_;
    int in_sz_ = 0;                        // input H (== W)
    int feat_dim_ = 0, num_anchors_ = 0;   // detection output [1, feat_dim, num_anchors]
    bool end2end_ = false;                 // NMS-free [1, num_det, 6] head (sidecar or shape)
    int pc_ = 0, ph_ = 0, pw_ = 0;         // proto output [1, pc, ph, pw] (0 = detection engine)
    float* h_in_ = nullptr; float* h_out_ = nullptr; float* h_proto_ = nullptr;   // pinned host staging
    size_t in_count_ = 0, out_count_ = 0, proto_count_ = 0;
    // GPU preprocessing (HAVE_CUDA_PREPROC): raw frame + geometry on the device, timing events
    bool gpu_preproc_ = false, want_gpu_preproc_ = true;
    uint8_t* h_raw_ = nullptr; uint8_t* d_raw_ = nullptr; size_t raw_cap_ = 0;
    void* d_params_ = nullptr; void* h_params_ = nullptr;                          // PreprocParams
    cudaEvent_t ev1_ = nullptr, ev2_ = nullptr;
    // CUDA graph replay of the fixed per-frame stream work
    bool want_graph_ = false, graph_ok_ = false;
    cudaGraph_t graph_ = nullptr; cudaGraphExec_t graph_exec_ = nullptr;
    long frames_ = 0;
    void load_engine(const std::string& engine_path);
    void ensure_raw_capacity(size_t bytes);
    void enqueue_frame(bool record_events);   // the captured / replayed sequence (everything after the raw H2D)
    bool try_capture_graph();
};

} // namespace yolomaster
