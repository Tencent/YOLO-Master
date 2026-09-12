// ONNX Runtime backend for YOLO-Master.
//
// Desktop: CPU / CUDA / TensorRT / CoreML execution providers (the CLI, the GUI). Android arm64:
// the Qualcomm QNN execution provider on the Hexagon NPU (HTP) with the CPU EP as the fallback
// for nodes the HTP refuses - the second runtime the app offers next to ncnn. One class, one
// forward_raw: the JNI bridge and the CLI never see which EP is underneath (Backend::active_ep
// tells them).
#pragma once
#include "yolomaster.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace yolomaster {

// Construction options. Every field has the value the desktop CLI used before this struct
// existed, so `OrtBackend(path, {})` is the old `OrtBackend(path)`.
struct OrtOptions {
    // "cpu" | "qnn" (Android arm64: HTP with CPU fallback) | "cuda" | "trt" | "coreml" (desktop).
    std::string device = "cpu";
    // CPU EP intra-op threads. With the QNN EP the graph runs on the HTP and the CPU EP only
    // gets the nodes the HTP refused, so the pool is capped at 2 there (see ort_backend.cpp).
    int threads = 4;
    // QNN `htp_performance_mode`: "burst" (Live), "sustained_high_performance" (Bench), "default".
    std::string htp_perf = "burst";
    // EPContext cache ("<stem>_ctx.onnx"). Empty = none. When the file exists it is opened
    // INSTEAD of the model (skips the seconds of on-device HTP graph finalization); when it does
    // not, the session generates it at init so the next start is fast.
    std::string ctx_cache_path;
    // `session.disable_cpu_ep_fallback=1`: a single node the HTP cannot run fails init instead of
    // silently landing on the CPU. The verification switch (the strict M0 row), not a Live setting.
    bool strict_htp = false;
    // logcat tag for the forwarded ORT diagnostics (stderr on desktop).
    std::string log_tag = "YMOrt";
};

class OrtBackend : public Backend {
public:
    OrtBackend(const std::string& model_path, const OrtOptions& opt);
    // Desktop shim: device "cpu" | "cuda" | "trt" | "coreml" (the CLI's --device).
    OrtBackend(const std::string& model_path, int threads = 4, const std::string& device = "cpu");
    ~OrtBackend() override;

    void forward_raw(const cv::Mat& bgr, const Config& cfg, bool decode = true) override;
    const char* runtime_name() const override { return "ort"; }

    // Graph placement as reported by ORT's partitioner at session init (0/0 when ORT logged
    // nothing parseable, e.g. a minimal build). on_cpu counts the nodes the requested
    // accelerator EP did not take; total is the post-optimization node count.
    int nodes_total() const { return nodes_total_; }
    int nodes_on_cpu() const { return nodes_on_cpu_; }
    // "" (float model), "a16w8" or "a8w8": from the exporter's `ym_quant` metadata, else the file name.
    const std::string& quant() const { return quant_; }
    // ultralytics `task` metadata ("detect" / "segment"; "" when the export carries none).
    const std::string& task() const { return task_; }

private:
    void build_session(const std::string& model_path, const OrtOptions& opt);
    void read_model_info();
    void log(int prio, const std::string& msg) const;
    // The Ort::Env log sink. ORT keeps ONE process-wide logging manager (the first Env's), so the
    // sink never trusts its `param`: it hands every message to the backend currently inside
    // build_session() (init is serialized process-wide by g_init_mu) and drops the rest.
    static void ORT_API_CALL log_sink(void* param, OrtLoggingLevel severity, const char* category,
                                      const char* logid, const char* code_location, const char* message);
    void on_log(OrtLoggingLevel severity, const char* message);

    std::string log_tag_ = "YMOrt";
    bool capturing_ = false;          // inside build_session(): parse placement lines
    int nodes_total_ = 0, nodes_on_cpu_ = 0;
    int ep_nodes_ = -1, cpu_nodes_ = -1;   // VERBOSE per-EP "Node placements" counts (when present)
    std::string last_ort_error_;      // last ERROR-level ORT line during init (surfaced in the exception)

    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions alloc_;
    Ort::MemoryInfo mem_;
    std::vector<std::string> in_names_s_, out_names_s_;
    std::vector<const char*> in_names_, out_names_;
    bool end2end_ = false;   // NMS-free [1,num_det,6] output (yolo26 lineage)
    bool in_fp16_ = false;   // graph input dtype (fp32 models run on the HTP as fp16 keep fp32 I/O)
    std::string quant_;
    std::string task_;
    std::vector<float> blob_;               // reused NCHW RGB/255 input
    std::vector<Ort::Float16_t> blob16_;    // fp16 twin, only for fp16-I/O graphs
    std::vector<float> out_f32_;            // fp16 -> fp32 output staging
};

} // namespace yolomaster
