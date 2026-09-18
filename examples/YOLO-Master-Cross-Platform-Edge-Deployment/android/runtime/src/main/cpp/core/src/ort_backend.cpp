#include "ort_backend.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// ORT takes the model path as wchar_t* on Windows, char* elsewhere (ORTCHAR_T).
#ifdef _WIN32
static std::wstring ort_path(const std::string& s) { return std::wstring(s.begin(), s.end()); }
#else
static const std::string& ort_path(const std::string& s) { return s; }
#endif

// The QNN EP exists only in the Android arm64 build of ORT (the onnxruntime-android-qnn AAR);
// requesting it anywhere else is answered with a note, never a crash.
#if defined(__ANDROID__) && defined(__aarch64__)
#define YM_ORT_HAS_QNN 1
#else
#define YM_ORT_HAS_QNN 0
#endif

namespace {

// ORT keeps ONE logging manager per process (the first Ort::Env's sink wins for every later Env
// while that one is alive), so placement capture cannot rely on the sink's `param`. Session
// construction is serialized here and the backend under construction is the capture target.
std::mutex g_init_mu;
OrtBackend* g_capturing = nullptr;
std::mutex g_capture_mu;   // guards g_capturing against a sink call racing a ctor exit

bool file_exists(const std::string& p) { return !p.empty() && std::ifstream(p).good(); }

std::string lower(std::string s) {
    for (char& c : s) c = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    return s;
}

// First integer after `key` in `msg` (-1 when absent).
int int_after(const std::string& msg, const char* key) {
    const size_t p = msg.find(key);
    if (p == std::string::npos) return -1;
    size_t q = p + std::strlen(key);
    while (q < msg.size() && (msg[q] == ' ' || msg[q] == ':')) ++q;
    if (q >= msg.size() || msg[q] < '0' || msg[q] > '9') return -1;
    return std::atoi(msg.c_str() + q);
}

}  // namespace

OrtBackend::OrtBackend(const std::string& model_path, int threads, const std::string& device)
    : OrtBackend(model_path, [&] { OrtOptions o; o.threads = threads; o.device = device; return o; }()) {}

OrtBackend::OrtBackend(const std::string& model_path, const OrtOptions& opt)
    : log_tag_(opt.log_tag),
      // INFO so the partitioner's placement summary reaches the sink at init. Per-Run logging is
      // dialed back to WARNING through RunOptions (see forward_raw), so the steady state is quiet.
      env_(ORT_LOGGING_LEVEL_INFO, opt.log_tag.c_str(), &OrtBackend::log_sink, this),
      mem_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
    std::lock_guard<std::mutex> init_lock(g_init_mu);
    {
        std::lock_guard<std::mutex> lk(g_capture_mu);
        g_capturing = this;
        capturing_ = true;
    }
    try {
        build_session(model_path, opt);
    } catch (...) {
        std::lock_guard<std::mutex> lk(g_capture_mu);
        g_capturing = nullptr;
        capturing_ = false;
        throw;
    }
    std::lock_guard<std::mutex> lk(g_capture_mu);
    g_capturing = nullptr;
    capturing_ = false;
}

OrtBackend::~OrtBackend() {
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (g_capturing == this) g_capturing = nullptr;
}

void OrtBackend::log(int prio, const std::string& msg) const {
#ifdef __ANDROID__
    __android_log_print(prio, log_tag_.c_str(), "%s", msg.c_str());
#else
    (void)prio;
    std::cerr << "[" << log_tag_ << "] " << msg << "\n";
#endif
}

void ORT_API_CALL OrtBackend::log_sink(void* /*param: not trusted, see g_capturing*/, OrtLoggingLevel severity,
                                       const char* /*category*/, const char* /*logid*/,
                                       const char* /*code_location*/, const char* message) {
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (g_capturing) g_capturing->on_log(severity, message ? message : "");
#ifdef __ANDROID__
    else if (severity >= ORT_LOGGING_LEVEL_WARNING)
        __android_log_print(severity >= ORT_LOGGING_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN, "YMOrt",
                            "%s", message ? message : "");
#else
    else if (severity >= ORT_LOGGING_LEVEL_WARNING) std::cerr << "[ort] " << (message ? message : "") << "\n";
#endif
}

// Placement lines ORT emits while the session is built (onnxruntime/core/framework/session_state.cc
// and the QNN EP's GetCapability):
//   INFO    "All nodes placed on [QNNExecutionProvider]. Number of nodes: 312"
//   INFO    "Number of partitions supported by QNN EP: 1, number of nodes in the graph: 312,
//            number of nodes supported by QNN: 312"
//   VERBOSE " [CPUExecutionProvider]. Number of nodes: 4" (one per EP, under "Node placements")
// The per-EP VERBOSE lines are the most precise (post-partition) and win when present; the QNN
// summary is the INFO-level fallback; "All nodes placed" settles the single-EP case.
void OrtBackend::on_log(OrtLoggingLevel severity, const char* message) {
    const std::string msg = message;
    if (severity >= ORT_LOGGING_LEVEL_ERROR) last_ort_error_ = msg;
    if (!capturing_) return;
    // Forward the init-time diagnostics (INFO and up): they are the M0 evidence (backend
    // libraries found, HTP arch, context binary, placement) and stop after init.
#ifdef __ANDROID__
    const int prio = severity >= ORT_LOGGING_LEVEL_ERROR ? ANDROID_LOG_ERROR
                   : severity >= ORT_LOGGING_LEVEL_WARNING ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
    if (severity >= ORT_LOGGING_LEVEL_INFO) __android_log_print(prio, log_tag_.c_str(), "%s", msg.c_str());
#else
    if (severity >= ORT_LOGGING_LEVEL_WARNING) std::cerr << "[ort] " << msg << "\n";
#endif
    if (msg.rfind("All nodes placed on [", 0) == 0) {
        const int n = int_after(msg, "Number of nodes");
        if (n >= 0) {
            nodes_total_ = n;
            nodes_on_cpu_ = (msg.find("[CPUExecutionProvider]") != std::string::npos) ? n : 0;
        }
        return;
    }
    if (msg.rfind("Number of partitions supported by QNN EP", 0) == 0) {
        const int total = int_after(msg, "number of nodes in the graph");
        const int on_qnn = int_after(msg, "number of nodes supported by QNN");
        if (total >= 0 && on_qnn >= 0 && ep_nodes_ < 0 && cpu_nodes_ < 0) {
            nodes_total_ = total;
            nodes_on_cpu_ = total - on_qnn;
        }
        return;
    }
    // " [XExecutionProvider]. Number of nodes: k" (VERBOSE per-EP breakdown)
    if (msg.size() > 2 && msg[0] == ' ' && msg[1] == '[' && msg.find("]. Number of nodes") != std::string::npos) {
        const int n = int_after(msg, "Number of nodes");
        if (n < 0) return;
        if (msg.find("[CPUExecutionProvider]") != std::string::npos) cpu_nodes_ = n;
        else ep_nodes_ = (ep_nodes_ < 0 ? 0 : ep_nodes_) + n;
        nodes_on_cpu_ = cpu_nodes_ < 0 ? 0 : cpu_nodes_;
        nodes_total_ = nodes_on_cpu_ + (ep_nodes_ < 0 ? 0 : ep_nodes_);
    }
}

void OrtBackend::build_session(const std::string& model_path, const OrtOptions& opt) {
    const std::string device = lower(opt.device);
    auto note = [this](const std::string& s) { if (!ep_note.empty()) ep_note += "; "; ep_note += s; };

    // "a16w8" / "a8w8" from the file name (scripts/quantize_onnx_qnn.py names the siblings
    // model-a16w8.onnx / model-a8w8.onnx); the exporter's `ym_quant` metadata overrides below.
    {
        const std::string lp = lower(model_path);
        if (lp.find("a16w8") != std::string::npos) quant_ = "a16w8";
        else if (lp.find("a8w8") != std::string::npos) quant_ = "a8w8";
    }

    std::string path_to_open = model_path;
    bool want_qnn = false, qnn_ok = false;

    if (device == "qnn" || device == "npu" || device == "htp") {
        want_qnn = true;
#if YM_ORT_HAS_QNN
        try {
            // fp32 graphs run on the HTP as fp16 (enable_htp_fp16_precision); QDQ graphs keep
            // float I/O because the CPU EP takes the input quantize / output dequantize
            // (offload_graph_io_quantization), so forward_raw never has to build u8/u16 tensors.
            // htp_arch stays at auto: the driver picks the skel it finds (V81 on the S26).
            std::unordered_map<std::string, std::string> qo = {
                {"backend_type", "htp"},
                {"htp_performance_mode", opt.htp_perf.empty() ? "burst" : opt.htp_perf},
                {"htp_graph_finalization_optimization_mode", "3"},
                {"enable_htp_fp16_precision", "1"},
                // Strict sessions forbid any CPU node, and ORT rejects offloading the I/O Q/DQ to
                // another EP in that mode (it logs a conflict); QNN places those nodes itself.
                {"offload_graph_io_quantization", opt.strict_htp ? "0" : "1"},
                {"qnn_context_priority", "high"},
            };
            opts_.AppendExecutionProvider("QNN", qo);
            qnn_ok = true;
        } catch (const std::exception& e) {
            if (opt.strict_htp) throw std::runtime_error(std::string("QNN EP unavailable: ") + e.what());
            note(std::string("QNN EP unavailable: ") + e.what());
            log(6, ep_note);
        }
#else
        if (opt.strict_htp) throw std::runtime_error("QNN EP is Android-arm64 only in this build");
        note("QNN EP is Android-arm64 only; using the CPU EP");
#endif
        if (qnn_ok) {
            // Extended/layout fusions can synthesize ops the HTP refuses (NhwcConv, FusedConv):
            // BASIC keeps the graph in the plain ONNX vocabulary the QNN builders cover.
            opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
            // The CPU EP only gets what the HTP declined; two threads, no spinning (the Live
            // loop shares the two prime cores with the camera and the UI).
            opts_.SetIntraOpNumThreads(std::max(1, std::min(opt.threads, 2)));
            opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");
            if (opt.strict_htp) opts_.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            if (!opt.ctx_cache_path.empty()) {
                if (file_exists(opt.ctx_cache_path)) {
                    // Pre-compiled HTP context: open it instead of the ONNX (no graph finalization).
                    path_to_open = opt.ctx_cache_path;
                    log(4, "opening EPContext cache " + opt.ctx_cache_path);
                } else {
                    opts_.AddConfigEntry("ep.context_enable", "1");
                    opts_.AddConfigEntry("ep.context_file_path", opt.ctx_cache_path.c_str());
                    opts_.AddConfigEntry("ep.context_embed_mode", "1");
                    log(4, "generating EPContext cache " + opt.ctx_cache_path);
                }
            }
        }
    }
#ifndef __ANDROID__
    else if (device == "trt" || device == "tensorrt") {
        // ONNXRuntime TensorRT EP: builds+caches a TRT engine internally (near-native TRT),
        // honors QDQ nodes for INT8 + FP16 elsewhere, and auto-falls-back to CUDA/CPU for
        // unsupported subgraphs. Portable: ship the .onnx; the engine cache builds on first run.
        try {
            OrtTensorRTProviderOptionsV2* trt = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt));
            const char* keys[] = {"trt_fp16_enable", "trt_int8_enable",
                                  "trt_engine_cache_enable", "trt_engine_cache_path"};
            const char* vals[] = {"1", "1", "1", "trt_engine_cache"};
            Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(trt, keys, vals, 4));
            opts_.AppendExecutionProvider_TensorRT_V2(*trt);
            Ort::GetApi().ReleaseTensorRTProviderOptions(trt);
            active_ep = "ort-TensorRT";
        } catch (const std::exception& e) {
            std::cerr << "[ort] TensorRT EP unavailable (" << e.what() << "); trying CUDA\n";
        }
        // CUDA fallback for TRT-unsupported nodes (and if the TRT EP failed to load)
        try {
            OrtCUDAProviderOptions cuda{}; cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);
            if (active_ep != "ort-TensorRT") active_ep = "ort-CUDA";
        } catch (const std::exception& e) {
            if (active_ep != "ort-TensorRT") { std::cerr << "[ort] CUDA EP unavailable; using CPU\n"; active_ep.clear(); }
        }
    } else if (device == "cuda" || device == "gpu") {
        try {                                    // graceful fallback if CUDA EP can't load
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);
            active_ep = "ort-CUDA";
        } catch (const std::exception& e) {
            std::cerr << "[ort] CUDA EP unavailable (" << e.what() << "); using CPU\n";
            note(std::string("CUDA EP failed: ") + e.what());
        }
    } else if (device == "coreml") {
        // Apple CoreML EP (macOS): ORT partitions the graph, runs supported subgraphs on ANE/GPU
        // and the rest on CPU. MLComputeUnits=CPUAndGPU because the GPU tolerates the graph
        // fragmentation of this MoE+attention model far better than the ANE. Falls back to CPU.
        try {
            std::unordered_map<std::string, std::string> co = {
                {"MLComputeUnits", "CPUAndGPU"},
                {"ModelFormat", "MLProgram"},
                {"RequireStaticInputShapes", "1"},
            };
            opts_.AppendExecutionProvider("CoreML", co);
            active_ep = "ort-CoreML";
        } catch (const std::exception& e) {
            std::cerr << "[ort] CoreML EP unavailable (" << e.what() << "); using CPU\n";
        }
    }
#endif
    if (!qnn_ok) {
        // CPU EP (alone or behind CUDA/TRT/CoreML): the historical desktop settings.
        opts_.SetIntraOpNumThreads(std::max(1, opt.threads));
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef __ANDROID__
        opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");   // phones: never burn a core waiting
#endif
    }
    opts_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);   // the session logger carries the placement lines

    auto open = [&](const std::string& p) {
        session_ = std::make_unique<Ort::Session>(env_, ort_path(p).c_str(), opts_);
    };
    try {
        open(path_to_open);
    } catch (const Ort::Exception& e) {
        std::string why = e.what();
        if (!last_ort_error_.empty() && why.find(last_ort_error_) == std::string::npos) why += " | " + last_ort_error_;
        if (qnn_ok && !opt.strict_htp) {
            // The HTP could not take the graph (driver, skel, context binary): fall back to a
            // plain CPU session so the caller still has a working model, and say so.
            log(6, "QNN session failed (" + why + "); retrying on the CPU EP");
            note("QNN session failed: " + why);
            opts_ = Ort::SessionOptions();
            opts_.SetIntraOpNumThreads(std::max(1, opt.threads));
            opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            opts_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
            nodes_total_ = nodes_on_cpu_ = 0; ep_nodes_ = cpu_nodes_ = -1;
            qnn_ok = false;
            open(model_path);
        } else {
            throw std::runtime_error(why);
        }
    }

    read_model_info();

    // ---- name the result honestly: what runs where, not what was asked for ----
    const std::string prec = quant_.empty() ? std::string("fp32") : quant_;
    if (qnn_ok) {
        // The QNN EP answers GetCapability with nothing when its backend failed to initialise
        // (missing skel, no DSP, rejected context): the whole graph then silently lands on the
        // CPU EP. That is a CPU run and is labelled as one.
        const bool all_cpu = nodes_total_ > 0 && nodes_on_cpu_ >= nodes_total_;
        if (all_cpu) {
            active_ep = "ort-CPU-" + prec;
            note("QNN requested but the HTP took 0/" + std::to_string(nodes_total_) + " nodes");
        } else {
            active_ep = "ort-QNN-htp-" + (quant_.empty() ? std::string("fp16") : quant_);
            if (nodes_on_cpu_ > 0) {
                active_ep += "-mixed";
                note("partial HTP: " + std::to_string(nodes_on_cpu_) + "/" + std::to_string(nodes_total_) + " nodes on CPU");
            }
        }
    } else if (active_ep.empty() || active_ep == "cpu") {
        active_ep = "ort-CPU-" + prec;
    }
    if (want_qnn && !qnn_ok && ep_note.empty()) note("QNN not used");
    log(4, "session ready: " + active_ep + " placement=" + std::to_string(nodes_total_ - nodes_on_cpu_) + "/" +
               std::to_string(nodes_total_) + " on the EP" + (ep_note.empty() ? "" : " note=" + ep_note));
}

void OrtBackend::read_model_info() {
    const size_t n_in = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    for (size_t i = 0; i < n_in; ++i)
        in_names_s_.push_back(session_->GetInputNameAllocated(i, alloc_).get());
    for (size_t i = 0; i < n_out; ++i)
        out_names_s_.push_back(session_->GetOutputNameAllocated(i, alloc_).get());
    for (auto& s : in_names_s_) in_names_.push_back(s.c_str());
    for (auto& s : out_names_s_) out_names_.push_back(s.c_str());

    // Input dtype + static size (H==W>0 -> hard constraint).
    {
        // GetTensorTypeAndShapeInfo() is a non-owning view: the TypeInfo must outlive it.
        Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
        auto info = ti.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType t = info.GetElementType();
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) in_fp16_ = true;
        else if (t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("ONNX input dtype " + std::to_string(static_cast<int>(t)) +
                                     " is not float/float16 (quantized I/O must be offloaded: offload_graph_io_quantization)");
        auto shape = info.GetShape();
        if (shape.size() == 4 && shape[2] > 0 && shape[2] == shape[3]) {
            fixed_imgsz = static_cast<int>(shape[2]);
            meta_imgsz = fixed_imgsz;   // authoritative over the metadata string
        }
    }

    // auto-read ultralytics-embedded metadata (class names + imgsz + our export stamps)
    Ort::ModelMetadata md = session_->GetModelMetadata();
    if (auto v = md.LookupCustomMetadataMapAllocated("names", alloc_))
        meta_names = meta::parse_names_dict(v.get());
    if (auto v = md.LookupCustomMetadataMapAllocated("imgsz", alloc_)) {
        const std::string s = v.get();
        const size_t p = s.find_first_of("0123456789");
        if (p != std::string::npos && fixed_imgsz == 0) meta_imgsz = std::atoi(s.c_str() + p);
    }
    if (auto v = md.LookupCustomMetadataMapAllocated("end2end", alloc_)) {
        const std::string s = v.get();
        end2end_ = s.find("rue") != std::string::npos || s == "1";
        if (end2end_) log(4, "end2end model: NMS-free [num_det,6] output");
    }
    if (auto v = md.LookupCustomMetadataMapAllocated("task", alloc_)) task_ = v.get();
    if (auto v = md.LookupCustomMetadataMapAllocated("ym_quant", alloc_)) {
        const std::string s = lower(v.get());
        if (s == "a16w8" || s == "a8w8") quant_ = s;
        else if (s.empty()) quant_.clear();
    }
}

void OrtBackend::forward_raw(const cv::Mat& bgr, const Config& cfg, bool decode) {
    // ---- preprocess: letterbox -> NCHW float RGB /255 (the ncnn path's from_pixels + normalize) ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, cfg.imgsz, cfg.stretch, lb);   // imgsz x imgsz, CV_8UC3 BGR
    const int sz = cfg.imgsz, hw = sz * sz;
    blob_.resize(static_cast<size_t>(3) * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;          // BGR
            const int idx = y * sz + x;
            blob_[idx]          = px[2] * (1.0f / 255); // R
            blob_[hw + idx]     = px[1] * (1.0f / 255); // G
            blob_[2 * hw + idx] = px[0] * (1.0f / 255); // B
        }
    }
    std::array<int64_t, 4> in_shape{1, 3, cfg.imgsz, cfg.imgsz};
    Ort::Value in_tensor{nullptr};
    if (in_fp16_) {
        blob16_.resize(blob_.size());
        for (size_t i = 0; i < blob_.size(); ++i) blob16_[i] = Ort::Float16_t(blob_[i]);
        in_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(mem_, blob16_.data(), blob16_.size(),
                                                              in_shape.data(), in_shape.size());
    } else {
        in_tensor = Ort::Value::CreateTensor<float>(mem_, blob_.data(), blob_.size(),
                                                    in_shape.data(), in_shape.size());
    }
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    Ort::RunOptions ro;
    ro.SetRunLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);   // the session is at INFO for init; runs stay quiet
    auto outs = session_->Run(ro, in_names_.data(), &in_tensor, 1, out_names_.data(), out_names_.size());
    infer_ms = ms_since(t1);

    // Every path below (re)fills the cached raw state; a bench-only forward leaves it empty so
    // stale candidates can never be mistaken for this frame's.
    candidates.clear();
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    if (!decode) { post_ms = 0; return; }

    // ---- postprocess: detection is the rank-3 output [1,feat,anchors]; proto (seg) is rank-4 ----
    auto t2 = clk::now();
    int det_i = -1, proto_i = -1;
    for (size_t i = 0; i < outs.size(); ++i) {
        const size_t r = outs[i].GetTensorTypeAndShapeInfo().GetShape().size();
        if (r == 4) proto_i = static_cast<int>(i);
        else if (r == 3) det_i = static_cast<int>(i);
    }
    if (det_i < 0) throw std::runtime_error("ONNX model has no rank-3 detection output");
    // fp16 outputs (fp16-I/O exports) are widened once; fp32 outputs are read in place.
    auto as_float = [this](Ort::Value& v, size_t count) -> const float* {
        const ONNXTensorElementDataType t = v.GetTensorTypeAndShapeInfo().GetElementType();
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return v.GetTensorData<float>();
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const Ort::Float16_t* h = v.GetTensorData<Ort::Float16_t>();
            out_f32_.resize(count);
            for (size_t i = 0; i < count; ++i) out_f32_[i] = h[i].ToFloat();
            return out_f32_.data();
        }
        throw std::runtime_error("ONNX output dtype " + std::to_string(static_cast<int>(t)) + " is not float/float16");
    };
    auto shape = outs[det_i].GetTensorTypeAndShapeInfo().GetShape();   // {1, d1, d2}
    const int d1 = static_cast<int>(shape[1]);
    const int d2 = static_cast<int>(shape[2]);
    const float* out = as_float(outs[det_i], static_cast<size_t>(d1) * d2);
    if (end2end_ || looks_end2end(d1, d2)) {                              // [1, num_det, 6] NMS-free
        candidates = decode_end2end(out, d1, cfg, lb);
    } else {
        // feat << anchors always (e.g. 84/116 vs 8400), so the smaller axis is the feature dim;
        // an anchors-major export is transposed into the channel-major layout decode expects.
        int feat_dim, num_anchors;
        std::vector<float> buf;
        const float* cm = out;
        if (d1 <= d2) { feat_dim = d1; num_anchors = d2; }
        else {
            feat_dim = d2; num_anchors = d1;
            buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
            for (int a = 0; a < num_anchors; ++a)
                for (int f = 0; f < feat_dim; ++f)
                    buf[static_cast<size_t>(f) * num_anchors + a] = out[static_cast<size_t>(a) * feat_dim + f];
            cm = buf.data();
        }
        candidates = decode_candidates(cm, feat_dim, num_anchors, cfg, lb);
    }
    if (proto_i >= 0) {                                                // segmentation model
        auto ps = outs[proto_i].GetTensorTypeAndShapeInfo().GetShape();  // {1, nm, mh, mw}
        proto_c = (int)ps[1]; proto_h = (int)ps[2]; proto_w = (int)ps[3];
        const size_t n = (size_t)proto_c * proto_h * proto_w;
        const float* pd = as_float(outs[proto_i], n);
        proto.assign(pd, pd + n);
    }
    post_ms = ms_since(t2);
}

} // namespace yolomaster
