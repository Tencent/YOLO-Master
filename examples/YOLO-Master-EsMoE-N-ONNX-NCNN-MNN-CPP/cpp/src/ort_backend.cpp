#include "ort_backend.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>

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

OrtBackend::OrtBackend(const std::string& model_path, int threads, const std::string& device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolomaster") {
    opts_.SetIntraOpNumThreads(threads);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device == "cuda") {
        try {                                    // graceful fallback if CUDA EP can't load
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);
            active_ep = "CUDA";
        } catch (const std::exception& e) {
            std::cerr << "[ort] CUDA EP unavailable (" << e.what() << "); using CPU\n";
            active_ep = "CPU";
        }
    }
    session_ = std::make_unique<Ort::Session>(env_, ort_path(model_path).c_str(), opts_);

    const size_t n_in = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    for (size_t i = 0; i < n_in; ++i)
        in_names_s_.push_back(session_->GetInputNameAllocated(i, alloc_).get());
    for (size_t i = 0; i < n_out; ++i)
        out_names_s_.push_back(session_->GetOutputNameAllocated(i, alloc_).get());
    for (auto& s : in_names_s_) in_names_.push_back(s.c_str());
    for (auto& s : out_names_s_) out_names_.push_back(s.c_str());

    // detect a static input size (H==W>0) -> hard constraint
    {
        auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4 && shape[2] > 0 && shape[2] == shape[3]) {
            fixed_imgsz = static_cast<int>(shape[2]);
            meta_imgsz = fixed_imgsz;   // authoritative over the metadata string
        }
    }

    // auto-read ultralytics-embedded metadata (class names + imgsz)
    Ort::ModelMetadata md = session_->GetModelMetadata();
    if (auto v = md.LookupCustomMetadataMapAllocated("names", alloc_))
        meta_names = meta::parse_names_dict(v.get());
    if (auto v = md.LookupCustomMetadataMapAllocated("imgsz", alloc_)) {
        const std::string s = v.get();
        const size_t p = s.find_first_of("0123456789");
        if (p != std::string::npos) meta_imgsz = std::atoi(s.c_str() + p);
    }
}

std::vector<Detection> OrtBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> NCHW float RGB /255 ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = letterbox(bgr, cfg.imgsz, lb);   // imgsz x imgsz, CV_8UC3 BGR
    // NCHW float RGB /255 (replaces cv::dnn::blobFromImage with swapRB=true)
    const int sz = cfg.imgsz, hw = sz * sz;
    std::vector<float> blob(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;          // BGR
            const int idx = y * sz + x;
            blob[idx]          = px[2] * (1.0f / 255); // R
            blob[hw + idx]     = px[1] * (1.0f / 255); // G
            blob[2 * hw + idx] = px[0] * (1.0f / 255); // B
        }
    }
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    std::array<int64_t, 4> in_shape{1, 3, cfg.imgsz, cfg.imgsz};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(),
        in_shape.data(), in_shape.size());
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names_.data(), &in_tensor, 1,
                              out_names_.data(), out_names_.size());
    infer_ms = ms_since(t1);

    // ---- postprocess: decode (1, feat_dim, num_anchors) ----
    auto t2 = clk::now();
    auto shape = outs.front().GetTensorTypeAndShapeInfo().GetShape();  // {1, 14, 8400}
    const int feat_dim = static_cast<int>(shape[1]);
    const int num_anchors = static_cast<int>(shape[2]);
    const float* out = outs.front().GetTensorMutableData<float>();
    auto dets = decode(out, feat_dim, num_anchors, cfg, lb);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
