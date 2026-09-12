// Shared backend construction - used by both the CLI (main.cpp) and the GUI so the
// two never drift. Header-only; guarded by the same USE_* defines as CMake sets.
#pragma once
#include <memory>
#include <string>
#include <system_error>
#include <filesystem>
#include "yolomaster.hpp"

#ifdef USE_ORT
#include "ort_backend.hpp"
#endif
#ifdef USE_NCNN
#include "ncnn_backend.hpp"
#endif
#ifdef USE_MNN
#include "mnn_backend.hpp"
#endif
#ifdef USE_TRT
#include "trt_backend.hpp"
#endif

namespace yolomaster {

// Infer backend name from a model path ("" if undecidable).
inline std::string detect_backend(const std::string& model) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto ends = [&](const char* s) {
        const std::string x = s; return model.size() >= x.size() &&
            model.compare(model.size() - x.size(), x.size(), x) == 0;
    };
    if (fs::is_directory(model, ec) || ends(".param")) return "ncnn";
    if (ends(".onnx")) return "onnx";
    if (ends(".mnn"))  return "mnn";
    if (ends(".engine") || ends(".trt")) return "trt";
    return "";
}

// Resolve an ncnn model argument (dir or bare .param) plus a precision to its param/bin pair.
// Precision::Int8 selects the "<name>-int8_ncnn" sibling (meta::ncnn_int8_sibling). A missing
// sibling is a HARD error: an "int8" number must never silently come from a float model.
inline bool resolve_ncnn_paths(const std::string& model, Precision precision,
                               std::string& param, std::string& bin, std::string& err) {
    namespace fs = std::filesystem;
    const std::string m = (precision == Precision::Int8) ? meta::ncnn_int8_sibling(model) : model;
    std::error_code ec;
    if (fs::is_directory(m, ec)) {
        param = (fs::path(m) / "model.ncnn.param").string();
        bin   = (fs::path(m) / "model.ncnn.bin").string();
    } else {
        param = m;
        bin = m.substr(0, m.rfind('.')) + ".bin";
    }
    if (!fs::exists(param, ec)) {
        err = std::string(precision == Precision::Int8 ? "int8 model not found: " : "ncnn model not found: ") + param;
        return false;
    }
    return true;
}

// Construct a backend. On failure returns nullptr and fills `err`. `backend` may be
// "auto" (detected from the path). `device` is onnx-only ("cpu|cuda|coreml|trt").
// `precision` is honoured by the ncnn backend only (see Precision in yolomaster.hpp).
// `precision` is honoured by ncnn (int8 sibling) and TensorRT (.onnx -> fp16/fp32 engine build,
// cached under `trt_cache_dir` or next to the .onnx).
inline std::unique_ptr<Backend> make_backend(std::string model, std::string backend,
                                             int threads, const std::string& device,
                                             std::string& resolved, std::string& err,
                                             Precision precision = Precision::Auto,
                                             const std::string& trt_cache_dir = "") {
    namespace fs = std::filesystem;
    if (backend == "auto") {
        backend = detect_backend(model);
        if (backend.empty()) { err = "cannot infer backend from '" + model + "'"; return nullptr; }
    }
    resolved = backend;
    // GPU maps to each backend's native accelerator: onnx->CUDA EP, ncnn->Vulkan, mnn->OpenCL.
    const bool want_gpu = (device == "gpu" || device == "cuda" || device == "vulkan" || device == "opencl");
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            std::string ep = want_gpu ? "cuda" : (device.empty() ? "cpu" : device);
            return std::make_unique<OrtBackend>(model, threads, ep);
#else
            err = "built without ONNXRuntime backend"; return nullptr;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            std::string param, bin;
            if (!resolve_ncnn_paths(model, precision, param, bin, err)) return nullptr;
            return std::make_unique<NcnnBackend>(param, bin, threads, want_gpu, precision);   // want_gpu = Vulkan
#else
            err = "built without ncnn backend"; return nullptr;
#endif
        } else if (backend == "mnn") {
#ifdef USE_MNN
            std::string fwd = "cpu";
            if (device == "vulkan") fwd = "vulkan";
            else if (device == "cuda") fwd = "cuda";
            else if (want_gpu) fwd = "opencl";
            return std::make_unique<MnnBackend>(model, threads, fwd, precision);
#else
            err = "built without MNN backend (rebuild with -DUSE_MNN=ON)"; return nullptr;
#endif
        } else if (backend == "trt") {
#ifdef USE_TRT
            TrtOptions topt;
            topt.fp16 = (precision == Precision::Fp16);
            topt.cache_dir = trt_cache_dir;
            return std::make_unique<TrtBackend>(model, topt);
#else
            err = "built without TensorRT backend"; return nullptr;
#endif
        }
        err = "unknown backend: " + backend; return nullptr;
    } catch (const std::exception& e) {
        err = std::string("backend init failed: ") + e.what(); return nullptr;
    }
}

} // namespace yolomaster
