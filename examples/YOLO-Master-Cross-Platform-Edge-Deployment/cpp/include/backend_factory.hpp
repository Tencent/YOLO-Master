// Shared backend construction - used by both the CLI (main.cpp) and the GUI so the
// two never drift. Header-only; guarded by the same USE_* defines as CMake sets.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <system_error>
#include <filesystem>
#include <vector>
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

// Backend/model options are user-facing tokens.  Normalize ASCII case once so
// Windows files such as MODEL.ONNX and flags such as --backend ONNX behave the
// same as their lowercase spellings.
inline std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool path_extension_is(const std::filesystem::path& path, const char* extension) {
    return lower_ascii(path.extension().string()) == extension;
}

inline bool path_stem_equal_ci(const std::filesystem::path& a, const std::filesystem::path& b) {
    return lower_ascii(a.stem().string()) == lower_ascii(b.stem().string());
}

// Resolve an NCNN .param/.bin pair.  Exports in the wild use names such as
// model.ncnn.param, best.param, or arbitrary stems; relying on one literal name
// made directory-model invocation fail even when a valid pair was present.
inline bool resolve_ncnn_pair(const std::string& model, std::string& param,
                              std::string& bin, std::string& err) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path input(model);
    std::vector<fs::path> params;
    std::vector<fs::path> bins;
    auto collect = [&](const fs::path& dir) {
        for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
             it != end; it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            const fs::path p = it->path();
            if (!it->is_regular_file(ec)) { ec.clear(); continue; }
            if (path_extension_is(p, ".param")) params.push_back(p);
            else if (path_extension_is(p, ".bin")) bins.push_back(p);
        }
    };
    const bool input_is_dir = fs::is_directory(input, ec);
    ec.clear();
    const bool input_is_file = fs::is_regular_file(input, ec);
    ec.clear();
    if (input_is_dir) {
        collect(input);
    } else if (input_is_file) {
        const fs::path parent = input.parent_path().empty() ? fs::path(".") : input.parent_path();
        collect(parent);
        if (path_extension_is(input, ".param")) {
            params.erase(std::remove_if(params.begin(), params.end(), [&](const fs::path& p) {
                return lower_ascii(p.lexically_normal().string()) !=
                       lower_ascii(input.lexically_normal().string());
            }), params.end());
        } else if (path_extension_is(input, ".bin")) {
            bins.erase(std::remove_if(bins.begin(), bins.end(), [&](const fs::path& p) {
                return lower_ascii(p.lexically_normal().string()) !=
                       lower_ascii(input.lexically_normal().string());
            }), bins.end());
        }
        else {
            err = "NCNN model must be a directory, .param, or .bin file: " + model;
            return false;
        }
    } else {
        err = "NCNN model path does not exist: " + model;
        return false;
    }
    auto by_name = [](const fs::path& a, const fs::path& b) {
        const std::string al = lower_ascii(a.filename().string());
        const std::string bl = lower_ascii(b.filename().string());
        return al < bl;
    };
    std::sort(params.begin(), params.end(), by_name);
    std::sort(bins.begin(), bins.end(), by_name);

    // Keep only params with a same-stem binary (case-insensitive extension and
    // stem matching).  This prevents selecting a sidecar or an unrelated bin.
    struct Pair { fs::path param; fs::path bin; };
    std::vector<Pair> pairs;
    for (const fs::path& p : params) {
        auto match = std::find_if(bins.begin(), bins.end(), [&](const fs::path& b) {
            return path_stem_equal_ci(p, b);
        });
        if (match != bins.end()) pairs.push_back({p, *match});
    }
    if (pairs.empty()) {
        err = "NCNN model directory has no matching .param/.bin pair: " + model;
        return false;
    }

    // Prefer conventional stems.  If more than one non-conventional pair is
    // present, fail rather than silently benchmarking the wrong network.
    auto score = [](const fs::path& p) {
        const std::string stem = lower_ascii(p.stem().string());
        if (stem == "model.ncnn") return 0;
        if (stem == "model") return 1;
        if (stem == "best") return 2;
        return 3;
    };
    std::sort(pairs.begin(), pairs.end(), [&](const Pair& a, const Pair& b) {
        const int sa = score(a.param), sb = score(b.param);
        if (sa != sb) return sa < sb;
        return by_name(a.param, b.param);
    });
    if (pairs.size() > 1 && score(pairs[0].param) == 3) {
        err = "NCNN model directory contains multiple ambiguous .param/.bin pairs; "
              "pass the .param file explicitly: " + model;
        return false;
    }
    param = pairs.front().param.string();
    bin = pairs.front().bin.string();
    return true;
}

// Infer backend name from a model path ("" if undecidable).
inline std::string detect_backend(const std::string& model) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path path(model);
    const std::string ext = lower_ascii(path.extension().string());
    if (fs::is_directory(model, ec) || ext == ".param" || ext == ".bin") return "ncnn";
    if (ext == ".onnx") return "onnx";
    if (ext == ".mnn")  return "mnn";
    if (ext == ".engine" || ext == ".trt") return "trt";
    return "";
}

// Construct a backend. On failure returns nullptr and fills `err`. `backend` may be
// "auto" (detected from the path). `device` is mapped to the selected backend's
// native execution provider (for example, CUDA for ONNX Runtime or Vulkan for NCNN).
inline std::unique_ptr<Backend> make_backend(std::string model, std::string backend,
                                              int threads, const std::string& device,
                                              std::string& resolved, std::string& err) {
    backend = lower_ascii(backend);
    const std::string normalized_device = lower_ascii(device);
    if (backend == "auto") {
        backend = detect_backend(model);
        if (backend.empty()) { err = "cannot infer backend from '" + model + "'"; return nullptr; }
    }
    if (backend != "onnx" && backend != "ncnn" && backend != "mnn" && backend != "trt") {
        err = "unknown backend: " + backend;
        return nullptr;
    }
    if (normalized_device != "" && normalized_device != "cpu" &&
        normalized_device != "gpu" && normalized_device != "cuda" &&
        normalized_device != "vulkan" && normalized_device != "opencl" &&
        normalized_device != "trt" && normalized_device != "tensorrt" &&
        normalized_device != "coreml") {
        err = "unknown device: " + device;
        return nullptr;
    }
    resolved = backend;
    // GPU maps to each backend's native accelerator: onnx->CUDA EP, ncnn->Vulkan, mnn->OpenCL.
    const bool want_gpu = (normalized_device == "gpu" || normalized_device == "cuda" ||
                           normalized_device == "vulkan" || normalized_device == "opencl" ||
                           normalized_device == "trt" || normalized_device == "tensorrt");
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            if (normalized_device == "vulkan" || normalized_device == "opencl") {
                err = "ONNX Runtime supports cpu, cuda, trt, or coreml devices; got " + device;
                return nullptr;
            }
            std::string ep = (normalized_device == "gpu") ? "cuda"
                : (normalized_device.empty() ? "cpu" : normalized_device);
            return std::make_unique<OrtBackend>(model, threads, ep);
#else
            err = "built without ONNXRuntime backend"; return nullptr;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            if (normalized_device == "cuda" || normalized_device == "trt" ||
                normalized_device == "tensorrt" || normalized_device == "coreml" ||
                normalized_device == "opencl") {
                err = "NCNN supports cpu or vulkan devices; got " + device;
                return nullptr;
            }
            std::string param, bin;
            if (!resolve_ncnn_pair(model, param, bin, err)) return nullptr;
            return std::make_unique<NcnnBackend>(param, bin, threads,
                                                 normalized_device == "gpu" ||
                                                 normalized_device == "vulkan");
#else
            err = "built without ncnn backend"; return nullptr;
#endif
        } else if (backend == "mnn") {
#ifdef USE_MNN
            if (normalized_device == "trt" || normalized_device == "tensorrt" ||
                normalized_device == "coreml") {
                err = "MNN supports cpu, cuda, vulkan, or opencl devices; got " + device;
                return nullptr;
            }
            std::string fwd = "cpu";
            if (normalized_device == "vulkan") fwd = "vulkan";
            else if (normalized_device == "cuda") fwd = "cuda";
            else if (want_gpu) fwd = "opencl";
            return std::make_unique<MnnBackend>(model, threads, fwd);
#else
            err = "built without MNN backend (rebuild with -DUSE_MNN=ON)"; return nullptr;
#endif
        } else if (backend == "trt") {
#ifdef USE_TRT
            return std::make_unique<TrtBackend>(model);
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
