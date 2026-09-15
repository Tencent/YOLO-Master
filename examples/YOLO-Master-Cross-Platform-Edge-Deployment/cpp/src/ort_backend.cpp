#include "ort_backend.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// ORT takes the model path as wchar_t* on Windows, char* elsewhere (ORTCHAR_T).
#ifdef _WIN32
static std::wstring ort_path(const std::string& s) {
    if (s.empty()) return {};
    const int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                           s.data(), static_cast<int>(s.size()),
                                           nullptr, 0);
    if (needed <= 0)
        throw std::runtime_error("model path is not valid UTF-8");
    std::wstring wide(static_cast<size_t>(needed), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                            s.data(), static_cast<int>(s.size()),
                            wide.data(), needed) != needed)
        throw std::runtime_error("failed to convert model path to UTF-16");
    return wide;
}
#else
static const std::string& ort_path(const std::string& s) { return s; }
#endif

// ORT exposes FP16 tensors as 16-bit storage.  Keep the conversion local so
// the runtime accepts both FP32 and ``--half`` exports without depending on a
// particular Ort::Float16_t constructor (which changed between ORT releases).
static float half_to_float(uint16_t bits) {
    const uint32_t sign = (bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t frac = bits & 0x03ffu;
    uint32_t value;
    if (exp == 0) {
        if (frac == 0) value = sign;
        else {
            uint32_t mant = frac;
            uint32_t e = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++e; }
            mant &= 0x03ffu;
            // Half subnormals have an implicit exponent of -14 before the
            // leading-bit normalization (not -15).  Using 127-15 here
            // underestimates every non-zero subnormal by a factor of two.
            const int exponent = 127 - 14 - static_cast<int>(e);
            value = sign | (static_cast<uint32_t>(exponent) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        value = sign | 0x7f800000u | (frac << 13);
    } else {
        value = sign | ((exp + (127u - 15u)) << 23) | (frac << 13);
    }
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent_bits = (bits >> 23) & 0xffu;
    uint32_t fraction = bits & 0x7fffffu;

    if (exponent_bits == 0xffu) {
        // Preserve infinities and emit a quiet, non-zero payload for NaNs.
        return static_cast<uint16_t>(
            sign | 0x7c00u | (fraction ? (0x0200u | (fraction >> 13)) : 0u));
    }

    int exponent = static_cast<int>(exponent_bits) - 127;
    if (exponent > 15) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exponent >= -14) {
        // Round the 23-bit float mantissa to ten bits using round-to-nearest,
        // ties-to-even.  Carrying out of the mantissa increments the exponent.
        fraction += 0x0fffu + ((fraction >> 13) & 1u);
        if (fraction & 0x800000u) {
            fraction = 0;
            if (++exponent > 15) return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(
            sign | (static_cast<uint32_t>(exponent + 15) << 10) |
            (fraction >> 13));
    }
    // Values at exponent -25 can round to the smallest half subnormal
    // (2^-24); only smaller exponents are guaranteed to round to zero.
    if (exponent < -25) return static_cast<uint16_t>(sign);

    // Half subnormal: restore float's implicit leading bit, shift to the
    // half-subnormal exponent, then apply the same ties-to-even rule.
    const uint32_t mantissa = fraction | 0x800000u;
    const int shift = -exponent - 1;  // 14 bits at exp=-14, 24 at exp=-24
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder_mask = (1u << shift) - 1u;
    const uint32_t remainder = mantissa & remainder_mask;
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
    return static_cast<uint16_t>(sign | rounded);
}

static std::vector<float> tensor_to_float(const Ort::Value& value) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    std::vector<float> output(count);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = value.GetTensorData<float>();
        std::copy(data, data + count, output.begin());
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const uint16_t* data = value.GetTensorData<uint16_t>();
        for (size_t i = 0; i < count; ++i) output[i] = half_to_float(data[i]);
    } else {
        throw std::runtime_error("ONNX tensor must use FP32 or FP16 elements");
    }
    if (!std::all_of(output.begin(), output.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("ONNX tensor contains NaN or Inf");
    return output;
}

OrtBackend::OrtBackend(const std::string& model_path, int threads, const std::string& device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolomaster") {
    opts_.SetIntraOpNumThreads(threads);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device == "trt" || device == "tensorrt") {
        // ONNXRuntime TensorRT EP: builds+caches a TRT engine internally (near-native TRT),
        // honors QDQ nodes for INT8 + FP16 elsewhere, and auto-falls-back to CUDA/CPU for
        // unsupported subgraphs. Portable: ship the .onnx; the engine cache builds on first run.
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        try {
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt));
            const char* keys[] = {"trt_fp16_enable", "trt_int8_enable",
                                  "trt_engine_cache_enable", "trt_engine_cache_path"};
            // Do not force INT8 for an arbitrary ONNX model.  TensorRT INT8
            // requires a calibrated/QDQ graph; enabling it unconditionally can
            // change the accuracy protocol or make engine construction fail.
            // Q/DQ nodes in an explicitly quantized model are still honored by
            // TensorRT when this option is disabled.
            const char* vals[] = {"1", "0", "1", "trt_engine_cache"};
            Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(trt, keys, vals, 4));
            opts_.AppendExecutionProvider_TensorRT_V2(*trt);
            Ort::GetApi().ReleaseTensorRTProviderOptions(trt);
            trt = nullptr;
            active_ep = "TensorRT-EP";
        } catch (const std::exception& e) {
            if (trt) Ort::GetApi().ReleaseTensorRTProviderOptions(trt);
            std::cerr << "[ort] TensorRT EP unavailable (" << e.what() << "); trying CUDA\n";
            ep_note = std::string("TensorRT EP unavailable: ") + e.what();
        }
        // CUDA fallback for TRT-unsupported nodes (and if the TRT EP failed to load)
        try {
            OrtCUDAProviderOptions cuda{}; cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);
            if (active_ep != "TensorRT-EP") active_ep = "CUDA";
        } catch (const std::exception& e) {
            if (active_ep != "TensorRT-EP") { std::cerr << "[ort] CUDA EP unavailable; using CPU\n"; active_ep = "CPU"; }
            if (active_ep == "CPU" && ep_note.empty())
                ep_note = std::string("CUDA EP unavailable: ") + e.what();
        }
    } else if (device == "cuda") {
        try {                                    // graceful fallback if CUDA EP can't load
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);
            active_ep = "CUDA";
        } catch (const std::exception& e) {
            std::cerr << "[ort] CUDA EP unavailable (" << e.what() << "); using CPU\n";
            active_ep = "CPU";
            ep_note = std::string("CUDA EP failed: ") + e.what();
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
            active_ep = "CoreML";
        } catch (const std::exception& e) {
            std::cerr << "[ort] CoreML EP unavailable (" << e.what() << "); using CPU\n";
            active_ep = "CPU";
            ep_note = std::string("CoreML EP unavailable: ") + e.what();
        }
    }
    // Provider registration can succeed even when provider initialization is
    // deferred until the session is constructed (for example, a CUDA/TensorRT
    // library may be missing at runtime).  Retry with a clean CPU-only option
    // set so the documented accelerator fallback also covers that case.
    auto create_session = [&](Ort::SessionOptions& options) {
#ifdef _WIN32
        const std::wstring wide = ort_path(model_path);
        return std::make_unique<Ort::Session>(env_, wide.c_str(), options);
#else
        return std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
#endif
    };
    try {
        session_ = create_session(opts_);
    } catch (const std::exception& first_error) {
        if (device == "cpu" || device.empty()) throw;
        const std::string requested_ep = active_ep;
        Ort::SessionOptions cpu_opts;
        cpu_opts.SetIntraOpNumThreads(threads);
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
            session_ = create_session(cpu_opts);
        } catch (const std::exception& cpu_error) {
            throw std::runtime_error(
                std::string("ONNX session initialization failed for ") + requested_ep +
                ": " + first_error.what() + "; CPU fallback failed: " + cpu_error.what());
        }
        active_ep = "CPU";
        ep_note = requested_ep + " unavailable; fell back to CPU: " + first_error.what();
        std::cerr << "[ort] " << ep_note << "\n";
    }

    const size_t n_in = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    if (n_in != 1)
        throw std::runtime_error("ONNX runner requires exactly one tensor input (found " +
                                 std::to_string(n_in) + ")");
    if (n_out == 0)
        throw std::runtime_error("ONNX model must expose at least one output");
    for (size_t i = 0; i < n_in; ++i)
        in_names_s_.push_back(session_->GetInputNameAllocated(i, alloc_).get());
    for (size_t i = 0; i < n_out; ++i)
        out_names_s_.push_back(session_->GetOutputNameAllocated(i, alloc_).get());
    for (auto& s : in_names_s_) in_names_.push_back(s.c_str());
    for (auto& s : out_names_s_) out_names_.push_back(s.c_str());
    if (in_names_.empty() || out_names_.empty())
        throw std::runtime_error("ONNX model must expose at least one input and one output");

    const auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto input_shape = input_info.GetShape();
    if (input_shape.size() != 4)
        throw std::runtime_error("ONNX input must have rank-4 shape [1,3,H,W]");
    if (input_shape[0] > 0 && input_shape[0] != 1)
        throw std::runtime_error("ONNX input batch dimension must be 1");
    if (input_shape[1] > 0 && input_shape[1] != 3)
        throw std::runtime_error("ONNX input channel dimension must be 3");
    for (size_t axis = 2; axis < 4; ++axis) {
        if (input_shape[axis] == 0 || input_shape[axis] < -1)
            throw std::runtime_error("ONNX input has an invalid spatial dimension");
    }
    if (input_shape[2] > 0 && input_shape[3] > 0 && input_shape[2] != input_shape[3])
        throw std::runtime_error("ONNX runner requires a square input (H must equal W)");
    const auto input_type = input_info.GetElementType();
    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        input_fp16_ = true;
    } else if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("ONNX input must use FP32 or FP16 elements");
    }

    // detect a static input size (H==W>0) -> hard constraint
    {
        auto shape = input_shape;
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
        if (p != std::string::npos) {
            const int metadata_imgsz = std::atoi(s.c_str() + p);
            if (metadata_imgsz > 0 && fixed_imgsz == 0) {
                meta_imgsz = metadata_imgsz;
            } else if (metadata_imgsz > 0 && metadata_imgsz != fixed_imgsz) {
                std::cerr << "[ort] metadata imgsz=" << metadata_imgsz
                          << " differs from static input=" << fixed_imgsz
                          << "; using the static input shape\n";
            }
        }
    }
}

std::vector<Detection> OrtBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> NCHW float RGB /255 ----
    if (cfg.imgsz <= 0)
        throw std::runtime_error("ONNX inference requires a positive image size");
    auto t0 = clk::now();
    const auto runtime_input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (runtime_input_shape.size() != 4 ||
        (runtime_input_shape[0] > 0 && runtime_input_shape[0] != 1) ||
        (runtime_input_shape[1] > 0 && runtime_input_shape[1] != 3)) {
        throw std::runtime_error("ONNX input shape changed to an unsupported layout; expected [1,3,H,W]");
    }
    if (runtime_input_shape[2] > 0 && runtime_input_shape[2] != cfg.imgsz)
        throw std::runtime_error("ONNX input height is fixed at " + std::to_string(runtime_input_shape[2]) +
                                 "; requested imgsz=" + std::to_string(cfg.imgsz));
    if (runtime_input_shape[3] > 0 && runtime_input_shape[3] != cfg.imgsz)
        throw std::runtime_error("ONNX input width is fixed at " + std::to_string(runtime_input_shape[3]) +
                                 "; requested imgsz=" + std::to_string(cfg.imgsz));
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, cfg.imgsz, cfg.stretch, lb);   // imgsz x imgsz, CV_8UC3 BGR
    // NCHW float RGB /255 (replaces cv::dnn::blobFromImage with swapRB=true)
    const int sz = cfg.imgsz;
    const size_t hw = static_cast<size_t>(sz) * static_cast<size_t>(sz);
    std::vector<float> blob(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;          // BGR
            const size_t idx = static_cast<size_t>(y) * sz + x;
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
    std::vector<uint16_t> blob16;
    Ort::Value in_tensor{nullptr};
    if (input_fp16_) {
        blob16.resize(blob.size());
        for (size_t i = 0; i < blob.size(); ++i) blob16[i] = float_to_half(blob[i]);
        in_tensor = Ort::Value::CreateTensor(
            mem, blob16.data(), blob16.size() * sizeof(uint16_t),
            in_shape.data(), in_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    } else {
        in_tensor = Ort::Value::CreateTensor<float>(
            mem, blob.data(), blob.size(), in_shape.data(), in_shape.size());
    }
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names_.data(), &in_tensor, 1,
                              out_names_.data(), out_names_.size());
    infer_ms = ms_since(t1);

    // ---- postprocess: detection is the rank-3 output [1,feat,anchors]; proto (seg) is rank-4 ----
    auto t2 = clk::now();
    int det_i = -1, proto_i = -1;
    std::vector<int> proto_candidates;
    struct DetectionCandidate {
        int index = -1;
        int features = 0;
        int anchors = 0;
        int distance = 0;
        size_t elements = 0;
        bool has_objectness = false;
        int mask_channels = 0;
    };
    std::vector<DetectionCandidate> detection_candidates;
    const int expected_features = std::max(5, 4 + cfg.num_classes());
    for (size_t i = 0; i < outs.size(); ++i) {
        auto info = outs[i].GetTensorTypeAndShapeInfo();
        const auto shape_i = info.GetShape();
        const auto type = info.GetElementType();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            continue;
        if (shape_i.size() == 4) {
            if (shape_i[0] != 1 || shape_i[1] <= 0 || shape_i[2] <= 0 || shape_i[3] <= 0)
                throw std::runtime_error("ONNX rank-4 output has an invalid shape; expected [1,C,H,W]");
            if (shape_i[1] > std::numeric_limits<int>::max() ||
                shape_i[2] > std::numeric_limits<int>::max() ||
                shape_i[3] > std::numeric_limits<int>::max())
                throw std::runtime_error("ONNX rank-4 output dimensions exceed runner limits");
            proto_candidates.push_back(static_cast<int>(i));
        } else if (shape_i.size() == 3 && shape_i[0] == 1 && shape_i[1] > 0 && shape_i[2] > 0) {
            if (shape_i[1] > std::numeric_limits<int>::max() ||
                shape_i[2] > std::numeric_limits<int>::max())
                throw std::runtime_error("ONNX detection output dimensions exceed runner limits");
            const int dim_a = static_cast<int>(shape_i[1]);
            const int dim_b = static_cast<int>(shape_i[2]);
            const int feat = std::min(dim_a, dim_b);
            const int anchors = std::max(dim_a, dim_b);
            if (feat < expected_features || anchors < feat) continue;
            const size_t elements = static_cast<size_t>(dim_a) * static_cast<size_t>(dim_b);
            detection_candidates.push_back({static_cast<int>(i), feat, anchors,
                                            std::abs(feat - expected_features), elements,
                                            false, 0});
        }
    }
    // A rank-3 tensor is a detection head only when its feature count matches
    // a detector layout, or when its extra channels match a rank-4 prototype.
    // Merely requiring features >= 4+nc can select an exported intermediate
    // feature map and yield plausible but invalid detections.
    detection_candidates.erase(
        std::remove_if(detection_candidates.begin(), detection_candidates.end(),
                       [&](DetectionCandidate& candidate) {
                           if (candidate.features == expected_features) return false;
                           if (candidate.features == expected_features + 1) {
                               candidate.has_objectness = true;
                               return false;
                           }
                           bool matched = false;
                           for (const int proto_index : proto_candidates) {
                               const auto proto_shape =
                                   outs[proto_index].GetTensorTypeAndShapeInfo().GetShape();
                               const int channels = static_cast<int>(proto_shape[1]);
                               bool objectness = false;
                               if (candidate.features == expected_features + channels) {
                                   objectness = false;
                               } else if (candidate.features == expected_features + 1 + channels) {
                                   objectness = true;
                               } else {
                                   continue;
                               }
                               if (matched && (candidate.has_objectness != objectness ||
                                               candidate.mask_channels != channels)) {
                                   throw std::runtime_error(
                                       "ONNX detection layout is ambiguous across prototype outputs");
                               }
                               matched = true;
                               candidate.has_objectness = objectness;
                               candidate.mask_channels = channels;
                           }
                           return !matched;
                       }),
        detection_candidates.end());
    if (detection_candidates.empty())
        throw std::runtime_error(
            "ONNX model has no FP32 rank-3 detection output (FP16 is also accepted) "
            "with a compatible feature dimension");
    std::sort(detection_candidates.begin(), detection_candidates.end(),
              [](const DetectionCandidate& a, const DetectionCandidate& b) {
                  if ((a.mask_channels > 0) != (b.mask_channels > 0))
                      return a.mask_channels > 0;
                  if (a.distance != b.distance) return a.distance < b.distance;
                  if (a.elements != b.elements) return a.elements > b.elements;
                  return a.index < b.index;
              });
    const DetectionCandidate& best_candidate = detection_candidates.front();
    std::vector<const DetectionCandidate*> tied;
    for (const DetectionCandidate& candidate : detection_candidates) {
        if ((candidate.mask_channels > 0) != (best_candidate.mask_channels > 0) ||
            candidate.distance != best_candidate.distance ||
            candidate.elements != best_candidate.elements) break;
        tied.push_back(&candidate);
    }
    if (tied.size() > 1) {
        std::ostringstream msg;
        msg << "ONNX detection output is ambiguous; equally plausible rank-3 tensors: ";
        for (size_t j = 0; j < tied.size(); ++j) {
            if (j) msg << ", ";
            const int index = tied[j]->index;
            msg << "'" << (index < static_cast<int>(out_names_s_.size())
                                ? out_names_s_[index] : std::to_string(index))
                << "' [1," << outs[index].GetTensorTypeAndShapeInfo().GetShape()[1]
                << "," << outs[index].GetTensorTypeAndShapeInfo().GetShape()[2] << "]";
        }
        msg << "; provide a model with one detection head";
        throw std::runtime_error(msg.str());
    }
    det_i = best_candidate.index;
    const bool has_objectness = best_candidate.has_objectness;
    const int mask_channels = best_candidate.mask_channels;
    auto shape = outs[det_i].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0)
        throw std::runtime_error("ONNX detection output must have shape [1, features, anchors]");
    const int first = static_cast<int>(shape[1]);
    const int second = static_cast<int>(shape[2]);
    const int feat_dim = std::min(first, second);
    const int num_anchors = std::max(first, second);
    if (feat_dim < expected_features || num_anchors < feat_dim)
        throw std::runtime_error("ONNX detection output dimensions are not plausible");
    std::vector<float> out_values = tensor_to_float(outs[det_i]);
    const float* out = out_values.data();
    std::vector<float> transposed;
    if (first <= second) {
        candidates = decode_candidates(out, feat_dim, num_anchors, cfg, lb, has_objectness);
    } else {
        // Some exporters emit [1, anchors, features].  Normalize that layout
        // before entering the shared decoder instead of silently swapping box
        // coordinates and class scores.
        transposed.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int anchor = 0; anchor < num_anchors; ++anchor)
            for (int feature = 0; feature < feat_dim; ++feature)
                transposed[static_cast<size_t>(feature) * num_anchors + anchor] =
                    out[static_cast<size_t>(anchor) * feat_dim + feature];
        candidates = decode_candidates(transposed.data(), feat_dim, num_anchors, cfg, lb,
                                       has_objectness);
    }
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    // A rank-4 tensor is a segmentation prototype only when its channel count
    // agrees with the mask-coefficient tail of the selected detection head.
    // This avoids treating an unrelated feature map as a mask tensor.
    for (const int candidate : proto_candidates) {
        const auto candidate_shape = outs[candidate].GetTensorTypeAndShapeInfo().GetShape();
        if (mask_channels > 0 && candidate_shape[1] == mask_channels) {
            if (proto_i >= 0)
                throw std::runtime_error(
                    "ONNX model has multiple prototype outputs matching the detection head");
            proto_i = candidate;
        }
    }
    if (mask_channels > 0 && proto_i < 0)
        throw std::runtime_error("ONNX detection head declares mask coefficients but no compatible prototype output exists");
    if (proto_i >= 0) {                                                // segmentation model
        auto ps = outs[proto_i].GetTensorTypeAndShapeInfo().GetShape();  // {1, nm, mh, mw}
        proto_c = (int)ps[1]; proto_h = (int)ps[2]; proto_w = (int)ps[3];
        proto = tensor_to_float(outs[proto_i]);
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
