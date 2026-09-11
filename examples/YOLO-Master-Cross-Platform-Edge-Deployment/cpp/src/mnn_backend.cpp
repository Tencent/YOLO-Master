#include "mnn_backend.hpp"
#include <MNN/MNNForwardType.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static std::string shape_string(const std::vector<int>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

static void validate_input_shape(const std::vector<int>& shape, int requested_imgsz = 0) {
    if (shape.size() != 4)
        throw std::runtime_error("MNN input must have rank-4 shape [1,3,H,W], got " + shape_string(shape));
    if (shape[0] > 0 && shape[0] != 1)
        throw std::runtime_error("MNN input batch dimension must be 1, got " + std::to_string(shape[0]));
    if (shape[1] > 0 && shape[1] != 3)
        throw std::runtime_error("MNN input channel dimension must be 3, got " + std::to_string(shape[1]));
    for (size_t i = 2; i < 4; ++i) {
        if (shape[i] == 0 || shape[i] < -1)
            throw std::runtime_error("MNN input has invalid spatial dimension in " + shape_string(shape));
    }
    if (shape[2] > 0 && shape[3] > 0 && shape[2] != shape[3])
        throw std::runtime_error("MNN runner requires square input, got " + shape_string(shape));
    if (requested_imgsz > 0) {
        if (shape[2] > 0 && shape[2] != requested_imgsz)
            throw std::runtime_error("MNN input height is fixed at " + std::to_string(shape[2]) +
                                     "; requested imgsz=" + std::to_string(requested_imgsz));
        if (shape[3] > 0 && shape[3] != requested_imgsz)
            throw std::runtime_error("MNN input width is fixed at " + std::to_string(shape[3]) +
                                     "; requested imgsz=" + std::to_string(requested_imgsz));
    }
}

static bool detection_shape(const std::vector<int>& shape, int expected_features,
                            int& feat_dim, int& num_anchors) {
    if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0)
        return false;
    const int feature = std::min(shape[1], shape[2]);
    const int anchors = std::max(shape[1], shape[2]);
    if (feature < expected_features || anchors < feature)
        return false;
    feat_dim = feature;
    num_anchors = anchors;
    return true;
}

static size_t checked_elements(const std::vector<int>& shape, const char* label) {
    if (shape.empty()) throw std::runtime_error(std::string("MNN ") + label + " shape is empty");
    size_t count = 1;
    for (const int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error(std::string("MNN ") + label +
                                     " shape has a non-positive dimension: " + shape_string(shape));
        }
        const size_t extent = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / extent)
            throw std::runtime_error(std::string("MNN ") + label + " shape is too large");
        count *= extent;
    }
    return count;
}

// MNN has kept the Interpreter API source-compatible across releases, but a
// few builds return ErrorCode/bool while others return void for tensor-copy
// helpers.  Check every status-bearing variant without baking one SDK's return
// type into this example.
template <typename Fn>
static void checked_mnn_call(const char* operation, Fn&& fn) {
    using result_type = std::invoke_result_t<Fn&>;
    if constexpr (std::is_void_v<result_type>) {
        fn();
    } else {
        const auto status = fn();
        using status_type = std::decay_t<decltype(status)>;
        bool failed = false;
        if constexpr (std::is_same_v<status_type, bool>) {
            failed = !status;
        } else if constexpr (std::is_pointer_v<status_type>) {
            failed = status == nullptr;
        } else if constexpr (std::is_integral_v<status_type> ||
                             std::is_enum_v<status_type>) {
            // MNN::ErrorCode uses NO_ERROR == 0; integer-compatible status
            // values follow the same convention in older SDKs.
            failed = static_cast<long long>(status) != 0;
        }
        if (failed)
            throw std::runtime_error(std::string("MNN ") + operation + " failed");
    }
}

template <typename TensorT, typename = void>
struct has_tensor_type : std::false_type {};

template <typename TensorT>
struct has_tensor_type<TensorT, std::void_t<decltype(std::declval<TensorT&>().getType())>>
    : std::true_type {};

template <typename TensorT>
static bool is_float32_tensor_impl(TensorT* tensor) {
    if (!tensor) return false;
    if constexpr (has_tensor_type<TensorT>::value) {
        const auto type = tensor->getType();
        return type.code == halide_type_float && type.bits == 32;
    }
    // Very old MNN headers do not expose getType(); host<float>() remains the
    // fallback validation after the tensor is copied to a host tensor.
    return true;
}

static bool is_float32_tensor(MNN::Tensor* tensor) {
    return is_float32_tensor_impl(tensor);
}

static void require_float32(MNN::Tensor* tensor, const char* label) {
    if (!tensor) throw std::runtime_error(std::string("MNN ") + label + " tensor is null");
    if (!is_float32_tensor_impl(tensor)) {
        throw std::runtime_error(std::string("MNN ") + label +
                                 " tensor must be float32 (unsupported element type)");
    }
}

MnnBackend::MnnBackend(const std::string& model_path, int threads, const std::string& forward)
    : threads_(threads) {
    interp_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model_path.c_str()),
        [](MNN::Interpreter* p) { if (p) MNN::Interpreter::destroy(p); });
    if (!interp_) throw std::runtime_error("MNN: failed to load " + model_path);

    const bool requested_gpu = forward == "opencl" || forward == "vulkan" || forward == "cuda";
    const auto requested_type = forward == "opencl" ? MNN_FORWARD_OPENCL
                               : forward == "vulkan" ? MNN_FORWARD_VULKAN
                               : forward == "cuda"   ? MNN_FORWARD_CUDA
                                                      : MNN_FORWARD_CPU;
    const std::string requested_ep = forward == "opencl" ? "MNN-OpenCL"
                                   : forward == "vulkan" ? "MNN-Vulkan"
                                   : forward == "cuda"   ? "MNN-CUDA" : "MNN-CPU";

    auto make_schedule = [&](decltype(MNN_FORWARD_CPU) type, MNN::BackendConfig& bc) {
        MNN::ScheduleConfig config{};
        config.numThread = threads;
        config.type = type;
        // Keep unsupported operators on CPU.  If the requested accelerator
        // cannot be initialized at all, the constructor below retries with a
        // clean CPU-only session instead of leaving a null backend.
        config.backupType = MNN_FORWARD_CPU;
        const bool gpu = type != MNN_FORWARD_CPU;
        // fp16 on GPU (OpenCL/Vulkan/CUDA) for speed; fp32 on CPU for parity.
        bc.precision = gpu ? MNN::BackendConfig::Precision_Low
                           : MNN::BackendConfig::Precision_High;
        bc.power = MNN::BackendConfig::Power_High;
        config.backendConfig = &bc;
        return config;
    };

    MNN::BackendConfig requested_bc;
    MNN::ScheduleConfig requested_sc = make_schedule(requested_type, requested_bc);
    std::string session_error;
    auto create_session = [&](MNN::ScheduleConfig config) -> MNN::Session* {
        try {
            return interp_->createSession(config);
        } catch (const std::exception& e) {
            session_error = e.what();
        } catch (...) {
            session_error = "unknown MNN exception";
        }
        return nullptr;
    };
    session_ = create_session(requested_sc);
    if (!session_ && requested_gpu) {
        // Some MNN builds report an unavailable OpenCL/Vulkan/CUDA backend by
        // returning nullptr from createSession.  Retry with a fresh CPU
        // schedule so --device gpu remains usable on a CPU-only target.
        std::string reason = requested_ep + " session creation failed";
        if (!session_error.empty()) reason += ": " + session_error;
        MNN::BackendConfig cpu_bc;
        MNN::ScheduleConfig cpu_sc = make_schedule(MNN_FORWARD_CPU, cpu_bc);
        session_error.clear();
        session_ = create_session(cpu_sc);
        if (!session_)
            throw std::runtime_error("MNN: " + reason + "; CPU fallback session creation failed" +
                                     (session_error.empty() ? std::string() : ": " + session_error));
        active_ep = "MNN-CPU";
        ep_note = reason + "; fell back to CPU";
        std::cerr << "[mnn] " << ep_note << "\n";
    } else if (!session_) {
        throw std::runtime_error("MNN: createSession failed for " + model_path +
                                 (session_error.empty() ? std::string() : ": " + session_error));
    } else {
        active_ep = requested_ep;
        if (requested_gpu)
            ep_note = requested_ep + " configured with CPU backup for unsupported operators";
    }

    std::string io_error;
    auto resolve_io = [&]() -> bool {
        try {
            input_ = interp_->getSessionInput(session_, nullptr);    // first input
            output_ = interp_->getSessionOutput(session_, nullptr);   // first output
        } catch (const std::exception& e) {
            io_error = e.what();
            input_ = nullptr;
            output_ = nullptr;
        } catch (...) {
            io_error = "unknown MNN exception";
            input_ = nullptr;
            output_ = nullptr;
        }
        return input_ != nullptr && output_ != nullptr;
    };
    const bool io_ok = resolve_io();
    if (!io_ok && requested_gpu && active_ep != "MNN-CPU") {
        // A few releases defer backend setup until tensors are resolved.  If
        // that path yields incomplete I/O, release it and retry CPU once.
        try {
            checked_mnn_call("releaseSession", [&] { return interp_->releaseSession(session_); });
        } catch (const std::exception& e) {
            // Continue with the CPU retry; some old MNN builds report a
            // teardown warning even though the session can be replaced.
            io_error = std::string("accelerator release warning: ") + e.what();
        }
        session_ = nullptr;
        MNN::BackendConfig cpu_bc;
        MNN::ScheduleConfig cpu_sc = make_schedule(MNN_FORWARD_CPU, cpu_bc);
        session_error.clear();
        session_ = create_session(cpu_sc);
        if (session_) resolve_io();
        if (!session_ || !input_ || !output_)
            throw std::runtime_error("MNN: accelerator I/O resolution failed and CPU fallback failed" +
                                     (!session_error.empty() ? ": " + session_error
                                                              : (io_error.empty() ? std::string() : ": " + io_error)));
        active_ep = "MNN-CPU";
        ep_note = requested_ep + " I/O resolution failed; fell back to CPU";
        std::cerr << "[mnn] " << ep_note << "\n";
    }
    if (!input_ || !output_)
        throw std::runtime_error("MNN: could not resolve input/output tensor" +
                                 (io_error.empty() ? std::string() : ": " + io_error));
    require_float32(input_, "input");

    // YOLO-Master graphs bake the attention token counts at the training size -> fixed input.
    auto ishape = input_->shape();   // NCHW, e.g. {1,3,640,640}
    validate_input_shape(ishape);
    if (ishape[2] > 0 && ishape[2] == ishape[3]) {
        fixed_imgsz = ishape[2];
        meta_imgsz  = ishape[2];
    }
    // MNN has no built-in metadata map -> read an optional class-name sidecar. Prefer a per-model
    // "<model>.metadata.yaml" (so several .mnn can share a dir); fall back to "metadata.yaml".
    const std::filesystem::path mp(model_path);
    const std::string per_model = (mp.parent_path() / (mp.stem().string() + ".metadata.yaml")).string();
    const std::string shared    = (mp.parent_path() / "metadata.yaml").string();
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(per_model, nm, mi) || meta::read_ncnn_yaml(shared, nm, mi)) {
        meta_names = nm;
        if (mi > 0) { meta_imgsz = mi; if (fixed_imgsz == 0) fixed_imgsz = mi; }
    }
}

MnnBackend::~MnnBackend() {
    if (interp_ && session_) {
        // Destructors must not throw, but still surface a non-zero SDK status
        // while keeping teardown noexcept.
        try {
            checked_mnn_call("releaseSession", [&] { return interp_->releaseSession(session_); });
        } catch (...) {}
    }
}

std::vector<Detection> MnnBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> NCHW float RGB /255 (identical to ORT) ----
    if (cfg.imgsz <= 0) throw std::runtime_error("MNN inference requires a positive image size");
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, cfg.imgsz, cfg.stretch, lb);   // imgsz x imgsz, CV_8UC3 BGR
    const int sz = cfg.imgsz;
    const size_t hw = static_cast<size_t>(sz) * static_cast<size_t>(sz);
    std::vector<float> blob(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;           // BGR
            const size_t idx = static_cast<size_t>(y) * sz + x;
            blob[idx]          = px[2] * (1.0f / 255);  // R
            blob[hw + idx]     = px[1] * (1.0f / 255);  // G
            blob[2 * hw + idx] = px[0] * (1.0f / 255);  // B
        }
    }
    // resize the session input if it doesn't already match imgsz (handles fixed & flexible graphs)
    auto ishape = input_->shape();
    validate_input_shape(ishape);
    if (ishape.size() != 4 || ishape[2] != sz || ishape[3] != sz) {
        checked_mnn_call("resizeTensor", [&] {
            return interp_->resizeTensor(input_, std::vector<int>{1, 3, sz, sz});
        });
        checked_mnn_call("resizeSession", [&] { return interp_->resizeSession(session_); });
        input_ = interp_->getSessionInput(session_, nullptr);
        output_ = interp_->getSessionOutput(session_, nullptr);
        require_float32(input_, "input");
        if (!output_) throw std::runtime_error("MNN: output tensor disappeared after resize");
    }
    validate_input_shape(input_->shape(), sz);
    pre_ms = ms_since(t0);

    // ---- inference: copy blob into the input tensor (NCHW/Caffe), run ----
    auto t1 = clk::now();
    {
        MNN::Tensor host(input_, MNN::Tensor::CAFFE);   // NCHW host tensor shaped like input_
        const size_t host_count = checked_elements(host.shape(), "input");
        if (host_count != blob.size())
            throw std::runtime_error("MNN input tensor size does not match the requested image size");
        float* host_data = host.host<float>();
        if (!host_data) throw std::runtime_error("MNN input host tensor has no float storage");
        std::memcpy(host_data, blob.data(), blob.size() * sizeof(float));
        checked_mnn_call("copyFromHostTensor", [&] {
            return input_->copyFromHostTensor(&host);
        });
    }
    checked_mnn_call("runSession", [&] { return interp_->runSession(session_); });
    infer_ms = ms_since(t1);

    // ---- postprocess: detection = rank-3 output [1,feat,anchors]; proto (seg) = rank-4 [1,nm,mh,mw] ----
    auto t2 = clk::now();
    auto all = interp_->getSessionOutputAll(session_);
    MNN::Tensor* detT = nullptr; MNN::Tensor* protoT = nullptr;
    const int expected_features = std::max(5, 4 + cfg.num_classes());
    int det_features = 0, det_anchors = 0;
    struct DetectionCandidate {
        MNN::Tensor* tensor = nullptr;
        std::string name;
        int features = 0;
        int anchors = 0;
        int distance = 0;
        size_t elements = 0;
        bool has_objectness = false;
        int mask_channels = 0;
    };
    std::vector<DetectionCandidate> detection_candidates;
    struct ProtoCandidate { MNN::Tensor* tensor = nullptr; std::string name; };
    std::vector<ProtoCandidate> proto_candidates;
    for (auto& kv : all) {
        MNN::Tensor* tensor = kv.second;
        if (!tensor) continue;
        const std::string tensor_name = kv.first;
        const auto shape = tensor->shape();
        if (shape.size() == 4) {
            if (shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0)
                throw std::runtime_error("MNN prototype output must have shape [1,C,H,W], got " + shape_string(shape));
            (void)checked_elements(shape, "prototype");
            if (is_float32_tensor(tensor))
                proto_candidates.push_back({tensor, tensor_name});
            continue;
        }
        int feat = 0, anchors = 0;
        if (!detection_shape(shape, expected_features, feat, anchors)) continue;
        const int distance = std::abs(feat - expected_features);
        const size_t elements = static_cast<size_t>(shape[1]) * static_cast<size_t>(shape[2]);
        if (!is_float32_tensor(tensor))
            continue;
        detection_candidates.push_back(
            {tensor, tensor_name, feat, anchors, distance, elements, false, 0});
    }
    detection_candidates.erase(
        std::remove_if(detection_candidates.begin(), detection_candidates.end(),
                       [&](DetectionCandidate& candidate) {
                           if (candidate.features == expected_features) return false;
                           if (candidate.features == expected_features + 1) {
                               candidate.has_objectness = true;
                               return false;
                           }
                           bool matched = false;
                           for (const ProtoCandidate& proto_candidate : proto_candidates) {
                               const int channels = proto_candidate.tensor->shape()[1];
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
                                       "MNN detection layout is ambiguous across prototype outputs");
                               }
                               matched = true;
                               candidate.has_objectness = objectness;
                               candidate.mask_channels = channels;
                           }
                           return !matched;
                       }),
        detection_candidates.end());
    if (!detection_candidates.empty()) {
        std::sort(detection_candidates.begin(), detection_candidates.end(),
                  [](const DetectionCandidate& a, const DetectionCandidate& b) {
                      if ((a.mask_channels > 0) != (b.mask_channels > 0))
                          return a.mask_channels > 0;
                      if (a.distance != b.distance) return a.distance < b.distance;
                      if (a.elements != b.elements) return a.elements > b.elements;
                      return a.name < b.name;
                  });
        const DetectionCandidate& best = detection_candidates.front();
        size_t ties = 0;
        for (const DetectionCandidate& candidate : detection_candidates) {
            if ((candidate.mask_channels > 0) != (best.mask_channels > 0) ||
                candidate.distance != best.distance || candidate.elements != best.elements) break;
            ++ties;
        }
        if (ties > 1) {
            std::ostringstream msg;
            msg << "MNN detection output is ambiguous; equally plausible rank-3 tensors: ";
            for (size_t i = 0; i < ties; ++i) {
                if (i) msg << ", ";
                const auto& candidate = detection_candidates[i];
                msg << "'" << (candidate.name.empty() ? "<unnamed>" : candidate.name)
                    << "' [features=" << candidate.features
                    << ", anchors=" << candidate.anchors << "]";
            }
            msg << "; provide a model with one detection head";
            throw std::runtime_error(msg.str());
        }
        detT = best.tensor;
        det_features = best.features;
        det_anchors = best.anchors;
    }
    if (!detT) {
        // Some MNN graphs expose only the default output through
        // getSessionOutputAll(); validate it before using it as a fallback.
        detT = output_;
        if (!detT || !detection_shape(detT->shape(), expected_features, det_features, det_anchors))
            throw std::runtime_error("MNN model has no compatible rank-3 detection output");
    }
    bool has_objectness = false;
    int mask_channels = 0;
    if (!detection_candidates.empty()) {
        has_objectness = detection_candidates.front().has_objectness;
        mask_channels = detection_candidates.front().mask_channels;
    } else {
        const int fallback_features = det_features;
        if (fallback_features == expected_features + 1) {
            has_objectness = true;
        } else if (fallback_features != expected_features) {
            bool matched = false;
            for (const ProtoCandidate& candidate : proto_candidates) {
                const int channels = candidate.tensor->shape()[1];
                if (fallback_features == expected_features + channels ||
                    fallback_features == expected_features + 1 + channels) {
                    if (matched)
                        throw std::runtime_error("MNN default detection output layout is ambiguous");
                    matched = true;
                    has_objectness = fallback_features == expected_features + 1 + channels;
                    mask_channels = channels;
                }
            }
            if (!matched)
                throw std::runtime_error("MNN default output feature count is incompatible with the class/prototype layout");
        }
    }
    MNN::Tensor detHost(detT, MNN::Tensor::CAFFE);
    require_float32(detT, "detection output");
    checked_mnn_call("copyToHostTensor", [&] {
        return detT->copyToHostTensor(&detHost);
    });
    const auto os = detHost.shape();
    if (!detection_shape(os, expected_features, det_features, det_anchors))
        throw std::runtime_error("MNN detection output must have shape [1,features,anchors], got " + shape_string(os));
    const size_t det_count = checked_elements(os, "detection");
    const float* raw = detHost.host<float>();
    if (!raw) throw std::runtime_error("MNN detection output has no host data");
    for (size_t i = 0; i < det_count; ++i) {
        if (!std::isfinite(raw[i])) throw std::runtime_error("MNN detection output contains NaN or Inf");
    }
    int feat_dim = det_features, num_anchors = det_anchors;
    const float* dec = raw;
    std::vector<float> buf;
    if (os[1] > os[2]) {                                                   // [1,anchors,feat] -> transpose
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int a = 0; a < num_anchors; ++a)
            for (int f = 0; f < feat_dim; ++f)
                buf[static_cast<size_t>(f) * num_anchors + a] =
                    raw[static_cast<size_t>(a) * feat_dim + f];
        dec = buf.data();
    }
    candidates = decode_candidates(dec, feat_dim, num_anchors, cfg, lb, has_objectness);
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    if (mask_channels > 0) {
        for (const ProtoCandidate& candidate : proto_candidates) {
            const auto ps = candidate.tensor->shape();
            if (ps[1] == mask_channels) {
                if (protoT)
                    throw std::runtime_error("MNN model has multiple prototype outputs matching the detection head");
                protoT = candidate.tensor;
            }
        }
        if (!protoT)
            throw std::runtime_error("MNN detection head declares mask coefficients but no compatible prototype output exists");
    }
    if (protoT) {                                       // segmentation proto
        MNN::Tensor protoHost(protoT, MNN::Tensor::CAFFE);
        require_float32(protoT, "prototype output");
        checked_mnn_call("copyToHostTensor", [&] {
            return protoT->copyToHostTensor(&protoHost);
        });
        const auto ps = protoHost.shape();              // {1, nm, mh, mw}
        if (ps.size() != 4 || ps[0] != 1 || ps[1] <= 0 || ps[2] <= 0 || ps[3] <= 0)
            throw std::runtime_error("MNN prototype output has an invalid shape: " + shape_string(ps));
        const size_t proto_count = checked_elements(ps, "prototype");
        proto_c = (int)ps[1]; proto_h = (int)ps[2]; proto_w = (int)ps[3];
        const float* pp = protoHost.host<float>();
        if (!pp) throw std::runtime_error("MNN prototype output has no host data");
        for (size_t i = 0; i < proto_count; ++i)
            if (!std::isfinite(pp[i])) throw std::runtime_error("MNN prototype output contains NaN or Inf");
        proto.assign(pp, pp + (size_t)proto_c * proto_h * proto_w);
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
