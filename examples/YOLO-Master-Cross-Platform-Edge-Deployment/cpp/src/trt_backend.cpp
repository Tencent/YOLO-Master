#include "trt_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
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

static std::string dims_string(const nvinfer1::Dims& dims) {
    std::ostringstream out;
    out << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) out << ",";
        out << dims.d[i];
    }
    out << "]";
    return out.str();
}

static size_t checked_elements(std::initializer_list<int> dims, const char* what) {
    size_t total = 1;
    for (const int dim : dims) {
        if (dim <= 0 || total > std::numeric_limits<size_t>::max() /
                                  static_cast<size_t>(dim))
            throw std::runtime_error(std::string("TRT ") + what + " dimensions are invalid or too large");
        total *= static_cast<size_t>(dim);
    }
    return total;
}

static size_t checked_bytes(size_t elements, size_t element_size, const char* what) {
    if (element_size == 0 || elements > std::numeric_limits<size_t>::max() / element_size)
        throw std::runtime_error(std::string("TRT ") + what + " buffer is too large");
    return elements * element_size;
}

static bool supported_type(nvinfer1::DataType type, bool& fp16) {
    if (type == nvinfer1::DataType::kFLOAT) {
        fp16 = false;
        return true;
    }
    if (type == nvinfer1::DataType::kHALF) {
        fp16 = true;
        return true;
    }
    return false;
}

// TensorRT engines can expose FP16 I/O even when the graph itself is otherwise
// identical to an FP32 export.  Keep conversion on the host so the shared
// decoder always receives a finite float32 tensor.
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
        return static_cast<uint16_t>(
            sign | 0x7c00u | (fraction ? (0x0200u | (fraction >> 13)) : 0u));
    }
    int exponent = static_cast<int>(exponent_bits) - 127;
    if (exponent > 15) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exponent >= -14) {
        fraction += 0x0fffu + ((fraction >> 13) & 1u);
        if (fraction & 0x800000u) {
            fraction = 0;
            if (++exponent > 15) return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(
            sign | (static_cast<uint32_t>(exponent + 15) << 10) | (fraction >> 13));
    }
    if (exponent < -25) return static_cast<uint16_t>(sign);
    const uint32_t mantissa = fraction | 0x800000u;
    const int shift = -exponent - 1;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder_mask = (1u << shift) - 1u;
    const uint32_t remainder = mantissa & remainder_mask;
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
    return static_cast<uint16_t>(sign | rounded);
}

static void convert_half(const std::vector<uint16_t>& src, std::vector<float>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = half_to_float(src[i]);
}

TrtBackend::TrtBackend(const std::string& engine_path) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open TensorRT engine: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty()) throw std::runtime_error("TensorRT engine is empty: " + engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!runtime_) throw std::runtime_error("failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_)
        throw std::runtime_error("failed to deserialize engine (built for a different GPU arch / TRT version?)");
    ctx_.reset(engine_->createExecutionContext());
    if (!ctx_) throw std::runtime_error("failed to create TensorRT execution context");

    // The runner deliberately accepts one data input, one rank-3 detection
    // output, and at most one rank-4 prototype output.  Rejecting auxiliary or
    // ambiguous tensors here is preferable to decoding an intermediate feature
    // map as if it were a detector head.
    int input_count = 0;
    int detection_count = 0;
    int proto_count = 0;
    std::vector<std::string> unsupported;
    const int io_count = engine_->getNbIOTensors();
    if (io_count <= 0) throw std::runtime_error("TensorRT engine exposes no named I/O tensors");
    for (int i = 0; i < io_count; ++i) {
        const char* raw_name = engine_->getIOTensorName(i);
        const std::string name = raw_name ? raw_name : "";
        if (name.empty()) throw std::runtime_error("TensorRT engine contains an unnamed I/O tensor");
        const nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
        const auto mode = engine_->getTensorIOMode(name.c_str());
        bool fp16 = false;
        const auto type = engine_->getTensorDataType(name.c_str());
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (!supported_type(type, fp16))
                throw std::runtime_error("TRT input '" + name + "' must be FP32 or FP16");
            if (++input_count > 1)
                throw std::runtime_error("TRT runner requires exactly one input tensor (found multiple)");
            if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3 ||
                dims.d[2] <= 0 || dims.d[3] <= 0 || dims.d[2] != dims.d[3] ||
                dims.d[2] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("TRT input '" + name + "' must have a static square shape [1,3,H,W], got " +
                                         dims_string(dims));
            }
            in_name_ = name;
            in_sz_ = static_cast<int>(dims.d[2]);
            input_fp16_ = fp16;
            continue;
        }

        if (dims.nbDims == 3) {
            if (!supported_type(type, fp16))
                throw std::runtime_error("TRT detection output '" + name + "' must be FP32 or FP16");
            if (dims.d[0] != 1 || dims.d[1] <= 0 || dims.d[2] <= 0 ||
                dims.d[1] > std::numeric_limits<int>::max() ||
                dims.d[2] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("TRT detection output '" + name + "' must have static shape [1,F,A], got " +
                                         dims_string(dims));
            }
            const int axis0 = static_cast<int>(dims.d[1]);
            const int axis1 = static_cast<int>(dims.d[2]);
            if (axis0 == axis1 || std::min(axis0, axis1) < 5) {
                throw std::runtime_error("TRT detection output '" + name +
                                         "' has an ambiguous/non-detector layout " + dims_string(dims));
            }
            if (++detection_count > 1)
                throw std::runtime_error("TRT engine exposes multiple rank-3 outputs; provide one unambiguous detection head");
            out_name_ = name;
            out_dim0_ = axis0;
            out_dim1_ = axis1;
            feat_dim_ = std::min(axis0, axis1);
            num_anchors_ = std::max(axis0, axis1);
            output_fp16_ = fp16;
            continue;
        }

        if (dims.nbDims == 4) {
            if (!supported_type(type, fp16))
                throw std::runtime_error("TRT prototype output '" + name + "' must be FP32 or FP16");
            if (dims.d[0] != 1 || dims.d[1] <= 0 || dims.d[2] <= 0 || dims.d[3] <= 0 ||
                dims.d[1] > std::numeric_limits<int>::max() ||
                dims.d[2] > std::numeric_limits<int>::max() ||
                dims.d[3] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("TRT rank-4 output '" + name +
                                         "' must have static shape [1,C,H,W], got " + dims_string(dims));
            }
            if (++proto_count > 1)
                throw std::runtime_error("TRT engine exposes multiple rank-4 outputs; prototype selection is ambiguous");
            proto_name_ = name;
            pc_ = static_cast<int>(dims.d[1]);
            ph_ = static_cast<int>(dims.d[2]);
            pw_ = static_cast<int>(dims.d[3]);
            proto_fp16_ = fp16;
            continue;
        }
        unsupported.push_back(name + " " + dims_string(dims));
    }
    if (input_count != 1)
        throw std::runtime_error("TRT runner requires exactly one input tensor (found " +
                                 std::to_string(input_count) + ")");
    if (detection_count != 1)
        throw std::runtime_error("TRT engine has no rank-3 detection output");
    if (!unsupported.empty()) {
        std::ostringstream msg;
        msg << "TRT engine exposes unsupported auxiliary I/O tensors: ";
        for (size_t i = 0; i < unsupported.size(); ++i) {
            if (i) msg << ", ";
            msg << unsupported[i];
        }
        throw std::runtime_error(msg.str());
    }
    fixed_imgsz = in_sz_;
    active_ep = "TRT-CUDA";

    // Metadata sidecar (engines embed no names/imgsz):
    // <engine-minus-ext>.metadata.yaml, then metadata.yaml next to the engine.
    {
        namespace fs = std::filesystem;
        const fs::path ep(engine_path);
        for (const fs::path& p : {fs::path(ep).replace_extension(".metadata.yaml"),
                                  ep.parent_path() / "metadata.yaml"}) {
            std::vector<std::string> names;
            int misz = 0;
            std::error_code ec;
            if (fs::exists(p, ec) && meta::read_ncnn_yaml(p.string(), names, misz)) {
                meta_names = std::move(names);
                meta_imgsz = misz;
                if (misz > 0 && misz != in_sz_)
                    std::cerr << "[trt] warn: sidecar imgsz=" << misz
                              << " but engine input is " << in_sz_ << "px (" << p.string() << ")\n";
                break;
            }
        }
    }

    const size_t input_count_elements = checked_elements({3, in_sz_, in_sz_}, "input");
    const size_t output_count_elements = checked_elements({feat_dim_, num_anchors_}, "detection output");
    if (pc_ > 0) (void)checked_elements({pc_, ph_, pw_}, "prototype output");
    try {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaMalloc(&d_in_, checked_bytes(input_count_elements,
                                                    input_fp16_ ? sizeof(uint16_t) : sizeof(float),
                                                    "input")));
        CUDA_CHECK(cudaMalloc(&d_out_, checked_bytes(output_count_elements,
                                                     output_fp16_ ? sizeof(uint16_t) : sizeof(float),
                                                     "detection output")));
        h_out_.resize(output_count_elements);
        if (output_fp16_) h_out16_.resize(output_count_elements);
        if (pc_ > 0) {
            const size_t proto_count_elements = checked_elements({pc_, ph_, pw_}, "prototype output");
            CUDA_CHECK(cudaMalloc(&d_proto_, checked_bytes(proto_count_elements,
                                                            proto_fp16_ ? sizeof(uint16_t) : sizeof(float),
                                                            "prototype output")));
            h_proto_.resize(proto_count_elements);
            if (proto_fp16_) h_proto16_.resize(proto_count_elements);
        }
        if (!ctx_->setTensorAddress(in_name_.c_str(), d_in_))
            throw std::runtime_error("TRT failed to bind input tensor '" + in_name_ + "'");
        if (!ctx_->setTensorAddress(out_name_.c_str(), d_out_))
            throw std::runtime_error("TRT failed to bind detection output tensor '" + out_name_ + "'");
        if (d_proto_ && !ctx_->setTensorAddress(proto_name_.c_str(), d_proto_))
            throw std::runtime_error("TRT failed to bind prototype output tensor '" + proto_name_ + "'");
    } catch (...) {
        if (d_in_) { cudaFree(d_in_); d_in_ = nullptr; }
        if (d_out_) { cudaFree(d_out_); d_out_ = nullptr; }
        if (d_proto_) { cudaFree(d_proto_); d_proto_ = nullptr; }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        throw;
    }
}

TrtBackend::~TrtBackend() {
    if (d_in_) cudaFree(d_in_);
    if (d_out_) cudaFree(d_out_);
    if (d_proto_) cudaFree(d_proto_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Detection> TrtBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    if (cfg.imgsz != in_sz_)
        throw std::runtime_error("TRT engine has fixed input size " + std::to_string(in_sz_) +
                                 "; requested imgsz=" + std::to_string(cfg.imgsz));
    if (cfg.num_classes() <= 0)
        throw std::runtime_error("TRT inference requires a positive class count");

    // Resolve the detector ABI only after the caller's class profile is known.
    // YOLOv8/EsMoE heads are 4+nc, YOLOv5 heads are 4+1+nc; segmentation heads
    // append prototype coefficients after either layout.
    const int expected_features = std::max(5, 4 + cfg.num_classes());
    has_objectness_ = false;
    mask_channels_ = 0;
    if (feat_dim_ == expected_features) {
        // plain YOLOv8/EsMoE detection
    } else if (feat_dim_ == expected_features + 1) {
        has_objectness_ = true;
    } else if (pc_ > 0 && feat_dim_ == expected_features + pc_) {
        mask_channels_ = pc_;
    } else if (pc_ > 0 && feat_dim_ == expected_features + 1 + pc_) {
        has_objectness_ = true;
        mask_channels_ = pc_;
    } else {
        std::ostringstream msg;
        msg << "TRT detection feature count " << feat_dim_ << " is incompatible with nc="
            << cfg.num_classes();
        if (pc_ > 0) msg << " and prototype channels=" << pc_;
        throw std::runtime_error(msg.str());
    }
    if (pc_ > 0 && mask_channels_ == 0)
        throw std::runtime_error(
            "TRT engine exposes a rank-4 output, but the detection head has no matching mask coefficients");

    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, in_sz_, cfg.stretch, lb);
    const int sz = in_sz_;
    const size_t hw = checked_elements({sz, sz}, "input");
    std::vector<float> in(static_cast<size_t>(3) * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;  // BGR -> RGB /255, NCHW
            const size_t idx = static_cast<size_t>(y) * sz + x;
            in[idx] = px[2] * (1.0f / 255);
            in[hw + idx] = px[1] * (1.0f / 255);
            in[static_cast<size_t>(2) * hw + idx] = px[0] * (1.0f / 255);
        }
    }
    pre_ms = ms_since(t0);

    std::vector<uint16_t> in16;
    auto t1 = clk::now();
    if (input_fp16_) {
        in16.resize(in.size());
        for (size_t i = 0; i < in.size(); ++i) in16[i] = float_to_half(in[i]);
        CUDA_CHECK(cudaMemcpyAsync(d_in_, in16.data(), in16.size() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream_));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_in_, in.data(), in.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
    }
    if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("TRT enqueueV3 failed");
    if (output_fp16_) {
        CUDA_CHECK(cudaMemcpyAsync(h_out16_.data(), d_out_, h_out16_.size() * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream_));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(h_out_.data(), d_out_, h_out_.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
    }
    if (d_proto_ && mask_channels_ > 0) {
        if (proto_fp16_) {
            CUDA_CHECK(cudaMemcpyAsync(h_proto16_.data(), d_proto_, h_proto16_.size() * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, stream_));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(h_proto_.data(), d_proto_, h_proto_.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    infer_ms = ms_since(t1);
    if (output_fp16_) convert_half(h_out16_, h_out_);
    if (!std::all_of(h_out_.begin(), h_out_.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("TRT detection output contains NaN or Inf");

    auto t2 = clk::now();
    std::vector<float> transposed;
    const float* decoded = h_out_.data();
    if (out_dim0_ > out_dim1_) {
        transposed.resize(h_out_.size());
        for (int anchor = 0; anchor < num_anchors_; ++anchor)
            for (int feature = 0; feature < feat_dim_; ++feature)
                transposed[static_cast<size_t>(feature) * num_anchors_ + anchor] =
                    h_out_[static_cast<size_t>(anchor) * feat_dim_ + feature];
        decoded = transposed.data();
    }
    candidates = decode_candidates(decoded, feat_dim_, num_anchors_, cfg, lb, has_objectness_);
    cand_orig_w = lb.orig_w;
    cand_orig_h = lb.orig_h;
    cand_lb = lb;
    proto.clear();
    proto_c = proto_h = proto_w = 0;
    if (mask_channels_ > 0 && d_proto_) {
        if (proto_fp16_) convert_half(h_proto16_, h_proto_);
        if (!std::all_of(h_proto_.begin(), h_proto_.end(),
                         [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("TRT prototype output contains NaN or Inf");
        proto = h_proto_;
        proto_c = pc_;
        proto_h = ph_;
        proto_w = pw_;
    }
    auto dets = nms_and_cap(candidates, cfg, lb.orig_w, lb.orig_h);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
