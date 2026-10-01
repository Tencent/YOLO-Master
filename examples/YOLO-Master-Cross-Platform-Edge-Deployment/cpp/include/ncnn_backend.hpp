// ncnn backend for YOLO-Master-EsMoE-N (CPU; Vulkan optional at build time).
#pragma once
#include "yolomaster.hpp"
#include "net.h"

namespace yolomaster {

class NcnnBackend : public Backend {
public:
    // `precision` is per-model: Auto resolves fp16-vs-fp32 from the .param fingerprint (see
    // meta::scan_ncnn_param); explicit requests the model cannot honour are downgraded with an
    // ep_note. Int8 is a property of the loaded .param (ncnn2int8 output), not a runtime switch:
    // callers select the "-int8_ncnn" sibling (meta::ncnn_int8_sibling) and pass Int8 here.
    NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads = 4,
                bool use_vulkan = false, Precision precision = Precision::Auto);
    // The Backend::forward_raw contract (preprocess + extractor + candidate decode; infer() is the
    // shared default = forward_raw + nms_and_cap). `decode = false`: the extractor still runs
    // (ncnn computes on extract) but the output is not reshaped/decoded, so infer_ms is the pure
    // kernel time and the cached candidates are cleared.
    void forward_raw(const cv::Mat& bgr, const Config& cfg, bool decode = true) override;
    const char* runtime_name() const override { return "ncnn"; }
    Precision requested_precision() const { return requested_; }
    bool fp16_active() const { return fp16_; }   // what net_.opt actually runs with
    bool int8_model() const { return int8_; }     // the loaded .param carries int8 layers

private:
    bool end2end_ = false;   // NMS-free [num_det,6] output (upstream auto-disables for ncnn, kept for robustness)
    ncnn::Net net_;
    int threads_;
    Precision requested_ = Precision::Auto;
    bool fp16_ = false;
    bool int8_ = false;
    std::string in_blob_ = "in0";
    std::string out_blob_ = "out0";
    std::string out_proto_ = "out1";   // segmentation proto (absent on detection models)
    bool has_proto_ = false;           // "out1" present among the net's output blobs
};

} // namespace yolomaster
