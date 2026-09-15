// ncnn backend for YOLO-Master-EsMoE-N (CPU; Vulkan optional at build time).
#pragma once
#include "yolomaster.hpp"
#include "net.h"

namespace yolomaster {

class NcnnBackend : public Backend {
public:
    NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads = 4,
                bool use_vulkan = false);
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    ncnn::Net net_;
    int threads_;
    // Defaults preserve compatibility with older pnnx exports. New exports
    // write these names to metadata.yaml and override them at construction.
    std::string in_blob_ = "in0";
    std::string out_blob_ = "out0";
    std::string out_proto_ = "out1";   // segmentation proto (absent on detection models)
    bool proto_required_ = false;
};

} // namespace yolomaster
