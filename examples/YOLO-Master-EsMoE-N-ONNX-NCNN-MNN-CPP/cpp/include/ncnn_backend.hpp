// ncnn backend for YOLO-Master-EsMoE-N (CPU; Vulkan optional at build time).
#pragma once
#include "yolomaster.hpp"
#include "net.h"

namespace yolomaster {

class NcnnBackend : public Backend {
public:
    NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads = 4);
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    ncnn::Net net_;
    int threads_;
    std::string in_blob_ = "in0";
    std::string out_blob_ = "out0";
};

} // namespace yolomaster
