#pragma once

#include "backend.h"

#ifdef WITH_NCNN
#include <memory>
#include <net.h>
#endif

class NcnnBackend final : public Backend {
public:
    NcnnBackend();
    ~NcnnBackend() override;
    void load(const std::string& model_path) override;
    Tensor infer(const Tensor& input) override;
    std::string name() const override;

private:
    std::string model_path_;
#ifdef WITH_NCNN
    std::string param_path_;
    std::string bin_path_;
    std::unique_ptr<ncnn::Net> net_;
    std::string input_name_;
    std::string output_name_;
#endif
};
