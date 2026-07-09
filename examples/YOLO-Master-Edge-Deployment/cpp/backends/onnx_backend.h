#pragma once

#include "backend.h"

#ifdef WITH_ONNXRUNTIME
#include <memory>
#include <onnxruntime_cxx_api.h>
#endif

class OnnxBackend final : public Backend {
public:
    OnnxBackend();
    void load(const std::string& model_path) override;
    Tensor infer(const Tensor& input) override;
    std::string name() const override;

private:
    std::string model_path_;
#ifdef WITH_ONNXRUNTIME
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::string input_name_;
    std::string output_name_;
#endif
};
