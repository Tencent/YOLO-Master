#pragma once

#include "backend.h"

#ifdef WITH_MNN
#include <MNN/Interpreter.hpp>
#endif

class MnnBackend final : public Backend {
public:
    MnnBackend();
    ~MnnBackend() override;
    void load(const std::string& model_path) override;
    Tensor infer(const Tensor& input) override;
    std::string name() const override;

private:
    std::string model_path_;
#ifdef WITH_MNN
    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
#endif
};

