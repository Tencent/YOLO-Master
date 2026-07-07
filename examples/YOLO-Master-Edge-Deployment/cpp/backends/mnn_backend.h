#pragma once

#include "backend.h"

class MnnBackend final : public Backend {
public:
    void load(const std::string& model_path) override;
    Tensor infer(const Tensor& input) override;
    std::string name() const override;

private:
    std::string model_path_;
};

