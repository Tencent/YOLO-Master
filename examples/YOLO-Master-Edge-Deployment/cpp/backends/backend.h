#pragma once

#include <string>
#include <vector>

struct Tensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual void load(const std::string& model_path) = 0;
    virtual Tensor infer(const Tensor& input) = 0;
    virtual std::string name() const = 0;
};
