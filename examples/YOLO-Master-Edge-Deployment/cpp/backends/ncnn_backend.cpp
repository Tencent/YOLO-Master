#include "ncnn_backend.h"

#include <stdexcept>

void NcnnBackend::load(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::invalid_argument("NCNN model directory is empty");
    }
    model_path_ = model_path;
}

Tensor NcnnBackend::infer(const Tensor& input) {
    // Placeholder until ncnn::Net loading/inference is wired in.
    Tensor output;
    output.shape = {1, 84, 1};
    output.data.assign(84, input.data.empty() ? 0.0f : input.data.front());
    return output;
}

std::string NcnnBackend::name() const {
    return "ncnn";
}

