#include "mnn_backend.h"

#include <stdexcept>

void MnnBackend::load(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::invalid_argument("MNN model path is empty");
    }
    model_path_ = model_path;
}

Tensor MnnBackend::infer(const Tensor& input) {
    // Placeholder until MNN Interpreter loading/inference is wired in.
    Tensor output;
    output.shape = {1, 84, 1};
    output.data.assign(84, input.data.empty() ? 0.0f : input.data.front());
    return output;
}

std::string MnnBackend::name() const {
    return "mnn";
}

