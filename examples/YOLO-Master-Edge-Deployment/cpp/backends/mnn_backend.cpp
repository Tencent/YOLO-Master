#include "mnn_backend.h"

#include <numeric>
#include <stdexcept>

#ifdef WITH_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#endif

namespace {
#ifdef WITH_MNN
size_t tensor_size(const std::vector<int64_t>& shape) {
    size_t size = 1;
    for (const int64_t dim : shape) {
        size *= static_cast<size_t>(dim > 0 ? dim : 1);
    }
    return size;
}

void validate_input(const Tensor& input) {
    if (input.shape.size() != 4 || input.shape[0] != 1 || input.shape[1] != 3) {
        throw std::invalid_argument("MNN input tensor must have shape [1, 3, H, W]");
    }
    const size_t expected_size = std::accumulate(
        input.shape.begin(),
        input.shape.end(),
        static_cast<size_t>(1),
        [](size_t acc, int64_t value) { return acc * static_cast<size_t>(value); });
    if (expected_size != input.data.size()) {
        throw std::invalid_argument("MNN input tensor data size does not match shape");
    }
}
#endif
}  // namespace

MnnBackend::MnnBackend() = default;

MnnBackend::~MnnBackend() {
#ifdef WITH_MNN
    if (interpreter_) {
        delete interpreter_;
        interpreter_ = nullptr;
    }
#endif
}

void MnnBackend::load(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::invalid_argument("MNN model path is empty");
    }
    model_path_ = model_path;

#ifdef WITH_MNN
    interpreter_ = MNN::Interpreter::createFromFile(model_path_.c_str());
    if (!interpreter_) {
        throw std::runtime_error("failed to create MNN interpreter: " + model_path_);
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    session_ = interpreter_->createSession(config);
    if (!session_) {
        throw std::runtime_error("failed to create MNN session: " + model_path_);
    }

    input_tensor_ = interpreter_->getSessionInput(session_, "images");
    if (!input_tensor_) {
        input_tensor_ = interpreter_->getSessionInput(session_, nullptr);
    }
    if (!input_tensor_) {
        throw std::runtime_error("failed to get MNN input tensor from model: " + model_path_);
    }
#else
    // Stub mode keeps the benchmark harness buildable without MNN.
    // Configure WITH_MNN=ON for real model execution.
#endif
}

Tensor MnnBackend::infer(const Tensor& input) {
#ifdef WITH_MNN
    validate_input(input);
    if (!interpreter_ || !session_ || !input_tensor_) {
        throw std::runtime_error("MNN backend used before load()");
    }

    std::vector<int> dims(input.shape.begin(), input.shape.end());
    interpreter_->resizeTensor(input_tensor_, dims);
    interpreter_->resizeSession(session_);

    auto* tmp_input = MNN::Tensor::create(
        dims,
        halide_type_of<float>(),
        const_cast<float*>(input.data.data()),
        MNN::Tensor::CAFFE);
    if (!tmp_input) {
        throw std::runtime_error("failed to create MNN input tensor");
    }
    input_tensor_->copyFromHostTensor(tmp_input);
    delete tmp_input;

    interpreter_->runSession(session_);

    auto* output_tensor = interpreter_->getSessionOutput(session_, nullptr);
    if (!output_tensor) {
        throw std::runtime_error("failed to get MNN output tensor");
    }

    MNN::Tensor host_output(output_tensor, MNN::Tensor::CAFFE);
    output_tensor->copyToHostTensor(&host_output);

    Tensor output;
    const auto output_shape = host_output.shape();
    output.shape.assign(output_shape.begin(), output_shape.end());
    const size_t output_size = tensor_size(output.shape);
    const float* output_data = host_output.host<float>();
    if (!output_data) {
        throw std::runtime_error("MNN output tensor is empty");
    }
    output.data.assign(output_data, output_data + output_size);
    return output;
#else
    Tensor output;
    output.shape = {1, 84, 1};
    output.data.assign(84, input.data.empty() ? 0.0f : input.data.front());
    return output;
#endif
}

std::string MnnBackend::name() const {
    return "mnn";
}
