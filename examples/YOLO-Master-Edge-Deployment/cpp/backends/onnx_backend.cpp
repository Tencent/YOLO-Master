#include "onnx_backend.h"

#include <numeric>
#include <stdexcept>

OnnxBackend::OnnxBackend()
#ifdef WITH_ONNXRUNTIME
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo_master_edge"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
}

void OnnxBackend::load(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::invalid_argument("ONNX model path is empty");
    }
    model_path_ = model_path;

#ifdef WITH_ONNXRUNTIME
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_.reset(new Ort::Session(env_, model_path.c_str(), session_options_));

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    output_name_ = output_name.get();
#else
    // Stub mode keeps the benchmark harness buildable without ONNX Runtime.
    // Configure WITH_ONNXRUNTIME=ON for real model execution.
#endif
}

Tensor OnnxBackend::infer(const Tensor& input) {
#ifdef WITH_ONNXRUNTIME
    if (!session_) {
        throw std::runtime_error("ONNX backend used before load()");
    }
    if (input.shape.empty() || input.data.empty()) {
        throw std::invalid_argument("ONNX input tensor is empty");
    }

    const size_t expected_size = std::accumulate(
        input.shape.begin(),
        input.shape.end(),
        static_cast<size_t>(1),
        [](size_t acc, int64_t value) { return acc * static_cast<size_t>(value); });
    if (expected_size != input.data.size()) {
        throw std::invalid_argument("ONNX input tensor data size does not match shape");
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input.data.data()),
        input.data.size(),
        input.shape.data(),
        input.shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("ONNX Runtime returned no tensor output");
    }

    auto shape_info = outputs[0].GetTensorTypeAndShapeInfo();
    Tensor output;
    output.shape = shape_info.GetShape();
    const size_t output_size = shape_info.GetElementCount();
    const float* output_data = outputs[0].GetTensorData<float>();
    output.data.assign(output_data, output_data + output_size);
    return output;
#else
    Tensor output;
    output.shape = {1, 84, 1};
    output.data.assign(84, input.data.empty() ? 0.0f : input.data.front());
    return output;
#endif
}

std::string OnnxBackend::name() const {
    return "onnx";
}
