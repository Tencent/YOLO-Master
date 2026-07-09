#include "backend_factory.h"

#include <stdexcept>

#include "mnn_backend.h"
#include "ncnn_backend.h"
#include "onnx_backend.h"

std::unique_ptr<Backend> create_backend(const std::string& backend) {
    if (backend == "onnx") {
        return std::unique_ptr<Backend>(new OnnxBackend());
    }
    if (backend == "ncnn") {
        return std::unique_ptr<Backend>(new NcnnBackend());
    }
    if (backend == "mnn") {
        return std::unique_ptr<Backend>(new MnnBackend());
    }
    throw std::invalid_argument("unsupported backend: " + backend);
}
