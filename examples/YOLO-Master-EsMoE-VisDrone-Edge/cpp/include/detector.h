// ONNX Runtime wrapper for YOLO-Master-EsMoE edge inference.
// Returns the raw (1, 4+nc, N) output tensor; decoding/NMS live in postprocess.h.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

class OnnxDetector {
 public:
  // providers: "cpu" or "cuda"
  OnnxDetector(const std::string& model_path, const std::string& providers = "cpu", int intra_op = 4) {
    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yolo-edge");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(intra_op);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (providers == "cuda") {
#ifdef USE_CUDA
      OrtSessionOptionsAppendExecutionProvider_CUDA(opts, 0);
#else
      (void)providers;
#endif
    }
    session_ = std::make_shared<Ort::Session>(*env_,
#ifdef _WIN32
                                              std::wstring(model_path.begin(), model_path.end()).c_str(),
#else
                                              model_path.c_str(),
#endif
                                              opts);

    // Capture I/O names into std::string. The AllocatedStringPtr smart pointers below
    // own their char buffers and free them on destruction, so they are kept alive as
    // locals across the assign() calls (NOT as members — the type is non-default-ctible).
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session_->GetInputNameAllocated(0, alloc);
    auto out_name = session_->GetOutputNameAllocated(0, alloc);
    input_name_.assign(in_name.get());
    output_name_.assign(out_name.get());
  }

  // chw: 3*H*W float32 buffer. Returns raw output tensor + its shape.
  std::vector<float> run(const std::vector<float>& chw, int H, int W,
                         std::vector<int64_t>& out_shape) {
    std::array<int64_t, 4> in_shape{1, 3, H, W};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(chw.data()),
                                                    chw.size(), in_shape.data(), in_shape.size());
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    Ort::TensorTypeAndShapeInfo info = outs[0].GetTensorTypeAndShapeInfo();
    out_shape = info.GetShape();
    return std::vector<float>(outs[0].GetTensorData<float>(),
                              outs[0].GetTensorData<float>() + info.GetElementCount());
  }

 private:
  std::shared_ptr<Ort::Env> env_;
  std::shared_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
};

}  // namespace yolo
