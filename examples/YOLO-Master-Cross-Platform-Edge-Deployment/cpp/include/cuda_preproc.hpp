// CUDA twin of preprocess_nchw(): letterbox (or stretch) + BGR -> RGB + /255 + NCHW in one kernel,
// reading the raw uint8 frame from device memory and writing straight into the backend's input
// tensor. The geometry comes from letterbox_params() on the host (pure arithmetic, no pixels) and
// is passed through a device-side struct so the launch is CUDA-graph stable (only the struct's
// contents change per frame, never the kernel arguments).
// Contract vs the CPU path: bilinear with half-pixel centres like cv::resize INTER_LINEAR, but in
// float rather than OpenCV's fixed-point coefficients, so |diff| <= 1/255 per value is the promise
// (cpp/tests/preproc_parity.cpp checks it), not bit equality.
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace yolomaster::cuda {

struct PreprocParams {
    int src_w = 0, src_h = 0, src_stride = 0;   // stride in bytes (row pitch of the BGR8 frame)
    int imgsz = 0;
    float fx = 1.f, fy = 1.f;                   // source pixels per destination pixel (cv::resize inverse scale)
    int pad_x = 0, pad_y = 0, out_w = 0, out_h = 0;
};

// d_bgr: HxWx3 uint8 (row pitch = src_stride) in device memory; d_params: one PreprocParams in
// device memory; d_out: float[3 * imgsz * imgsz] (planar RGB). Asynchronous on `stream`.
void preprocess_nchw_cuda(const uint8_t* d_bgr, const PreprocParams* d_params, float* d_out, int imgsz,
                          cudaStream_t stream);
// Same, writing IEEE half (for fp16-input ONNX graphs); d_out_half: 2 * 3 * imgsz * imgsz bytes.
void preprocess_nchw_cuda_fp16(const uint8_t* d_bgr, const PreprocParams* d_params, void* d_out_half, int imgsz,
                               cudaStream_t stream);

} // namespace yolomaster::cuda
