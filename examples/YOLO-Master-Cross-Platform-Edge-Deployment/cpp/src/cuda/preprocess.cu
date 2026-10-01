#include "cuda_preproc.hpp"
#include <cuda_fp16.h>

namespace yolomaster::cuda {

namespace {

// One thread per destination pixel; writes the three planes. Inside the resized rectangle the
// source coordinate is (dst + 0.5) * f - 0.5 clamped at 0 (cv::resize INTER_LINEAR), outside it is
// the 114 letterbox gray.
template <typename T>
__global__ void k_preprocess(const uint8_t* __restrict__ src, const PreprocParams* __restrict__ pp,
                             T* __restrict__ dst, int imgsz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgsz || y >= imgsz) return;
    const PreprocParams p = *pp;
    const int hw = imgsz * imgsz;
    const int idx = y * imgsz + x;
    float r, g, b;
    const int rx = x - p.pad_x, ry = y - p.pad_y;
    if (rx < 0 || ry < 0 || rx >= p.out_w || ry >= p.out_h) {
        r = g = b = 114.f / 255.f;
    } else {
        float sx = (rx + 0.5f) * p.fx - 0.5f;
        float sy = (ry + 0.5f) * p.fy - 0.5f;
        if (sx < 0.f) sx = 0.f;
        if (sy < 0.f) sy = 0.f;
        int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
        if (x0 > p.src_w - 1) x0 = p.src_w - 1;
        if (y0 > p.src_h - 1) y0 = p.src_h - 1;
        const int x1 = x0 + 1 < p.src_w ? x0 + 1 : x0;
        const int y1 = y0 + 1 < p.src_h ? y0 + 1 : y0;
        const float wx = sx - x0, wy = sy - y0;
        const uint8_t* r0 = src + static_cast<size_t>(y0) * p.src_stride;
        const uint8_t* r1 = src + static_cast<size_t>(y1) * p.src_stride;
        const uint8_t* p00 = r0 + x0 * 3; const uint8_t* p01 = r0 + x1 * 3;
        const uint8_t* p10 = r1 + x0 * 3; const uint8_t* p11 = r1 + x1 * 3;
        const float w00 = (1.f - wx) * (1.f - wy), w01 = wx * (1.f - wy), w10 = (1.f - wx) * wy, w11 = wx * wy;
        // source is BGR
        b = (p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11) * (1.f / 255.f);
        g = (p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11) * (1.f / 255.f);
        r = (p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11) * (1.f / 255.f);
    }
    dst[idx] = static_cast<T>(r);
    dst[hw + idx] = static_cast<T>(g);
    dst[2 * hw + idx] = static_cast<T>(b);
}

__global__ void k_preprocess_half(const uint8_t* __restrict__ src, const PreprocParams* __restrict__ pp,
                                  __half* __restrict__ dst, int imgsz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgsz || y >= imgsz) return;
    const PreprocParams p = *pp;
    const int hw = imgsz * imgsz;
    const int idx = y * imgsz + x;
    float r, g, b;
    const int rx = x - p.pad_x, ry = y - p.pad_y;
    if (rx < 0 || ry < 0 || rx >= p.out_w || ry >= p.out_h) {
        r = g = b = 114.f / 255.f;
    } else {
        float sx = (rx + 0.5f) * p.fx - 0.5f;
        float sy = (ry + 0.5f) * p.fy - 0.5f;
        if (sx < 0.f) sx = 0.f;
        if (sy < 0.f) sy = 0.f;
        int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
        if (x0 > p.src_w - 1) x0 = p.src_w - 1;
        if (y0 > p.src_h - 1) y0 = p.src_h - 1;
        const int x1 = x0 + 1 < p.src_w ? x0 + 1 : x0;
        const int y1 = y0 + 1 < p.src_h ? y0 + 1 : y0;
        const float wx = sx - x0, wy = sy - y0;
        const uint8_t* r0 = src + static_cast<size_t>(y0) * p.src_stride;
        const uint8_t* r1 = src + static_cast<size_t>(y1) * p.src_stride;
        const uint8_t* p00 = r0 + x0 * 3; const uint8_t* p01 = r0 + x1 * 3;
        const uint8_t* p10 = r1 + x0 * 3; const uint8_t* p11 = r1 + x1 * 3;
        const float w00 = (1.f - wx) * (1.f - wy), w01 = wx * (1.f - wy), w10 = (1.f - wx) * wy, w11 = wx * wy;
        b = (p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11) * (1.f / 255.f);
        g = (p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11) * (1.f / 255.f);
        r = (p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11) * (1.f / 255.f);
    }
    dst[idx] = __float2half(r);
    dst[hw + idx] = __float2half(g);
    dst[2 * hw + idx] = __float2half(b);
}

} // namespace

void preprocess_nchw_cuda(const uint8_t* d_bgr, const PreprocParams* d_params, float* d_out, int imgsz,
                          cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((imgsz + block.x - 1) / block.x, (imgsz + block.y - 1) / block.y);
    k_preprocess<float><<<grid, block, 0, stream>>>(d_bgr, d_params, d_out, imgsz);
}

void preprocess_nchw_cuda_fp16(const uint8_t* d_bgr, const PreprocParams* d_params, void* d_out_half, int imgsz,
                               cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((imgsz + block.x - 1) / block.x, (imgsz + block.y - 1) / block.y);
    k_preprocess_half<<<grid, block, 0, stream>>>(d_bgr, d_params, static_cast<__half*>(d_out_half), imgsz);
}

} // namespace yolomaster::cuda
