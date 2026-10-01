// CUDA preprocessing parity: for every image in a directory, preprocess_nchw (CPU reference) vs
// preprocess_nchw_cuda, max |diff| per image; exit 1 when any exceeds 1/255 + 1e-6.
//   preproc_parity <images_dir> [imgsz=640] [--stretch]
#include "yolomaster.hpp"
#include "cuda_preproc.hpp"
#include "stb_image.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace yolomaster;
namespace fs = std::filesystem;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); return 2; } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: preproc_parity <images_dir> [imgsz] [--stretch]\n"); return 2; }
    const int imgsz = argc > 2 && argv[2][0] != '-' ? std::atoi(argv[2]) : 640;
    bool stretch = false;
    for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--stretch") stretch = true;
    std::vector<std::string> imgs;
    for (const auto& e : fs::directory_iterator(argv[1]))
        if (e.is_regular_file() && (e.path().extension() == ".jpg" || e.path().extension() == ".png")) imgs.push_back(e.path().string());
    std::sort(imgs.begin(), imgs.end());
    const size_t n = static_cast<size_t>(3) * imgsz * imgsz;
    float* d_out = nullptr; uint8_t* d_raw = nullptr; size_t d_raw_cap = 0;
    cuda::PreprocParams* d_pp = nullptr;
    CK(cudaMalloc(reinterpret_cast<void**>(&d_out), n * sizeof(float)));
    CK(cudaMalloc(reinterpret_cast<void**>(&d_pp), sizeof(cuda::PreprocParams)));
    std::vector<float> ref(n), got(n);
    double worst = 0; int bad = 0;
    for (const auto& p : imgs) {
        int w, h, c;
        unsigned char* px = stbi_load(p.c_str(), &w, &h, &c, 3);
        if (!px) { std::fprintf(stderr, "unreadable %s\n", p.c_str()); continue; }
        cv::Mat rgb(h, w, CV_8UC3, px), bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        stbi_image_free(px);
        LetterboxInfo lb;
        preprocess_nchw(bgr, imgsz, stretch, ref.data(), lb);
        cuda::PreprocParams pp;
        int ow = 0, oh = 0;
        letterbox_params(bgr.cols, bgr.rows, imgsz, stretch, lb, ow, oh);
        pp.src_w = bgr.cols; pp.src_h = bgr.rows; pp.src_stride = static_cast<int>(bgr.step); pp.imgsz = imgsz;
        pp.fx = static_cast<float>(bgr.cols) / ow; pp.fy = static_cast<float>(bgr.rows) / oh;
        pp.pad_x = lb.pad_x; pp.pad_y = lb.pad_y; pp.out_w = ow; pp.out_h = oh;
        const size_t raw_bytes = static_cast<size_t>(bgr.step) * bgr.rows;
        if (raw_bytes > d_raw_cap) { if (d_raw) cudaFree(d_raw); CK(cudaMalloc(reinterpret_cast<void**>(&d_raw), raw_bytes)); d_raw_cap = raw_bytes; }
        CK(cudaMemcpy(d_raw, bgr.data, raw_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pp, &pp, sizeof pp, cudaMemcpyHostToDevice));
        cuda::preprocess_nchw_cuda(d_raw, d_pp, d_out, imgsz, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        double m = 0;
        for (size_t i = 0; i < n; ++i) m = std::max(m, static_cast<double>(std::fabs(ref[i] - got[i])));
        worst = std::max(worst, m);
        const bool ok = m <= 1.0 / 255.0 + 1e-6;
        if (!ok) ++bad;
        std::printf("%-40s %dx%d  max|diff|=%.6f %s\n", fs::path(p).filename().string().c_str(), w, h, m, ok ? "" : "FAIL");
    }
    std::printf("images=%zu worst=%.6f (limit %.6f) failures=%d\n", imgs.size(), worst, 1.0 / 255.0, bad);
    cudaFree(d_out); if (d_raw) cudaFree(d_raw); cudaFree(d_pp);
    return bad ? 1 : 0;
}
