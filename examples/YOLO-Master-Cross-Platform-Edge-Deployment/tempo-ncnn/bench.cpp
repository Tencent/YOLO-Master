// Standalone ncnn CPU latency bench for the project03 Orin test.
// Deliberately independent of the main yolomaster runner: no shared headers,
// no metadata, no postprocess - pure net->forward latency, median/p90.
//
//   ./ncnn_tempo <param> <bin> <threads> [iters=100] [fp16=1]
//
// fp16=1 enables fp16 packed/storage/arithmetic (ncnn's ARM fast path);
// fp16=0 forces fp32 kernels. INT8 models quantize per-layer regardless of
// the fp16 flag (unquantized layers then run in the chosen float mode).
#include "net.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <param> <bin> <threads> [iters=100] [fp16=1]\n", argv[0]);
        return 1;
    }
    const char* param = argv[1];
    const char* bin = argv[2];
    const int threads = atoi(argv[3]);
    const int iters = argc > 4 ? atoi(argv[4]) : 100;
    const int fp16 = argc > 5 ? atoi(argv[5]) : 1;

    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_vulkan_compute = false;          // CPU test
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = fp16;
    if (net.load_param(param) || net.load_model(bin)) {
        fprintf(stderr, "failed to load %s / %s\n", param, bin);
        return 1;
    }

    ncnn::Mat in(640, 640, 3);
    in.fill(0.5f);

    ncnn::Mat out;
    for (int i = 0; i < 10; i++) {               // warmup
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ex.extract("out0", out);
    }

    std::vector<double> ms(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ex.extract("out0", out);
        ms[i] = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0;                              // output checksum: precision sanity
    for (int i = 0; i < std::min(1000, (int)(out.total())); i++) sum += out[i];
    printf("[tempo] %s threads=%d fp16=%d  median %.2fms  p90 %.2fms  "
           "out[%dx%d] checksum %.3f\n",
           param, threads, fp16, ms[iters / 2], ms[(int)(iters * 0.9)],
           out.w, out.h, sum);
    return 0;
}
