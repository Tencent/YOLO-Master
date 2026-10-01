// Minimal ncnn raw-tensor dumper for parity validation.
//
// The python ncnn wheels segfault at inference on this machine (even on known-good
// models the C++ runtime handles fine), so the mixture harness extracts raw tensors
// through this tool, linked against the SAME vendored ncnn SDK the edge runtime ships.
//
// usage: ncnn_rawdump [--fp16|--fp32] [--int8=0|1] model.param model.bin in.f32 C H W out.f32 [blob]
//   in.f32: raw float32, channel-major [C,H,W]; out.f32: raw float32 out0, channel-major.
//   Prints "dims=? c=? h=? w=? fp16=?" for the caller to reshape.
//   --fp16 enables ncnn's fp16 packed/storage/arithmetic (default --fp32, the certified path).
//   NOTE: ncnn's fp16 kernels are the armv8.2 (asimdhp) path; on x86 the flags are inert, so a
//   passing --fp16 run on x86 proves nothing about ARM. --int8=0 disables int8 inference for a
//   float-vs-int8 raw diff on a quantized model (default 1 = ncnn's default).
#include <net.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    bool fp16 = false, int8 = true;
    std::vector<char*> pos;                       // positional args after stripping the flags
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fp16") fp16 = true;
        else if (a == "--fp32") fp16 = false;
        else if (a.rfind("--int8=", 0) == 0) int8 = a.substr(7) != "0";
        else pos.push_back(argv[i]);
    }
    const int n = (int)pos.size();
    if (n != 7 && n != 8) {
        fprintf(stderr, "usage: %s [--fp16|--fp32] [--int8=0|1] param bin in.f32 C H W out.f32 [blob]\n", argv[0]);
        return 2;
    }
    const char* blob = n == 8 ? pos[7] : "out0";
    const int C = atoi(pos[3]), H = atoi(pos[4]), W = atoi(pos[5]);
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = fp16;
    net.opt.use_bf16_storage = false;
    net.opt.use_int8_inference = int8;
    if (net.load_param(pos[0]) != 0) { fprintf(stderr, "load_param failed\n"); return 3; }
    if (net.load_model(pos[1]) != 0) { fprintf(stderr, "load_model failed\n"); return 3; }

    std::vector<float> buf((size_t)C * H * W);
    FILE* f = fopen(pos[2], "rb");
    if (!f || fread(buf.data(), 4, buf.size(), f) != buf.size()) { fprintf(stderr, "input read failed\n"); return 4; }
    fclose(f);
    ncnn::Mat in(W, H, C);
    for (int c = 0; c < C; ++c)
        memcpy(in.channel(c), buf.data() + (size_t)c * H * W, (size_t)H * W * 4);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract(blob, out) != 0) { fprintf(stderr, "extract failed\n"); return 5; }
    printf("dims=%d c=%d h=%d w=%d fp16=%d int8=%d\n", out.dims, out.c, out.h, out.w, fp16 ? 1 : 0, int8 ? 1 : 0);

    FILE* g = fopen(pos[6], "wb");
    if (!g) { fprintf(stderr, "output open failed\n"); return 6; }
    const int chans = out.dims == 3 ? out.c : 1;
    for (int c = 0; c < chans; ++c)
        fwrite(out.dims == 3 ? (const float*)out.channel(c) : (const float*)out.data,
               4, (size_t)out.h * out.w, g);
    fclose(g);
    return 0;
}
