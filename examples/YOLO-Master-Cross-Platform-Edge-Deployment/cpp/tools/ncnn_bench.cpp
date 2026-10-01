// ncnn_bench - standalone ncnn latency bench for the mixed-INT8-vs-fp16 question (CPU) plus
// the Vulkan GPU path measured by the same protocol.
//
// Port of tempo-ncnn/bench.cpp (commit 9d0f537): deliberately independent of the
// yolomaster runner (no shared headers, no metadata, no NMS, no OpenCV) - pure
// create_extractor -> input("in0") -> extract("out0") latency. The input is a raw
// float32 CHW blob produced by scripts/make_probe_f32.py (the runtime letterbox of a
// real image), so every variant also yields a crude detection count + checksum that
// flags fp16 underflow / int8 garbage as valid=0.
//
//   ncnn_bench <model_dir | model.param model.bin> --input probe_640.f32
//              [--shape 3,640,640] [--threads 1,2,4,big] [--iters 100] [--warmup 20]
//              [--rounds 3] [--variants fp32,fp16,int8+fp32,int8+fp16[,vulkan,vulkan-fp32]]
//              [--powersave 2] [--label <device>] [--json out.json] [--conf 0.25] [--nc N]
//              [--idle-ms 2000]
//
// CPU variants (use_vulkan_compute=false, use_bf16_storage=false):
//   fp32      float model, fp16 packed/storage/arithmetic OFF
//   fp16      float model, fp16 packed/storage/arithmetic ON  (needs asimdhp; else "fp16(inert)")
//   int8+fp32 int8 sibling model, float remainder in fp32     (use_int8_packed/storage ON)
//   int8+fp16 int8 sibling model, float remainder in fp16
// GPU variants (use_vulkan_compute=true, Vulkan device 0; only in an NCNN_VULKAN build):
//   vulkan      float model, fp16 packed/storage/arithmetic ON  = what NcnnBackend calls ncnn-Vulkan
//   vulkan-fp32 float model, all fp16 flags OFF                 = ncnn-Vulkan-fp32 (router-emulated)
// An int8 model is a property of the .param (Convolution / ConvolutionDepthWise /
// InnerProduct lines carrying a non-zero `8=`); ncnn dispatches int8 and float layers in
// one graph, the float remainder runs in whatever fp16/fp32 mode the flags select. There are
// no Vulkan int8 kernels, so the GPU variants are float-only (skipped on an int8 model).
//
// Sibling rule: <name>_ncnn -> <name>-int8_ncnn (a bare x.ncnn.param -> x-int8.param). A given
// model whose .param already carries int8 layers (any -int8*_ncnn dir) is used as-is for the
// int8 variants and its float variants are skipped. Missing sibling -> int8 variants skipped.
// Router-emulated params (literal 1.000000e30 or a layer named amax_*) -> fp16 variants
// skipped: their 1e-9 / 1e30 constants flush / overflow under ARM FZ16. The same rule skips
// `vulkan` (fp16 shaders) and runs `vulkan-fp32` in its place.
//
// Protocol: per thread count, per round, CPU variants run INTERLEAVED (reload if needed,
// --warmup untimed infers, --iters timed infers, round median recorded), --idle-ms between
// variants. Reported per (variant, threads): median of round medians, min round median,
// p90 of the last round. Threads are irrelevant on the GPU: the Vulkan variants run ONCE
// per model after the CPU pass (same warmup/iters/rounds, threads=gpu) and the summary line
// carries the first inference after load separately as first_ms= (pipeline / shader compile,
// which the warmup otherwise absorbs). CPU output lines are unchanged by the GPU variants.
// Exit code is non-zero only on load / input failures.
//
// x86 builds are load-smoke only: fp16 flags are inert without asimdhp and AVX-VNNI int8
// says nothing about ARM sdot/i8mm, and third_party/ncnn-x86-* is built with NCNN_VULKAN=OFF
// (the Vulkan variants print a skip line there). Only an arm64 device yields the verdict.
#include "cpu.h"
#include "net.h"
#include "platform.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::string model;          // dir or .param
    std::string bin;            // only when a bare param+bin pair is given
    std::string input;
    std::string shape = "3,640,640";
    std::string threads = "1,2,4,big";
#if NCNN_VULKAN
    // Vulkan-enabled ncnn (Android SDK, Jetson): the GPU path is part of the default run.
    std::string variants = "fp32,fp16,int8+fp32,int8+fp16,vulkan";
#else
    std::string variants = "fp32,fp16,int8+fp32,int8+fp16";
#endif
    std::string label = "unknown";
    std::string json;
    int iters = 100;
    int warmup = 20;
    int rounds = 3;
    int powersave = 2;
    int nc = -1;                // -1: rows-4 of out0
    int idle_ms = 2000;
    float conf = 0.25f;
};

struct ModelFiles {
    std::string param, bin, display;   // display = basename used in the result line
    bool exists = false;
};

struct ParamScan {
    int int8_layers = 0;      // Convolution/ConvolutionDepthWise/InnerProduct with non-zero 8=
    int e30_literals = 0;     // lines containing 1.000000e30
    int amax_layers = 0;      // layers named amax_*
    bool ok = false;
    bool router_emulated() const { return e30_literals > 0 || amax_layers > 0; }
};

struct Caps {
    int asimdhp = 0, asimddp = 0, i8mm = 0, cpus = 0, big = 0;
};

struct Variant {
    std::string name;          // as requested: fp32 / fp16 / int8+fp32 / int8+fp16 / vulkan / vulkan-fp32
    std::string label;         // printed: may carry "(inert)"
    bool int8 = false;
    bool fp16 = false;
    bool vulkan = false;       // GPU variant: runs once per model, not per thread count
    ModelFiles files;
    ParamScan scan;
};

struct RoundResult {
    double median_ms = 0, p90_ms = 0;
    int dets = 0;
    double checksum = 0;
    bool finite = true;
    std::string out_shape;
};

struct ConfigResult {
    std::string variant, model;
    int threads = 0;
    double median_ms = 0, min_median_ms = 0, p90_ms = 0;
    int dets = 0;
    double checksum = 0;
    bool valid = false;
    std::string out_shape, param;
    std::vector<double> round_medians;
    bool vulkan = false;       // threads is printed as "gpu" and first_ms is reported
    double first_ms = 0;       // first inference after load (Vulkan pipeline compile)
    std::string gpu;           // Vulkan device name
};

void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s <model_dir | model.param model.bin> --input probe.f32 [--shape 3,640,640]\n"
            "          [--threads 1,2,4,big] [--iters 100] [--warmup 20] [--rounds 3]\n"
            "          [--variants fp32,fp16,int8+fp32,int8+fp16,vulkan,vulkan-fp32] [--powersave 2]\n"
            "          [--label dev] [--json out.json] [--conf 0.25] [--nc N] [--idle-ms 2000]\n"
            "variants: fp32 fp16 int8+fp32 int8+fp16 run on the CPU per --threads entry;\n"
            "          vulkan (fp16 shaders) and vulkan-fp32 run once per model on Vulkan device 0\n"
            "          (threads=gpu, first_ms= reported); skipped when ncnn is built without\n"
            "          Vulkan (%s) or no device is present. Default variants: %s\n",
            argv0, NCNN_VULKAN ? "not the case here" : "this build", Args().variants.c_str());
}

bool parse_args(int argc, char** argv, Args& a) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto need = [&](std::string& dst) {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); return false; }
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (s == "--input") { if (!need(a.input)) return false; }
        else if (s == "--shape") { if (!need(a.shape)) return false; }
        else if (s == "--threads") { if (!need(a.threads)) return false; }
        else if (s == "--variants") { if (!need(a.variants)) return false; }
        else if (s == "--label") { if (!need(a.label)) return false; }
        else if (s == "--json") { if (!need(a.json)) return false; }
        else if (s == "--iters") { if (!need(v)) return false; a.iters = atoi(v.c_str()); }
        else if (s == "--warmup") { if (!need(v)) return false; a.warmup = atoi(v.c_str()); }
        else if (s == "--rounds") { if (!need(v)) return false; a.rounds = atoi(v.c_str()); }
        else if (s == "--powersave") { if (!need(v)) return false; a.powersave = atoi(v.c_str()); }
        else if (s == "--nc") { if (!need(v)) return false; a.nc = atoi(v.c_str()); }
        else if (s == "--idle-ms") { if (!need(v)) return false; a.idle_ms = atoi(v.c_str()); }
        else if (s == "--conf") { if (!need(v)) return false; a.conf = (float)atof(v.c_str()); }
        else if (s == "-h" || s == "--help") { return false; }
        else if (!s.empty() && s[0] == '-') { fprintf(stderr, "unknown option %s\n", s.c_str()); return false; }
        else pos.push_back(s);
    }
    if (pos.empty() || pos.size() > 2) return false;
    a.model = pos[0];
    if (pos.size() == 2) a.bin = pos[1];
    if (a.iters < 1 || a.rounds < 1 || a.warmup < 0) { fprintf(stderr, "iters/rounds must be >= 1\n"); return false; }
    return true;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, sep)) if (!tok.empty()) out.push_back(tok);
    return out;
}

bool file_exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode); }
bool dir_exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); }

std::string strip_trailing_slash(std::string p) {
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    return p;
}
std::string basename_of(const std::string& p) {
    std::string q = strip_trailing_slash(p);
    size_t k = q.find_last_of('/');
    return k == std::string::npos ? q : q.substr(k + 1);
}
bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// The model as given: a dir (model.ncnn.param/bin) or a bare param [+ bin].
ModelFiles resolve_given(const Args& a) {
    ModelFiles m;
    if (dir_exists(a.model)) {
        std::string d = strip_trailing_slash(a.model);
        m.param = d + "/model.ncnn.param";
        m.bin = d + "/model.ncnn.bin";
        m.display = basename_of(d);
    } else {
        m.param = a.model;
        if (!a.bin.empty()) m.bin = a.bin;
        else if (ends_with(a.model, ".param")) m.bin = a.model.substr(0, a.model.size() - 6) + ".bin";
        else m.bin = a.model + ".bin";
        m.display = basename_of(a.model);
    }
    m.exists = file_exists(m.param) && file_exists(m.bin);
    return m;
}

// Sibling rule: <name>_ncnn -> <name>-int8_ncnn; -int8_ncnn stays; x.ncnn.param -> x-int8.param.
ModelFiles resolve_int8_sibling(const Args& a, const ModelFiles& given) {
    ModelFiles m;
    if (dir_exists(a.model)) {
        std::string d = strip_trailing_slash(a.model);
        std::string base = basename_of(d);
        std::string parent = d.size() > base.size() ? d.substr(0, d.size() - base.size()) : "";
        std::string sib;
        if (ends_with(base, "-int8_ncnn")) sib = base;
        else if (ends_with(base, "_ncnn")) sib = base.substr(0, base.size() - 5) + "-int8_ncnn";
        else sib = base + "-int8_ncnn";
        std::string sd = parent + sib;
        m.param = sd + "/model.ncnn.param";
        m.bin = sd + "/model.ncnn.bin";
        m.display = sib;
    } else {
        std::string p = a.model;
        if (ends_with(p, "-int8.param")) { m.param = p; m.bin = given.bin; }
        else {
            std::string stem = p;
            if (ends_with(stem, ".ncnn.param")) stem = stem.substr(0, stem.size() - 11);
            else if (ends_with(stem, ".param")) stem = stem.substr(0, stem.size() - 6);
            m.param = stem + "-int8.param";
            m.bin = stem + "-int8.bin";
        }
        m.display = basename_of(m.param);
    }
    m.exists = file_exists(m.param) && file_exists(m.bin);
    return m;
}

ParamScan scan_param(const std::string& path) {
    ParamScan s;
    std::ifstream f(path);
    if (!f) return s;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("1.000000e30") != std::string::npos) s.e30_literals++;
        std::istringstream ls(line);
        std::string type, name;
        if (!(ls >> type >> name)) continue;
        if (name.rfind("amax_", 0) == 0) s.amax_layers++;
        if (type == "Convolution" || type == "ConvolutionDepthWise" || type == "InnerProduct") {
            std::string tok;
            while (ls >> tok) {
                if (tok.rfind("8=", 0) == 0 && atoi(tok.c_str() + 2) != 0) { s.int8_layers++; break; }
            }
        }
    }
    s.ok = true;
    return s;
}

bool load_input(const Args& a, ncnn::Mat& in, int& c, int& h, int& w, std::string& note) {
    auto dims = split(a.shape, ',');
    if (dims.size() != 3) { fprintf(stderr, "--shape must be C,H,W\n"); return false; }
    c = atoi(dims[0].c_str()); h = atoi(dims[1].c_str()); w = atoi(dims[2].c_str());
    if (c <= 0 || h <= 0 || w <= 0) { fprintf(stderr, "bad --shape %s\n", a.shape.c_str()); return false; }
    in.create(w, h, c);
    if (in.empty()) { fprintf(stderr, "input alloc failed\n"); return false; }
    const size_t plane = (size_t)w * h;
    if (a.input.empty()) {
        in.fill(0.5f);
        note = "synthetic(0.5)";
        fprintf(stderr, "[ncnn_bench] WARNING: no --input; constant 0.5 blob, det counts are meaningless\n");
        return true;
    }
    std::ifstream f(a.input, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open --input %s\n", a.input.c_str()); return false; }
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    const size_t want = plane * c * sizeof(float);
    if (bytes != want) {
        fprintf(stderr, "--input %s has %zu bytes, expected %zu for shape %s (float32 CHW)\n",
                a.input.c_str(), bytes, want, a.shape.c_str());
        return false;
    }
    std::vector<float> buf(plane * c);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)want);
    for (int ch = 0; ch < c; ch++) memcpy(in.channel(ch), buf.data() + plane * ch, plane * sizeof(float));
    note = a.input;
    return true;
}

// Crude det count on out0: 2-D (h rows, w anchors) -> rows 4..4+nc-1 are class scores; a column
// counts if its max class score > conf. 3-D: channel 0 is taken as the 2-D map.
void score_output(const ncnn::Mat& out, float conf, int nc_override, RoundResult& r) {
    r.finite = true;
    r.checksum = 0;
    r.dets = 0;
    {
        std::ostringstream os;
        if (out.dims == 1) os << out.w;
        else if (out.dims == 2) os << out.h << "x" << out.w;
        else if (out.dims == 3) os << out.c << "x" << out.h << "x" << out.w;
        else os << out.c << "x" << out.d << "x" << out.h << "x" << out.w;
        r.out_shape = os.str();
    }
    if (out.empty()) { r.finite = false; return; }
    const size_t per_ch = (size_t)out.w * out.h * out.d;
    for (int ch = 0; ch < out.c; ch++) {
        const float* p = out.channel(ch);
        for (size_t i = 0; i < per_ch; i++) if (!std::isfinite(p[i])) { r.finite = false; break; }
        if (!r.finite) break;
    }
    const float* flat = out.channel(0);
    const size_t n0 = std::min<size_t>(1000, per_ch);
    for (size_t i = 0; i < n0; i++) r.checksum += flat[i];
    if (out.dims < 2) return;
    ncnn::Mat m2 = out.dims == 2 ? out : out.channel(0);
    if (out.dims == 3 && out.d > 1) return;   // 4-D: no crude decode defined
    const int rows = m2.h, cols = m2.w;
    if (rows <= 4) return;
    const int nc = (nc_override > 0) ? std::min(nc_override, rows - 4) : rows - 4;
    for (int col = 0; col < cols; col++) {
        float best = -1e30f;
        for (int rr = 4; rr < 4 + nc; rr++) best = std::max(best, m2.row(rr)[col]);
        if (best > conf) r.dets++;
    }
}

double percentile_sorted(const std::vector<double>& v, double q) {
    if (v.empty()) return 0;
    size_t idx = (size_t)std::ceil(q * v.size());
    if (idx == 0) idx = 1;
    return v[std::min(v.size() - 1, idx - 1)];
}
double median_sorted(const std::vector<double>& v) {
    if (v.empty()) return 0;
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// threads <= 0 leaves ncnn's default team (CPU-side layers of a Vulkan net still use it).
std::unique_ptr<ncnn::Net> load_net(const Variant& v, int threads, bool fp16_effective) {
    auto net = std::make_unique<ncnn::Net>();
    ncnn::Option& o = net->opt;
    if (threads > 0) o.num_threads = threads;
    o.use_vulkan_compute = v.vulkan;
#if NCNN_VULKAN
    if (v.vulkan) net->set_vulkan_device(0);   // after create_gpu_instance(), before load_param
#endif
    o.use_bf16_storage = false;
    o.use_fp16_packed = fp16_effective;
    o.use_fp16_storage = fp16_effective;
    o.use_fp16_arithmetic = fp16_effective;
    o.use_int8_inference = true;          // ncnn default; int8-ness lives in the .param
    if (v.int8) {
        o.use_int8_packed = true;
        o.use_int8_storage = true;
    }
    if (net->load_param(v.files.param.c_str()) != 0) return nullptr;
    if (net->load_model(v.files.bin.c_str()) != 0) return nullptr;
    return net;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) { usage(argv[0]); return 2; }

    ModelFiles given = resolve_given(a);
    if (!given.exists) {
        fprintf(stderr, "[ncnn_bench] ERROR: model not found: %s / %s\n", given.param.c_str(), given.bin.c_str());
        return 1;
    }
    ParamScan given_scan = scan_param(given.param);
    // int8-ness is a property of the .param: a model that already carries int8 layers (any
    // -int8*_ncnn dir, e.g. -int8-aciq_ncnn) is used as-is for the int8 variants; otherwise the
    // sibling is resolved by name.
    const bool given_is_int8 = given_scan.int8_layers > 0;
    ModelFiles int8 = given_is_int8 ? given : resolve_int8_sibling(a, given);
    ParamScan int8_scan = given_is_int8 ? given_scan : (int8.exists ? scan_param(int8.param) : ParamScan());

    Caps caps;
    caps.asimdhp = ncnn::cpu_support_arm_asimdhp();
    caps.asimddp = ncnn::cpu_support_arm_asimddp();
    caps.i8mm = ncnn::cpu_support_arm_i8mm();
    caps.cpus = ncnn::get_cpu_count();
    caps.big = ncnn::get_big_cpu_count();
    const int big_threads = caps.big > 0 ? caps.big : caps.cpus;
    char capstr[96];
    snprintf(capstr, sizeof capstr, "asimdhp:%d,dp:%d,i8mm:%d", caps.asimdhp, caps.asimddp, caps.i8mm);

    printf("[ncnn_bench] caps=%s cpus=%d big=%d powersave=%d label=%s\n", capstr, caps.cpus, caps.big, a.powersave, a.label.c_str());
    if (!caps.asimdhp)
        printf("[ncnn_bench] note: no asimdhp -> fp16 flags are inert here; 'fp16' variants run fp32 kernels and are labelled fp16(inert)\n");
    if (ncnn::set_cpu_powersave(a.powersave) != 0)
        fprintf(stderr, "[ncnn_bench] WARNING: set_cpu_powersave(%d) failed (no-op off Android/ARM)\n", a.powersave);
    printf("[ncnn_bench] model=%s param=%s int8_layers=%d e30=%d amax=%d\n", given.display.c_str(), given.param.c_str(),
           given_scan.int8_layers, given_scan.e30_literals, given_scan.amax_layers);
    if (int8.exists && !given_is_int8)
        printf("[ncnn_bench] int8 sibling=%s param=%s int8_layers=%d e30=%d amax=%d\n", int8.display.c_str(), int8.param.c_str(),
               int8_scan.int8_layers, int8_scan.e30_literals, int8_scan.amax_layers);

    // Input blob.
    ncnn::Mat in;
    int C = 0, H = 0, W = 0;
    std::string input_note;
    if (!load_input(a, in, C, H, W, input_note)) return 1;
    printf("[ncnn_bench] input=%s shape=%d,%d,%d conf=%.3f\n", input_note.c_str(), C, H, W, a.conf);

    // Variant plan. The GPU instance is created once, on the first Vulkan variant, and destroyed
    // at exit (after every Net is gone).
    std::vector<Variant> plan;
    bool gpu_ready = false;
    std::string gpu_name;
    for (const std::string& vn : split(a.variants, ',')) {
        Variant v;
        v.name = vn;
        if (vn == "fp32") { v.int8 = false; v.fp16 = false; }
        else if (vn == "fp16") { v.int8 = false; v.fp16 = true; }
        else if (vn == "int8+fp32") { v.int8 = true; v.fp16 = false; }
        else if (vn == "int8+fp16") { v.int8 = true; v.fp16 = true; }
        else if (vn == "vulkan") { v.vulkan = true; v.fp16 = true; }
        else if (vn == "vulkan-fp32") { v.vulkan = true; v.fp16 = false; }
        else { printf("[ncnn_bench] skip variant=%s reason=unknown variant name\n", vn.c_str()); continue; }

        if (v.vulkan) {
#if NCNN_VULKAN
            if (!gpu_ready) {
                if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() <= 0) {
                    printf("[ncnn_bench] skip variant=%s reason=no Vulkan device (create_gpu_instance failed or get_gpu_count()==0)\n", vn.c_str());
                    continue;
                }
                gpu_ready = true;
                gpu_name = ncnn::get_gpu_info(0).device_name();
                printf("[ncnn_bench] vulkan gpu_count=%d gpu=%s\n", ncnn::get_gpu_count(), gpu_name.c_str());
            }
            // Dedupe: `vulkan` on a router-emulated model is re-issued as `vulkan-fp32` below, which
            // may also have been requested explicitly.
            bool dup = false;
            for (const Variant& p : plan) dup = dup || (p.vulkan && p.fp16 == v.fp16);
            if (dup) continue;
#else
            printf("[ncnn_bench] skip variant=%s reason=ncnn built without Vulkan\n", vn.c_str());
            continue;
#endif
        }

        if (v.int8) {
            if (!int8.exists) {
                printf("[ncnn_bench] skip variant=%s reason=int8 sibling absent (%s)\n", vn.c_str(), int8.param.c_str());
                continue;
            }
            v.files = int8;
            v.scan = int8_scan;
            if (v.scan.int8_layers == 0)
                printf("[ncnn_bench] WARNING: variant=%s model %s carries no int8 layers (no non-zero 8= on conv/fc)\n", vn.c_str(), int8.display.c_str());
        } else {
            if (given_is_int8) {
                printf("[ncnn_bench] skip variant=%s reason=given model is int8 (%d int8 layers); pass the float dir for float variants\n",
                       vn.c_str(), given_scan.int8_layers);
                continue;
            }
            v.files = given;
            v.scan = given_scan;
        }
        if (v.fp16 && v.scan.router_emulated()) {
            printf("[ncnn_bench] skip variant=%s reason=router-emulated param (%d x 1.000000e30, %d x amax_ layers): 1e-9/1e30 constants flush/overflow under fp16, fp16 is invalid for %s%s\n",
                   vn.c_str(), v.scan.e30_literals, v.scan.amax_layers, v.files.display.c_str(),
                   v.vulkan ? "; running vulkan-fp32 instead" : "");
            if (!v.vulkan) continue;
            bool dup = false;
            for (const Variant& p : plan) dup = dup || (p.vulkan && !p.fp16);
            if (dup) continue;
            v.name = "vulkan-fp32";
            v.fp16 = false;
        }
        v.label = v.name;
        if (v.fp16 && !v.vulkan && !caps.asimdhp) v.label += "(inert)";
        plan.push_back(v);
    }
    std::vector<Variant> gpu_plan;
    {
        std::vector<Variant> cpu_plan;
        for (const Variant& v : plan) (v.vulkan ? gpu_plan : cpu_plan).push_back(v);
        plan.swap(cpu_plan);
    }
    if (plan.empty() && gpu_plan.empty()) {
        printf("[ncnn_bench] nothing to run (all variants skipped)\n");
#if NCNN_VULKAN
        if (gpu_ready) ncnn::destroy_gpu_instance();
#endif
        return 0;
    }

    // One (variant, model, threads) measurement: --warmup untimed infers then --iters timed ones.
    // first_ms (the very first inference after load) is only recorded when the caller asks.
    auto measure = [&](ncnn::Net& net, double* first_ms) -> RoundResult {
        ncnn::Mat out;
        if (first_ms) {
            auto t0 = std::chrono::steady_clock::now();
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ex.extract("out0", out);
            *first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        for (int i = 0; i < a.warmup; i++) {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ex.extract("out0", out);
        }
        std::vector<double> ms(a.iters);
        for (int i = 0; i < a.iters; i++) {
            auto t0 = std::chrono::steady_clock::now();
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ex.extract("out0", out);
            ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        std::sort(ms.begin(), ms.end());
        RoundResult rr;
        rr.median_ms = median_sorted(ms);
        rr.p90_ms = percentile_sorted(ms, 0.90);
        score_output(out, a.conf, a.nc, rr);
        return rr;
    };
    auto summarize = [&](const Variant& v, const std::vector<RoundResult>& rounds, int T) -> ConfigResult {
        ConfigResult c;
        c.variant = v.label;
        c.model = given.display;
        c.param = v.files.param;
        c.threads = T;
        c.vulkan = v.vulkan;
        std::vector<double> meds;
        for (const RoundResult& rr : rounds) meds.push_back(rr.median_ms);
        c.round_medians = meds;
        std::sort(meds.begin(), meds.end());
        c.median_ms = median_sorted(meds);
        c.min_median_ms = meds.front();
        const RoundResult& last = rounds.back();
        c.p90_ms = last.p90_ms;
        c.dets = last.dets;
        c.checksum = last.checksum;
        c.out_shape = last.out_shape;
        bool finite = true;
        for (const RoundResult& rr : rounds) finite = finite && rr.finite;
        c.valid = finite && c.dets > 0;
        return c;
    };

    // Thread list.
    std::vector<int> thread_list;
    for (const std::string& t : split(a.threads, ',')) {
        int n = (t == "big") ? big_threads : atoi(t.c_str());
        if (n <= 0) { fprintf(stderr, "bad thread count '%s'\n", t.c_str()); return 2; }
        if (std::find(thread_list.begin(), thread_list.end(), n) == thread_list.end()) thread_list.push_back(n);
    }

    std::vector<ConfigResult> results;
    for (int T : thread_list) {
        // Nets are loaded once per thread count (opt must be set before load) and reused across rounds.
        std::vector<std::unique_ptr<ncnn::Net>> nets(plan.size());
        std::vector<std::vector<RoundResult>> per_round(plan.size());
        for (int r = 0; r < a.rounds; r++) {
            for (size_t vi = 0; vi < plan.size(); vi++) {
                const Variant& v = plan[vi];
                const bool fp16_eff = v.fp16 && caps.asimdhp;
                if (!nets[vi]) {
                    nets[vi] = load_net(v, T, fp16_eff);
                    if (!nets[vi]) {
                        fprintf(stderr, "[ncnn_bench] ERROR: load failed variant=%s param=%s bin=%s\n", v.label.c_str(),
                                v.files.param.c_str(), v.files.bin.c_str());
                        return 1;
                    }
                }
                RoundResult rr = measure(*nets[vi], nullptr);
                per_round[vi].push_back(rr);
                printf("[ncnn_bench]   round=%d variant=%s threads=%d median_ms=%.3f p90_ms=%.3f dets=%d checksum=%.4f out0=%s\n",
                       r + 1, v.label.c_str(), T, rr.median_ms, rr.p90_ms, rr.dets, rr.checksum, rr.out_shape.c_str());
                fflush(stdout);
                const bool last = (r == a.rounds - 1) && (vi + 1 == plan.size());
                if (!last && a.idle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a.idle_ms));
            }
        }
        for (size_t vi = 0; vi < plan.size(); vi++) {
            ConfigResult c = summarize(plan[vi], per_round[vi], T);
            results.push_back(c);
            printf("[ncnn_bench] device=%s model=%s variant=%s threads=%d iters=%d rounds=%d median_ms=%.3f min_median_ms=%.3f p90_ms=%.3f dets=%d checksum=%.4f valid=%d caps=%s big=%d\n",
                   a.label.c_str(), c.model.c_str(), c.variant.c_str(), T, a.iters, a.rounds, c.median_ms, c.min_median_ms, c.p90_ms,
                   c.dets, c.checksum, c.valid ? 1 : 0, capstr, caps.big);
            fflush(stdout);
        }
    }

    // GPU variants: once per model (thread count is meaningless on Vulkan), same rounds protocol,
    // nets loaded once; the first inference after load (pipeline / shader compile) is timed on
    // its own and reported as first_ms= on the summary line, before the warmup absorbs it.
    if (!gpu_plan.empty()) {
        std::vector<std::unique_ptr<ncnn::Net>> nets(gpu_plan.size());
        std::vector<std::vector<RoundResult>> per_round(gpu_plan.size());
        std::vector<double> first_ms(gpu_plan.size(), 0.0);
        if (!plan.empty() && a.idle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a.idle_ms));
        for (int r = 0; r < a.rounds; r++) {
            for (size_t vi = 0; vi < gpu_plan.size(); vi++) {
                const Variant& v = gpu_plan[vi];
                const bool fresh = !nets[vi];
                if (fresh) {
                    nets[vi] = load_net(v, 0, v.fp16);
                    if (!nets[vi]) {
                        fprintf(stderr, "[ncnn_bench] ERROR: load failed variant=%s param=%s bin=%s\n", v.label.c_str(),
                                v.files.param.c_str(), v.files.bin.c_str());
                        nets.clear();
#if NCNN_VULKAN
                        ncnn::destroy_gpu_instance();
#endif
                        return 1;
                    }
                }
                RoundResult rr = measure(*nets[vi], fresh ? &first_ms[vi] : nullptr);
                per_round[vi].push_back(rr);
                printf("[ncnn_bench]   round=%d variant=%s threads=gpu median_ms=%.3f p90_ms=%.3f dets=%d checksum=%.4f out0=%s\n",
                       r + 1, v.label.c_str(), rr.median_ms, rr.p90_ms, rr.dets, rr.checksum, rr.out_shape.c_str());
                fflush(stdout);
                const bool last = (r == a.rounds - 1) && (vi + 1 == gpu_plan.size());
                if (!last && a.idle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a.idle_ms));
            }
        }
        for (size_t vi = 0; vi < gpu_plan.size(); vi++) {
            ConfigResult c = summarize(gpu_plan[vi], per_round[vi], 0);
            c.first_ms = first_ms[vi];
            c.gpu = gpu_name;
            results.push_back(c);
            printf("[ncnn_bench] device=%s model=%s variant=%s threads=gpu iters=%d rounds=%d median_ms=%.3f min_median_ms=%.3f p90_ms=%.3f dets=%d checksum=%.4f valid=%d first_ms=%.3f gpu=%s caps=%s big=%d\n",
                   a.label.c_str(), c.model.c_str(), c.variant.c_str(), a.iters, a.rounds, c.median_ms, c.min_median_ms, c.p90_ms,
                   c.dets, c.checksum, c.valid ? 1 : 0, c.first_ms, c.gpu.c_str(), capstr, caps.big);
            fflush(stdout);
        }
        nets.clear();   // Vulkan nets must be gone before destroy_gpu_instance()
    }
#if NCNN_VULKAN
    if (gpu_ready) ncnn::destroy_gpu_instance();
#endif

    if (!a.json.empty()) {
        std::ofstream j(a.json);
        if (!j) { fprintf(stderr, "[ncnn_bench] WARNING: cannot write --json %s\n", a.json.c_str()); }
        else {
            j << std::setprecision(10);
            j << "{\n  \"device\": \"" << json_escape(a.label) << "\",\n"
              << "  \"model\": \"" << json_escape(given.display) << "\",\n"
              << "  \"param\": \"" << json_escape(given.param) << "\",\n"
              << "  \"int8_sibling\": " << (int8.exists ? "\"" + json_escape(int8.param) + "\"" : "null") << ",\n"
              << "  \"input\": \"" << json_escape(input_note) << "\",\n"
              << "  \"shape\": [" << C << ", " << H << ", " << W << "],\n"
              << "  \"conf\": " << a.conf << ",\n"
              << "  \"iters\": " << a.iters << ", \"warmup\": " << a.warmup << ", \"rounds\": " << a.rounds
              << ", \"idle_ms\": " << a.idle_ms << ", \"powersave\": " << a.powersave << ",\n"
              << "  \"caps\": {\"asimdhp\": " << caps.asimdhp << ", \"asimddp\": " << caps.asimddp << ", \"i8mm\": " << caps.i8mm
              << ", \"cpus\": " << caps.cpus << ", \"big\": " << caps.big << "},\n"
              << "  \"speed_result\": " << (caps.asimdhp ? "true" : "false")
              << ",\n  \"results\": [\n";
            for (size_t i = 0; i < results.size(); i++) {
                const ConfigResult& c = results[i];
                j << "    {\"variant\": \"" << c.variant << "\", \"model\": \"" << json_escape(c.model) << "\", \"param\": \""
                  << json_escape(c.param) << "\", \"threads\": " << c.threads << ", \"median_ms\": " << c.median_ms
                  << ", \"min_median_ms\": " << c.min_median_ms << ", \"p90_ms\": " << c.p90_ms << ", \"dets\": " << c.dets
                  << ", \"checksum\": " << c.checksum << ", \"valid\": " << (c.valid ? "true" : "false") << ", \"out0\": \""
                  << c.out_shape << "\", \"round_medians\": [";
                for (size_t k = 0; k < c.round_medians.size(); k++) j << (k ? ", " : "") << c.round_medians[k];
                j << "]";
                // GPU rows only (CPU rows keep the exact key set of the CPU-only bench).
                if (c.vulkan) j << ", \"first_ms\": " << c.first_ms << ", \"gpu\": \"" << json_escape(c.gpu) << "\"";
                j << "}" << (i + 1 < results.size() ? "," : "") << "\n";
            }
            j << "  ]\n}\n";
            printf("[ncnn_bench] json=%s\n", a.json.c_str());
        }
    }
    return 0;
}
