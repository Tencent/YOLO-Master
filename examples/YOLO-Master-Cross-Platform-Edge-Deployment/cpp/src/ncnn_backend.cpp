#include "ncnn_backend.hpp"
#include "cpu.h"   // ncnn::cpu_support_arm_asimdhp (returns 0 off-ARM)
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <cstdlib>
#include <cstdio>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

NcnnBackend::NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads,
                         bool use_vulkan, Precision precision)
    : threads_(threads), requested_(precision) {
    // ---- per-model numeric precision policy (decided before load_param; opt is load-time) ----
    // fp16 is enabled only for models whose .param carries no fp16 hazard. The emulated-router
    // mixture graphs (scripts/export_ncnn_mixture.py: 1e-9 mask nudges, 1e30 expert masks, FLT_MAX
    // clamps) are pinned fp32: those constants are unrepresentable in fp16 (1e-9 flushes to 0 under
    // ARM FZ16 and breaks the ceil() one-hot, 1e30 overflows fp16's 65504 max) and routing then
    // returns zero detections. That is runtime-unfixable (the export-side fix is a separate task).
    // Explicit requests a model cannot honour are DOWNGRADED and explained in ep_note - never a
    // silent zero-detection run. ncnn's fp16 kernels are the armv8.2 (asimdhp) path; elsewhere the
    // flags are inert, so the run is labelled fp32 rather than claiming an fp16 speedup.
    const std::string mdir = std::filesystem::path(param_path).parent_path().string();
    const meta::NcnnParamScan scan = meta::scan_ncnn_param(param_path);
    int yaml_fp16_safe = -1;   // -1 unknown, 0 false, 1 true (optional corroboration stamped by the exporter)
    {
        std::string v;
        if (meta::read_ncnn_yaml_scalar(mdir + "/metadata.yaml", "fp16_safe", v))
            yaml_fp16_safe = (v.find("rue") != std::string::npos) ? 1 : 0;
    }
    auto note = [this](const std::string& s) { if (!ep_note.empty()) ep_note += "; "; ep_note += s; };

    // fp16 is allowed when the model is clean, or when its hazards are confined to a pin set that
    // stays fp32 per layer (ncnn Layer::featmask) while the rest of the net runs fp16. A stamped
    // "fp16_safe: false" is a hard override in both cases (the exporter knows something we do not).
    // "fp16_safe: false" (stamped by export_ncnn_mixture.py) vetoes whole-net fp16 only; the mixed
    // per-layer mode has its own veto key "fp16_mixed: false" for exporters that know better.
    int yaml_fp16_mixed = -1;
    {
        std::string v;
        if (meta::read_ncnn_yaml_scalar(mdir + "/metadata.yaml", "fp16_mixed", v))
            yaml_fp16_mixed = (v.find("rue") != std::string::npos) ? 1 : 0;
    }
    const bool pinnable = scan.fp16_pinnable() && yaml_fp16_mixed != 0;
    const bool model_fp16_safe = (scan.fp16_safe() && yaml_fp16_safe != 0) || pinnable;   // any "unsafe" source wins
    if (yaml_fp16_safe == 1 && !scan.fp16_safe() && !pinnable)
        note("metadata says fp16_safe but the .param scan disagrees; trusting the scan");
    int8_ = scan.is_int8();
    if (precision == Precision::Int8 && !int8_) note("int8 requested but the .param has no int8 layers");
    if (int8_ && use_vulkan) {
        use_vulkan = false;
        note("int8 model: Vulkan disabled (no ncnn Vulkan int8 kernels)");
    }

    bool want_fp16 = false;
    switch (precision) {
        case Precision::Fp32:
            want_fp16 = false;
            break;
        case Precision::Fp16:
            want_fp16 = model_fp16_safe;
            if (!model_fp16_safe) note("fp16 requested but refused: " + scan.reason());
            break;
        default:   // Auto, Int8: derive from the model
            want_fp16 = model_fp16_safe;
            if (!model_fp16_safe) note("fp32 pinned: " + scan.reason());
            break;
    }
    const bool cpu_fp16 = ncnn::cpu_support_arm_asimdhp() != 0;
    fp16_ = want_fp16 && (use_vulkan || cpu_fp16);
    if (want_fp16 && !use_vulkan && !cpu_fp16 && precision == Precision::Fp16)
        note("fp16 requested: CPU has no fp16 arithmetic (asimdhp); running fp32");

    net_.opt.num_threads = threads;
    net_.opt.use_vulkan_compute = use_vulkan;   // GPU path (the -shared prebuilt is Vulkan-enabled)
    net_.opt.use_fp16_packed = fp16_;
    net_.opt.use_fp16_storage = fp16_;
    net_.opt.use_fp16_arithmetic = fp16_;
    net_.opt.use_bf16_storage = false;          // bf16 is never part of this policy
    net_.opt.use_int8_inference = true;         // ncnn's default, stated explicitly
    if (int8_) {
        net_.opt.use_int8_packed = true;
        net_.opt.use_int8_storage = true;       // use_int8_arithmetic stays at ncnn's default
    }
    if (net_.load_param(param_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load param " + param_path);
    // ---- per-layer fp32 pin (between load_param and load_model: featmask is read at pipeline creation) ----
    // Bits per ncnn src/net.cpp get_masked_option: 1<<0 fp16 arithmetic, 1<<1 fp16 packed+storage.
    int pinned_layers = 0;
    if (pinnable && !scan.pin_layer_idx.empty()) {
        std::vector<ncnn::Layer*>& layers = net_.mutable_layers();
        bool aligned = true;
        for (size_t k = 0; k < scan.pin_layer_idx.size(); ++k) {
            const int i = scan.pin_layer_idx[k];
            if (i < 0 || i >= static_cast<int>(layers.size()) || !layers[i] || layers[i]->name != scan.pin_layer_names[k]) {
                aligned = false; break;
            }
        }
        if (aligned) {
            if (fp16_) {
                for (int i : scan.pin_layer_idx) layers[i]->featmask |= (1 << 0) | (1 << 1);
                pinned_layers = static_cast<int>(scan.pin_layer_idx.size());
                note(std::to_string(pinned_layers) + " router layers pinned fp32, rest fp16");
            }
        } else {
            // the scan and ncnn disagree about layer order: never guess, run the whole net fp32
            fp16_ = false;
            net_.opt.use_fp16_packed = net_.opt.use_fp16_storage = net_.opt.use_fp16_arithmetic = false;
            note("fp32 pinned: per-layer pin set did not align with the loaded net (" + scan.reason() + ")");
        }
    }
    if (std::getenv("YOLOMASTER_NCNN_VERBOSE") && !scan.pin_layer_idx.empty()) {
        fprintf(stderr, "[ncnn] fp32 pin set (%zu of %d layers, pinnable=%d, applied=%d):", scan.pin_layer_idx.size(),
                scan.layers, pinnable ? 1 : 0, pinned_layers ? 1 : 0);
        for (const auto& n : scan.pin_layer_names) fprintf(stderr, " %s", n.c_str());
        fprintf(stderr, "\n");
    }
    active_ep = use_vulkan ? (fp16_ ? (pinned_layers ? "ncnn-Vulkan-mixed" : "ncnn-Vulkan") : "ncnn-Vulkan-fp32")
                           : std::string("ncnn-CPU-") + (int8_ ? "int8+" : "") +
                             (fp16_ ? (pinned_layers ? "fp16-mixed" : "fp16") : "fp32");
    if (net_.load_model(bin_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load bin " + bin_path);
    for (const char* n : net_.output_names()) if (out_proto_ == n) has_proto_ = true;

    // auto-read ultralytics metadata sidecar (class names + imgsz)
    const std::string dir = std::filesystem::path(param_path).parent_path().string();
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(dir + "/metadata.yaml", nm, mi, end2end_)) { meta_names = nm; meta_imgsz = mi; }
    // YOLO-Master ncnn graphs bake the attention token counts at the training size,
    // so the input size is effectively fixed.
    fixed_imgsz = meta_imgsz;
}

void NcnnBackend::forward_raw(const cv::Mat& bgr, const Config& cfg, bool decode) {
    // ---- preprocess: letterbox -> ncnn RGB /255 ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = preprocess(bgr, cfg.imgsz, cfg.stretch, lb);
    ncnn::Mat in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          padded.cols, padded.rows);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean, norm);
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    ncnn::Extractor ex = net_.create_extractor();  // uses net_.opt.num_threads set in ctor
    ex.input(in_blob_.c_str(), in);
    ncnn::Mat out, pm;
    ex.extract(out_blob_.c_str(), out);
    if (has_proto_) ex.extract(out_proto_.c_str(), pm);   // proto (seg models only; avoids ncnn's per-frame "find_blob_index_by_name out1 failed" log on detection models)
    infer_ms = ms_since(t1);

    // Every path below (re)fills the cached raw state; a bench-only forward leaves it empty so
    // stale candidates can never be mistaken for this frame's.
    candidates.clear();
    cand_orig_w = lb.orig_w; cand_orig_h = lb.orig_h; cand_lb = lb;
    proto.clear(); proto_c = proto_h = proto_w = 0;
    if (!decode) { post_ms = 0; return; }

    // ---- reshape to channel-major [feat_dim x num_anchors] then decode ----
    // feat << anchors always (e.g. 14/116 vs 8400), so the smaller axis is the feature dim.
    auto t2 = clk::now();
    // end2end [num_det,6] before the smaller-axis heuristic (which would mangle it)
    if (end2end_ || (out.w == 6 && out.h >= 32)) {
        std::vector<float> rows(static_cast<size_t>(out.h) * 6);
        for (int i = 0; i < out.h; ++i)
            std::memcpy(rows.data() + static_cast<size_t>(i) * 6, out.row(i), 6 * sizeof(float));
        candidates = decode_end2end(rows.data(), out.h, cfg, lb);
        post_ms = ms_since(t2);
        return;
    }
    int feat_dim, num_anchors;
    std::vector<float> buf;
    if (out.h <= out.w) {                      // rows = features (expected, channel-major)
        feat_dim = out.h; num_anchors = out.w;
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int f = 0; f < feat_dim; ++f)
            std::memcpy(buf.data() + static_cast<size_t>(f) * num_anchors,
                        out.row(f), num_anchors * sizeof(float));
    } else {                                   // rows = anchors -> transpose
        feat_dim = out.w; num_anchors = out.h;
        buf.resize(static_cast<size_t>(feat_dim) * num_anchors);
        for (int a = 0; a < num_anchors; ++a) {
            const float* r = out.row(a);
            for (int f = 0; f < feat_dim; ++f)
                buf[static_cast<size_t>(f) * num_anchors + a] = r[f];
        }
    }
    candidates = decode_candidates(buf.data(), feat_dim, num_anchors, cfg, lb);
    if (!pm.empty()) {                         // segmentation proto [c=nm, h=mh, w=mw]
        proto_c = pm.c; proto_h = pm.h; proto_w = pm.w;
        const size_t plane = static_cast<size_t>(proto_h) * proto_w;
        proto.resize(static_cast<size_t>(proto_c) * plane);
        for (int c = 0; c < proto_c; ++c)
            std::memcpy(proto.data() + c * plane, pm.channel(c), plane * sizeof(float));
    }
    post_ms = ms_since(t2);
}

} // namespace yolomaster
