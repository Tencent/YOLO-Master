// YOLO-Master edge inference - shared types & ops (backend/model-agnostic).
#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace yolomaster {

struct Detection {
    int class_id = 0;
    float conf = 0.f;
    cv::Rect2f box;               // original-image pixel coords (float, sub-pixel precise)
    std::vector<float> mask_coeffs; // segmentation mask coefficients (empty for detection models)
    // Index into the RawDet pool nms_and_cap() was run on (-1 when not produced by it). Lets a
    // cached-raw consumer (the Android RawOutput) re-render masks for a chosen subset by index
    // instead of shipping mask_coeffs across the JNI boundary; the CLI ignores it.
    int cand_index = -1;
};

// Pre-NMS candidate (decoded to original-image px, unclipped). The GUI caches these after one forward
// pass and re-runs nms_and_cap() on conf/IoU changes without re-inferring ("forward once, tune cheap").
struct RawDet {
    cv::Rect2f box;
    float score = 0.f;
    int cls = 0;
    std::vector<float> mask_coeffs;
};

struct LetterboxInfo {
    float scale = 1.f;            // uniform scale (== scale_x == scale_y for letterbox)
    float scale_x = 1.f;         // per-axis scale (differs from scale_y only in stretch mode)
    float scale_y = 1.f;
    int pad_x = 0, pad_y = 0, orig_w = 0, orig_h = 0;   // pad is 0 in stretch mode
};

// NMS variant: Standard = plain per-class greedy; ClusterWeighted = greedy survivors get a
// post-hoc coordinate refinement (weighted average over their cluster; ultralytics cluster
// branch semantics). Survivor set/order/scores/classes are identical - only rects move.
enum class NmsMode { Standard, ClusterWeighted };

struct Config {
    int imgsz = 640;
    float conf_thresh = 0.25f;    // low default: VisDrone small/dense objects
    float iou_thresh  = 0.50f;
    int   max_det = 300;          // cap detections after NMS (ultralytics val default)
    bool  multi_label = false;    // true = one detection per class>conf per anchor (ultralytics val); false = argmax
    bool  stretch = false;        // preprocess: false = letterbox (aspect-preserving); true = stretch to square
    NmsMode nms_mode = NmsMode::Standard;
    float cw_sigma = 0.1f;        // CW-NMS weight falloff: w = score * exp(-(1-IoU)^2 / sigma)
    std::vector<std::string> class_names;
    int num_classes() const { return static_cast<int>(class_names.size()); }
};

// ---- numeric precision policy (ncnn; other backends ignore it) ----
// Auto derives fp16-vs-fp32 from the model itself (meta::scan_ncnn_param); explicit modes that a
// model cannot honour are DOWNGRADED and explained in Backend::ep_note, never silently zero-det.
// Int8 selects the pre-quantized "<name>-int8_ncnn" sibling directory (meta::ncnn_int8_sibling).
// The integer values are the JNI ABI (android/runtime): keep them stable.
enum class Precision { Auto = 0, Fp32 = 1, Fp16 = 2, Int8 = 3 };
inline const char* precision_name(Precision p) {
    switch (p) {
        case Precision::Fp32: return "fp32";
        case Precision::Fp16: return "fp16";
        case Precision::Int8: return "int8";
        default: return "auto";
    }
}
// case-insensitive; false on an unknown spelling
inline bool parse_precision(const std::string& s, Precision& out) {
    std::string t;
    for (char c : s) t += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    if (t == "auto") { out = Precision::Auto; return true; }
    if (t == "fp32" || t == "float32") { out = Precision::Fp32; return true; }
    if (t == "fp16" || t == "half") { out = Precision::Fp16; return true; }
    if (t == "int8") { out = Precision::Int8; return true; }
    return false;
}

const std::vector<std::string>& visdrone_classes();  // 10
const std::vector<std::string>& sku110k_classes();   // 1

// Resize to imgsz x imgsz: stretch=false letterboxes (min-scale, 114-pad, centered);
// stretch=true resizes to square ignoring aspect (per-axis scale, no pad).
cv::Mat preprocess(const cv::Mat& img, int imgsz, bool stretch, LetterboxInfo& info);
// Back-compat alias: aspect-preserving letterbox (== preprocess(..., stretch=false)).
cv::Mat letterbox(const cv::Mat& img, int imgsz, LetterboxInfo& info);
// Decode raw model output -> pre-NMS candidates (score >= cfg.conf_thresh; pass a low floor to cache).
std::vector<RawDet> decode_candidates(const float* out, int feat_dim, int num_anchors,
                                      const Config& cfg, const LetterboxInfo& lb);
// End-to-end (NMS-free) detection output decode: rows [x1,y1,x2,y2,score,cls] in
// letterboxed-input px (yolo26 lineage exports with end2end: True -> [1, num_det, 6]).
// Filters score >= cfg.conf_thresh, un-letterboxes, and yields the same RawDet pool the
// rest of the pipeline consumes (nms_and_cap is harmless on one-to-one detections and
// keeps conf/IoU/CW retuning, slicing pooling and the max_det cap working unchanged).
std::vector<RawDet> decode_end2end(const float* out, int num_det,
                                   const Config& cfg, const LetterboxInfo& lb);
// Shape heuristic for rank-3 det outputs when no metadata says so: [1, N>=32, 6].
inline bool looks_end2end(int d1, int d2) { return d2 == 6 && d1 >= 32; }
// Per-class NMS + max_det cap + clip-to-frame on cached candidates (cheap; re-run on conf/IoU change).
std::vector<Detection> nms_and_cap(const std::vector<RawDet>& cands, const Config& cfg,
                                   int orig_w, int orig_h);
// Full postprocess = decode_candidates + nms_and_cap (used by the CLI).
std::vector<Detection> decode(const float* out, int feat_dim, int num_anchors,
                              const Config& cfg, const LetterboxInfo& lb);
void draw(cv::Mat& img, const std::vector<Detection>& dets, const Config& cfg);

// Segmentation: composite per-detection masks (from proto x mask_coeffs) into an RGBA overlay
// (orig_h x orig_w, CV_8UC4, transparent where no mask), per-class tinted with soft edges.
// `lb`/`imgsz` map original px -> mask space. Empty proto -> fully transparent. mask_alpha 0..255.
cv::Mat seg_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                    int pc, int ph, int pw, const LetterboxInfo& lb, int imgsz,
                    int orig_w, int orig_h, int mask_alpha = 165);
// Sized variant: same masks rendered into an `out_h x out_w` RGBA overlay (the Live/Photo
// screens draw at display size, not image size). Box bounds are scaled by fx = out_w/orig_w
// (fy likewise) and the mask is sampled at the un-scaled position (ox/fx, oy/fy), so the
// per-box clipping and the smoothstep edge are unchanged; out == orig reproduces the overload
// above bit-for-bit. `dets` stay in original-image px.
cv::Mat seg_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                    int pc, int ph, int pw, const LetterboxInfo& lb, int imgsz,
                    int orig_w, int orig_h, int out_w, int out_h, int mask_alpha);
// The 10-color class palette (RGB 0..1, indexed cls%10) shared by draw/overlay/GUI.
const float* class_color(int class_id);   // returns pointer to 3 floats

// ---- model metadata (ultralytics embeds names/imgsz in the model) ----
namespace meta {
// parse a python-dict string "{0: 'pedestrian', 1: 'people', ...}" -> ordered names
std::vector<std::string> parse_names_dict(const std::string& s);
// parse an ultralytics ncnn metadata.yaml sidecar -> names + imgsz (false if unusable)
bool read_ncnn_yaml(const std::string& yaml_path, std::vector<std::string>& names, int& imgsz);
// same, additionally reading the `end2end:` key (v26.08 sidecars; false when absent)
bool read_ncnn_yaml(const std::string& yaml_path, std::vector<std::string>& names, int& imgsz,
                    bool& end2end);

// Static scan of an ncnn .param text (45-190 KB of ASCII, sub-millisecond) for numeric hazards.
// pnnx emits generic layer names, but the emulated MoE router (scripts/export_ncnn_mixture.py)
// leaves an exact fingerprint in the graph: literal 1e-9 mask nudges, 1e30 expert masks and
// "amax_*" Reduction layers. Both constants are unrepresentable in fp16 (1e-9 flushes to 0 under
// ARM FZ16 and breaks the ceil() one-hot; 1e30 overflows fp16's 65504 max), so such models must
// stay fp32 on CPU. Dense models carry none of them. int8 layers (ncnn2int8 output) carry a
// non-zero "8=" param on Convolution / ConvolutionDepthWise / InnerProduct.
// NOTE: fp16_flush counts benign sub-normal guards too (e.g. a "+1e-6" denominator); it is
// informational only and does not make a model unsafe - p03_v01n has three and runs fp16 fine.
struct NcnnParamScan {
    bool ok = false;         // file opened and header parsed
    int layers = 0;
    int router_amax = 0;     // Reduction layers named amax_*
    int nudge_1e9 = 0;       // "=1.000000e-9" literals (router mask nudge)
    int mask_1e30 = 0;       // "=1.000000e30" literals (router expert mask)
    int fp16_overflow = 0;   // other float literals with |v| > 65504 (e.g. FLT_MAX clamps)
    int fp16_flush = 0;      // other float literals with 0 < |v| < 6.1035e-5 (informational)
    int int8_layers = 0;     // quantized layers
    bool router_emulated() const { return nudge_1e9 > 0 || mask_1e30 > 0 || router_amax > 0; }
    bool fp16_safe() const { return ok && !router_emulated() && fp16_overflow == 0; }
    bool is_int8() const { return int8_layers > 0; }
    std::string reason() const;   // why-not-fp16, for ep_note ("" when fp16_safe())
};
NcnnParamScan scan_ncnn_param(const std::string& param_path);   // never throws; ok=false on failure
// One top-level scalar "key: value" from a metadata.yaml (false when absent).
bool read_ncnn_yaml_scalar(const std::string& yaml_path, const std::string& key, std::string& value);
// Pure string rule shared by the factory, the CLI and the JNI bridge:
//   "<name>_ncnn"      -> "<name>-int8_ncnn"   (a dir already ending in -int8_ncnn is returned as-is)
//   "x.ncnn.param"     -> "x-int8.param"       (bare pair; the .bin is derived by the caller)
inline std::string ncnn_int8_sibling(const std::string& model_path) {
    auto ends_with = [](const std::string& a, const std::string& suf) {
        return a.size() >= suf.size() && a.compare(a.size() - suf.size(), suf.size(), suf) == 0;
    };
    std::string p = model_path;
    while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    if (ends_with(p, "-int8_ncnn") || ends_with(p, "-int8.param")) return p;
    if (ends_with(p, ".ncnn.param")) return p.substr(0, p.size() - 11) + "-int8.param";
    if (ends_with(p, ".param")) return p.substr(0, p.size() - 6) + "-int8.param";
    if (ends_with(p, "_ncnn")) return p.substr(0, p.size() - 5) + "-int8_ncnn";
    return p + "-int8_ncnn";
}
}

// ---- versatile input source ----
enum class SourceKind { Image, Dir, Video, Dataset, Unknown };
SourceKind classify_source(const std::string& src);
// image list for Image/Dir/Dataset (Video is streamed separately by the caller).
// For Dataset (.yaml) it resolves the `val` split best-effort. `limit` caps count (0 = all).
std::vector<std::string> gather_images(const std::string& src, int limit);

// ---- backend interface ----
class Backend {
public:
    virtual ~Backend() = default;
    // The "forward once, tune cheap" seam, shared by every runtime the app can host (ncnn, ORT):
    // preprocess -> one forward -> candidate decode, filling candidates/cand_lb/cand_orig_*/proto*
    // and pre_ms/infer_ms/post_ms (post_ms = decode only; infer() adds the NMS time on top).
    // `decode = false` is the bench path (iOS `inferOnly`, Android nativeInferOnly): the forward
    // still runs, but the output is not decoded, so infer_ms is the pure kernel time and the
    // cached candidates are CLEARED (stale candidates must never pass for this frame's).
    // Backends that only implement infer() (MNN, TensorRT) keep the default, which throws.
    virtual void forward_raw(const cv::Mat& bgr, const Config& cfg, bool decode = true);
    // infer() == forward_raw(bgr, cfg) + nms_and_cap(candidates, ...). The default is the
    // contract every backend is expected to keep; overriding is for runtimes without a
    // forward_raw (they must then fill the cache themselves).
    virtual std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg);
    // Short runtime tag ("ncnn", "ort", ...): the runtime, not the device (that is active_ep).
    virtual const char* runtime_name() const { return "unknown"; }
    std::vector<std::string> meta_names;   // auto-read from the model (may be empty)
    int meta_imgsz = 0;                    // auto-read (0 = unknown)
    int fixed_imgsz = 0;                   // hard input constraint (0 = flexible)
    std::string active_ep = "cpu";         // execution provider actually in use
    std::string ep_note;                   // why a requested GPU/accelerator EP fell back (for the UI)
    double pre_ms = 0, infer_ms = 0, post_ms = 0;
    // "forward once, tune cheap": each infer() also stashes the pre-NMS candidates so a GUI can
    // re-run nms_and_cap(candidates,...) on conf/IoU changes without another forward pass.
    std::vector<RawDet> candidates;
    int cand_orig_w = 0, cand_orig_h = 0;
    LetterboxInfo cand_lb;              // letterbox used for `candidates` (maps orig<->mask space for seg)
    // segmentation prototype masks [proto_c * proto_h * proto_w], plane-major; empty for detection models.
    std::vector<float> proto;
    int proto_c = 0, proto_h = 0, proto_w = 0;
    bool is_seg() const { return proto_c > 0 && !proto.empty(); }
};

} // namespace yolomaster
