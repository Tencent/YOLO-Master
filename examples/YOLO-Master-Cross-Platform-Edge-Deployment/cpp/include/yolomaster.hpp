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
    // Optional area-adaptive confidence floor used by the VisDrone NMS sweep.
    // A negative value disables the override; when enabled, boxes whose
    // original-image area is below `small_area` use the lower of the global
    // and small-object thresholds.  Keeping this disabled by default retains
    // generic detector behaviour while allowing Python and C++ validation runs
    // to share one explicit small-object policy.
    float small_conf_thresh = -1.f;
    float small_area = 32.f * 32.f;
    int   max_det = 300;          // cap detections after NMS (ultralytics val default)
    bool  multi_label = false;    // true = one detection per class>conf per anchor (ultralytics val); false = argmax
    bool  stretch = false;        // preprocess: false = letterbox (aspect-preserving); true = stretch to square
    NmsMode nms_mode = NmsMode::Standard;
    float cw_sigma = 0.1f;        // CW-NMS weight falloff: w = score * exp(-(1-IoU)^2 / sigma)
    std::vector<std::string> class_names;
    int num_classes() const { return static_cast<int>(class_names.size()); }
};

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
// Explicit-layout overload for backends that can disambiguate an objectness
// channel from segmentation mask coefficients using the model's prototype
// output.  This is required for ``4 + 1 + nc + nm`` heads.
std::vector<RawDet> decode_candidates(const float* out, int feat_dim, int num_anchors,
                                      const Config& cfg, const LetterboxInfo& lb,
                                      bool has_objectness);
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
// The 10-color class palette (RGB 0..1, indexed cls%10) shared by draw/overlay/GUI.
const float* class_color(int class_id);   // returns pointer to 3 floats

// ---- model metadata (ultralytics embeds names/imgsz in the model) ----
namespace meta {
// parse a python-dict string "{0: 'pedestrian', 1: 'people', ...}" -> ordered names
std::vector<std::string> parse_names_dict(const std::string& s);
// Parse an ultralytics ncnn metadata.yaml sidecar -> names + imgsz.  Optional
// blob-name outputs let the runtime follow pnnx graphs whose tensors are not
// named in0/out0/out1.  Existing callers may omit the optional pointers.
bool read_ncnn_yaml(const std::string& yaml_path, std::vector<std::string>& names, int& imgsz,
                    std::string* input_blob = nullptr, std::string* output_blob = nullptr,
                    std::string* proto_blob = nullptr);
}

// ---- versatile input source ----
enum class SourceKind { Image, Dir, Video, Dataset, List, Unknown };
SourceKind classify_source(const std::string& src);
// image list for Image/Dir/Dataset/List (Video is streamed separately by the caller).
// A .txt/.list List is newline-delimited and preserves file order; relative entries are
// resolved against the list's directory. For Dataset (.yaml) the `val` split
// accepts a scalar, an inline sequence, or a block sequence of directories,
// image-list files, or image files. `limit` caps count (0 = all).
std::vector<std::string> gather_images(const std::string& src, int limit);
// Evidence runs require a one-to-one mapping between an image and its
// per-image prediction file. Compare stems case-insensitively so a run has
// identical semantics on Windows and Linux.
bool validate_unique_stems(const std::vector<std::string>& images, std::string& error);

// ---- backend interface ----
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) = 0;
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
