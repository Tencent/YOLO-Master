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
};

struct LetterboxInfo {
    float scale = 1.f;
    int pad_x = 0, pad_y = 0, orig_w = 0, orig_h = 0;
};

struct Config {
    int imgsz = 640;
    float conf_thresh = 0.25f;    // low default: VisDrone small/dense objects
    float iou_thresh  = 0.50f;
    int   max_det = 300;          // cap detections after NMS (ultralytics val default)
    bool  multi_label = false;    // true = one detection per class>conf per anchor (ultralytics val); false = argmax
    std::vector<std::string> class_names;
    int num_classes() const { return static_cast<int>(class_names.size()); }
};

const std::vector<std::string>& visdrone_classes();  // 10
const std::vector<std::string>& sku110k_classes();   // 1

cv::Mat letterbox(const cv::Mat& img, int imgsz, LetterboxInfo& info);
std::vector<Detection> decode(const float* out, int feat_dim, int num_anchors,
                              const Config& cfg, const LetterboxInfo& lb);
void draw(cv::Mat& img, const std::vector<Detection>& dets, const Config& cfg);

// ---- model metadata (ultralytics embeds names/imgsz in the model) ----
namespace meta {
// parse a python-dict string "{0: 'pedestrian', 1: 'people', ...}" -> ordered names
std::vector<std::string> parse_names_dict(const std::string& s);
// parse an ultralytics ncnn metadata.yaml sidecar -> names + imgsz (false if unusable)
bool read_ncnn_yaml(const std::string& yaml_path, std::vector<std::string>& names, int& imgsz);
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
    virtual std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) = 0;
    std::vector<std::string> meta_names;   // auto-read from the model (may be empty)
    int meta_imgsz = 0;                    // auto-read (0 = unknown)
    int fixed_imgsz = 0;                   // hard input constraint (0 = flexible)
    std::string active_ep = "cpu";         // execution provider actually in use
    double pre_ms = 0, infer_ms = 0, post_ms = 0;
};

} // namespace yolomaster
