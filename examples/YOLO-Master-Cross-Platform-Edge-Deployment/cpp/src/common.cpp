// Shared, backend/model-agnostic ops: class tables, letterbox, decode+NMS,
// drawing, model-metadata parsing, and versatile source resolution.
#include "yolomaster.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <filesystem>
#include <system_error>
#include <stdexcept>
#include <limits>
#include <utility>

namespace fs = std::filesystem;

namespace yolomaster {

const std::vector<std::string>& visdrone_classes() {
    static const std::vector<std::string> c = {
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor"};
    return c;
}
const std::vector<std::string>& sku110k_classes() {
    static const std::vector<std::string> c = {"object"};
    return c;
}

cv::Mat preprocess(const cv::Mat& img, int imgsz, bool stretch, LetterboxInfo& info) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0 || imgsz <= 0)
        throw std::invalid_argument("preprocess requires a non-empty image and positive imgsz");
    info.orig_w = img.cols;
    info.orig_h = img.rows;
    if (stretch) {
        // resize straight to imgsz x imgsz, ignoring aspect -> per-axis scale, no pad.
        info.scale_x = imgsz / static_cast<float>(img.cols);
        info.scale_y = imgsz / static_cast<float>(img.rows);
        info.scale   = info.scale_x;   // ambiguous under stretch; keep x for any legacy reader
        info.pad_x = info.pad_y = 0;
        cv::Mat out;
        cv::resize(img, out, cv::Size(imgsz, imgsz));
        return out;
    }
    // letterbox: min-scale aspect-preserving, 114-gray padded, centered.
    const float r = std::min(imgsz / static_cast<float>(img.cols),
                             imgsz / static_cast<float>(img.rows));
    const int nw = std::max(1, static_cast<int>(std::round(img.cols * r)));
    const int nh = std::max(1, static_cast<int>(std::round(img.rows * r)));
    info.scale = info.scale_x = info.scale_y = r;
    info.pad_x = (imgsz - nw) / 2;
    info.pad_y = (imgsz - nh) / 2;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat out(imgsz, imgsz, img.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, nw, nh)));
    return out;
}

cv::Mat letterbox(const cv::Mat& img, int imgsz, LetterboxInfo& info) {
    return preprocess(img, imgsz, /*stretch=*/false, info);
}

static double box_iou(const cv::Rect2d& a, const cv::Rect2d& b) {
    const double xx1 = std::max(a.x, b.x);
    const double yy1 = std::max(a.y, b.y);
    const double xx2 = std::min(a.x + a.width,  b.x + b.width);
    const double yy2 = std::min(a.y + a.height, b.y + b.height);
    const double inter = std::max(0.0, xx2 - xx1) * std::max(0.0, yy2 - yy1);
    const double uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0;
}

// greedy per-box NMS (score-descending, IoU suppression) - replaces
// cv::dnn::NMSBoxes; identical semantics (keep is returned score-descending).
static void nms_greedy(const std::vector<cv::Rect2d>& boxes, const std::vector<float>& scores,
                       float conf, float iou_thr, std::vector<int>& keep) {
    std::vector<int> order;
    order.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
        if (std::isfinite(scores[i]) && scores[i] >= conf) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    std::vector<char> dead(boxes.size(), 0);
    for (size_t m = 0; m < order.size(); ++m) {
        const int i = order[m];
        if (dead[i]) continue;
        keep.push_back(i);
        for (size_t n = m + 1; n < order.size(); ++n) {
            const int j = order[n];
            if (dead[j]) continue;
            if (box_iou(boxes[i], boxes[j]) > iou_thr) dead[j] = 1;
        }
    }
}

static float candidate_conf_threshold(const Config& cfg, double area) {
    if (cfg.small_conf_thresh >= 0.f && area < cfg.small_area)
        return std::min(cfg.conf_thresh, cfg.small_conf_thresh);
    return cfg.conf_thresh;
}

// Decode raw output -> pre-NMS candidates.  Ultralytics YOLOv8/EsMoE heads use
// ``4 + nc [+ nm]`` features, while YOLOv5-style exports include one objectness
// channel (``4 + 1 + nc``).  The latter is detected only when model metadata
// supplies a class count, avoiding an ambiguous guess for arbitrary tensors.
static std::vector<RawDet> decode_candidates_impl(const float* out, int feat_dim,
                                                  int num_anchors, const Config& cfg,
                                                  const LetterboxInfo& lb,
                                                  bool has_objectness) {
    const int nc = cfg.num_classes() > 0 ? cfg.num_classes() : (feat_dim - 4);
    const int class_offset = 4 + (has_objectness ? 1 : 0);
    const int nm = feat_dim - class_offset - nc;     // mask-coeff count (0 = detection)
    if (!out || nc <= 0 || nm < 0 || num_anchors <= 0)
        throw std::invalid_argument("invalid detection tensor dimensions");
    std::vector<RawDet> cands;
    auto make = [&](int a, int cls, float score) {
        const float cx = out[0 * num_anchors + a];
        const float cy = out[1 * num_anchors + a];
        const float w  = out[2 * num_anchors + a];
        const float h  = out[3 * num_anchors + a];
        const float x0 = (cx - 0.5f * w - lb.pad_x) / lb.scale_x;
        const float y0 = (cy - 0.5f * h - lb.pad_y) / lb.scale_y;
        if (!std::isfinite(score) || !std::isfinite(x0) || !std::isfinite(y0) ||
            !std::isfinite(w) || !std::isfinite(h) || w <= 0.f || h <= 0.f) return;
        RawDet d; d.box = cv::Rect2f(x0, y0, w / lb.scale_x, h / lb.scale_y); d.score = score; d.cls = cls;
        if (nm > 0) { d.mask_coeffs.resize(nm);
            for (int k = 0; k < nm; ++k) d.mask_coeffs[k] = out[(class_offset + nc + k) * num_anchors + a]; }
        cands.push_back(std::move(d));
    };
    for (int a = 0; a < num_anchors; ++a) {
        // Compute the box geometry before thresholding so an optional
        // small-object floor is measured in original-image pixels, matching
        // the Python MNN validator and the Issue #51 NMS sweep.
        const float cx = out[0 * num_anchors + a];
        const float cy = out[1 * num_anchors + a];
        const float w = out[2 * num_anchors + a];
        const float h = out[3 * num_anchors + a];
        const float bw = w / lb.scale_x;
        const float bh = h / lb.scale_y;
        const double area = static_cast<double>(bw) * static_cast<double>(bh);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
            !std::isfinite(h) || !std::isfinite(bw) || !std::isfinite(bh) ||
            !std::isfinite(area) || w <= 0.f || h <= 0.f || bw <= 0.f || bh <= 0.f)
            continue;
        const float threshold = candidate_conf_threshold(cfg, area);
        int best = -1; float bestv = -std::numeric_limits<float>::infinity(); bool any = false;
        const float objectness = has_objectness ? out[4 * num_anchors + a] : 1.0f;
        for (int c = 0; c < nc; ++c) {
            const float v = objectness * out[(class_offset + c) * num_anchors + a];
            if (!std::isfinite(v)) continue;
            if (v > bestv) { bestv = v; best = c; }
            if (cfg.multi_label && v >= threshold) any = true;
        }
        if (!(cfg.multi_label ? any : (bestv >= threshold))) continue;
        if (cfg.multi_label) {                       // one candidate per class >= conf
            for (int c = 0; c < nc; ++c) {
                const float v = objectness * out[(class_offset + c) * num_anchors + a];
                if (v >= threshold) make(a, c, v);
            }
        } else make(a, best, bestv);                 // single best class
    }
    return cands;
}

std::vector<RawDet> decode_candidates(const float* out, int feat_dim, int num_anchors,
                                      const Config& cfg, const LetterboxInfo& lb) {
    const int nc = cfg.num_classes();
    const bool has_objectness = nc > 0 && feat_dim == 5 + nc;
    return decode_candidates_impl(out, feat_dim, num_anchors, cfg, lb, has_objectness);
}

std::vector<RawDet> decode_candidates(const float* out, int feat_dim, int num_anchors,
                                      const Config& cfg, const LetterboxInfo& lb,
                                      bool has_objectness) {
    return decode_candidates_impl(out, feat_dim, num_anchors, cfg, lb, has_objectness);
}

// Per-class NMS (ultralytics agnostic=False via class offset) + max_det cap + clip-to-frame.
std::vector<Detection> nms_and_cap(const std::vector<RawDet>& cands, const Config& cfg,
                                   int orig_w, int orig_h) {
    std::vector<cv::Rect2d> boxes; std::vector<float> scores; std::vector<int> idx;
    boxes.reserve(cands.size()); scores.reserve(cands.size()); idx.reserve(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        const double area = static_cast<double>(cands[i].box.width) *
                            static_cast<double>(cands[i].box.height);
        if (!std::isfinite(area) || cands[i].box.width <= 0.f || cands[i].box.height <= 0.f)
            continue;
        if (cands[i].score < candidate_conf_threshold(cfg, area)) continue;
        boxes.emplace_back(cands[i].box.x, cands[i].box.y, cands[i].box.width, cands[i].box.height);
        scores.push_back(cands[i].score); idx.push_back(static_cast<int>(i));
    }
    // Match Ultralytics' max_nms guard before the quadratic suppression pass.
    // A low VisDrone confidence floor combined with multi-label decoding can
    // otherwise create tens of thousands of candidates and make NMS dominate
    // runtime.  The tie-break by original index keeps the result deterministic.
    constexpr size_t kMaxNmsCandidates = 30000;
    if (boxes.size() > kMaxNmsCandidates) {
        std::vector<int> order(boxes.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::partial_sort(
            order.begin(), order.begin() + static_cast<std::ptrdiff_t>(kMaxNmsCandidates), order.end(),
            [&](int a, int b) {
                if (scores[a] != scores[b]) return scores[a] > scores[b];
                return a < b;
            });
        order.resize(kMaxNmsCandidates);
        std::vector<cv::Rect2d> capped_boxes;
        std::vector<float> capped_scores;
        std::vector<int> capped_idx;
        capped_boxes.reserve(kMaxNmsCandidates);
        capped_scores.reserve(kMaxNmsCandidates);
        capped_idx.reserve(kMaxNmsCandidates);
        for (const int position : order) {
            capped_boxes.push_back(boxes[position]);
            capped_scores.push_back(scores[position]);
            capped_idx.push_back(idx[position]);
        }
        boxes.swap(capped_boxes);
        scores.swap(capped_scores);
        idx.swap(capped_idx);
    }
    std::vector<int> keep;
    // Per-class stratification: translate each class into its own disjoint stratum so one
    // greedy pass does agnostic=False NMS. Candidates are intentionally kept unclipped until
    // after suppression, so derive the offset from the actual finite coordinates rather than
    // from the frame dimensions. A malformed but finite export can otherwise place a box
    // beyond the old fixed margin and make two classes suppress one another.
    double max_extent = std::max(1.0, static_cast<double>(std::max(orig_w, orig_h)));
    for (const auto& box : boxes) {
        max_extent = std::max(max_extent, std::abs(box.x));
        max_extent = std::max(max_extent, std::abs(box.y));
        max_extent = std::max(max_extent, std::abs(box.x + box.width));
        max_extent = std::max(max_extent, std::abs(box.y + box.height));
    }
    // All coordinates originate as finite float values, so this multiplication remains well
    // below DBL_MAX. The extra margin makes equality at a class boundary impossible.
    const double OFF = 2.0 * max_extent + 1.0;
    std::vector<cv::Rect2d> off = boxes;
    for (size_t k = 0; k < off.size(); ++k) {
        const int cls = cands[idx[k]].cls;
        off[k].x += cls * OFF; off[k].y += cls * OFF;
    }
    // Every candidate has already been filtered with its area-specific
    // threshold. Pass the lowest active floor so NMS does not discard a
    // small-object candidate accepted by the override.
    const float nms_floor = cfg.small_conf_thresh >= 0.f
        ? std::min(cfg.conf_thresh, cfg.small_conf_thresh) : cfg.conf_thresh;
    nms_greedy(off, scores, nms_floor, cfg.iou_thresh, keep);

    // Cluster-Weighted refinement (ultralytics cluster branch): each greedy survivor's box
    // becomes the score-and-proximity weighted average of its cluster - every same-class
    // candidate overlapping it above iou_thresh, weighted w = s * exp(-(1-IoU)^2 / sigma).
    // One-shot over the ORIGINAL coords; survivor set/order/scores/classes are untouched.
    std::vector<cv::Rect2d> refined;
    if (cfg.nms_mode == NmsMode::ClusterWeighted && !keep.empty()) {
        // pool = top-3000 of the conf-filtered candidates by score (upstream 3000-cap)
        std::vector<int> pool(boxes.size());
        for (size_t i = 0; i < pool.size(); ++i) pool[i] = static_cast<int>(i);
        auto by_score = [&](int a, int b) {
            if (scores[a] != scores[b]) return scores[a] > scores[b];
            // Explicit tie-break keeps cluster membership reproducible across
            // standard-library implementations and compiler versions.
            return a < b;
        };
        if (pool.size() > 3000) {
            std::partial_sort(pool.begin(), pool.begin() + 3000, pool.end(), by_score);
            pool.resize(3000);
        } else std::sort(pool.begin(), pool.end(), by_score);
        refined.resize(keep.size());
        for (size_t s = 0; s < keep.size(); ++s) {
            const int k = keep[s];
            double sw = 0, ax = 0, ay = 0, ax2 = 0, ay2 = 0;
            for (int m : pool) {
                const double ov = box_iou(off[k], off[m]);   // offset boxes: cross-class IoU is 0
                if (ov <= cfg.iou_thresh) continue;
                const double w = scores[m] * std::exp(-std::pow(1.0 - ov, 2.0) / cfg.cw_sigma);
                sw += w;
                ax  += w * boxes[m].x;                ay  += w * boxes[m].y;
                ax2 += w * (boxes[m].x + boxes[m].width);
                ay2 += w * (boxes[m].y + boxes[m].height);
            }
            if (sw > 1e-6) {                          // guard: near-zero weight keeps the original box
                const double x0 = ax / sw, y0 = ay / sw;
                refined[s] = cv::Rect2d(x0, y0, std::max(0.0, ax2 / sw - x0),
                                                std::max(0.0, ay2 / sw - y0));
            } else refined[s] = boxes[k];
        }
    }

    std::vector<Detection> dets;
    const cv::Rect2d frame(0, 0, orig_w, orig_h);
    for (size_t s = 0; s < keep.size(); ++s) {       // keep is score-descending
        const int k = keep[s];
        if (static_cast<int>(dets.size()) >= cfg.max_det) break;
        const RawDet& c = cands[idx[k]];
        const cv::Rect2d raw = refined.empty()
            ? cv::Rect2d(c.box.x, c.box.y, c.box.width, c.box.height) : refined[s];
        cv::Rect2d b = raw & frame;                  // clip in float
        if (b.width > 0 && b.height > 0) {
            Detection d; d.class_id = c.cls; d.conf = c.score;
            d.box = cv::Rect2f(static_cast<float>(b.x), static_cast<float>(b.y),
                               static_cast<float>(b.width), static_cast<float>(b.height));
            d.mask_coeffs = c.mask_coeffs;
            dets.push_back(std::move(d));
        }
    }
    return dets;
}

std::vector<Detection> decode(const float* out, int feat_dim, int num_anchors,
                              const Config& cfg, const LetterboxInfo& lb) {
    auto cands = decode_candidates(out, feat_dim, num_anchors, cfg, lb);
    return nms_and_cap(cands, cfg, lb.orig_w, lb.orig_h);
}

// 10-color class palette (RGB 0..1), indexed cls%10 - identical to the GUI and the Mac runner.
static const float kPalette[10][3] = {
    {0.98f,0.26f,0.30f},{0.20f,0.71f,0.98f},{0.16f,0.85f,0.52f},{0.99f,0.79f,0.12f},
    {0.72f,0.40f,0.98f},{0.99f,0.55f,0.18f},{0.10f,0.83f,0.80f},{0.98f,0.36f,0.66f},
    {0.55f,0.82f,0.28f},{0.40f,0.52f,0.98f},
};
const float* class_color(int class_id) { return kPalette[((class_id % 10) + 10) % 10]; }

static inline float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

cv::Mat seg_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                    int pc, int ph, int pw, const LetterboxInfo& lb, int imgsz,
                    int orig_w, int orig_h, int mask_alpha) {
    cv::Mat rgba(orig_h, orig_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    if (proto.empty() || pc <= 0 || imgsz <= 0) return rgba;
    const size_t plane = static_cast<size_t>(ph) * pw;
    const float sx = lb.scale_x * pw / imgsz, sy = lb.scale_y * ph / imgsz;  // orig px -> mask space
    const float ox0 = lb.pad_x * (float)pw / imgsz, oy0 = lb.pad_y * (float)ph / imgsz;
    std::vector<float> ml(plane);
    for (const auto& d : dets) {
        if (static_cast<int>(d.mask_coeffs.size()) != pc) continue;   // detection-only box: skip
        const float* co = d.mask_coeffs.data();
        // lowres mask = sigmoid(coeffs . protos), computed once per detection
        for (size_t i = 0; i < plane; ++i) {
            float s = 0.f;
            for (int c = 0; c < pc; ++c) s += co[c] * proto[c * plane + i];
            ml[i] = 1.f / (1.f + std::exp(-s));
        }
        const float* col = class_color(d.class_id);
        const int x0 = std::max(0, (int)std::floor(d.box.x));
        const int y0 = std::max(0, (int)std::floor(d.box.y));
        const int x1 = std::min(orig_w, (int)std::ceil(d.box.x + d.box.width));
        const int y1 = std::min(orig_h, (int)std::ceil(d.box.y + d.box.height));
        for (int oy = y0; oy < y1; ++oy) {
            const float my = oy * sy + oy0;
            const int myi = std::clamp((int)my, 0, ph - 1);
            const int myi2 = std::min(myi + 1, ph - 1);
            const float fy = std::clamp(my - myi, 0.f, 1.f);
            uint8_t* dst = rgba.ptr<uint8_t>(oy);
            for (int ox = x0; ox < x1; ++ox) {
                const float mx = ox * sx + ox0;
                const int mxi = std::clamp((int)mx, 0, pw - 1);
                const int mxi2 = std::min(mxi + 1, pw - 1);
                const float fx = std::clamp(mx - mxi, 0.f, 1.f);
                // bilinear sample of the lowres mask
                const float v = ml[myi * pw + mxi]   * (1 - fx) * (1 - fy)
                              + ml[myi * pw + mxi2]  * fx       * (1 - fy)
                              + ml[myi2 * pw + mxi]  * (1 - fx) * fy
                              + ml[myi2 * pw + mxi2] * fx       * fy;
                const float edge = smoothstep(0.5f - 0.14f, 0.5f + 0.14f, v);  // soft anti-aliased border
                if (edge <= 0.f) continue;
                const uint8_t a = (uint8_t)(edge * mask_alpha);
                uint8_t* px = dst + ox * 4;
                if (a > px[3]) {   // overlapping masks: keep the more opaque one
                    px[0] = (uint8_t)(col[0] * 255); px[1] = (uint8_t)(col[1] * 255);
                    px[2] = (uint8_t)(col[2] * 255); px[3] = a;
                }
            }
        }
    }
    return rgba;
}

void draw(cv::Mat& img, const std::vector<Detection>& dets, const Config& cfg) {
    for (const auto& d : dets) {
        const cv::Rect r(cvRound(d.box.x), cvRound(d.box.y), cvRound(d.box.width), cvRound(d.box.height));
        const float* pc = class_color(d.class_id);
        const cv::Scalar color(pc[2] * 255, pc[1] * 255, pc[0] * 255);   // RGB palette -> BGR
        cv::rectangle(img, r, color, 2);
        const std::string name = (d.class_id < cfg.num_classes()) ? cfg.class_names[d.class_id]
                                                                  : std::to_string(d.class_id);
        char buf[80];
        std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), d.conf);
        int base = 0;
        cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(img, cv::Rect(r.x, std::max(0, r.y - ts.height - 4),
                                    ts.width + 2, ts.height + 4), color, cv::FILLED);
        cv::putText(img, buf, cv::Point(r.x, std::max(ts.height, r.y - 3)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ---------------- metadata ----------------
namespace meta {

std::vector<std::string> parse_names_dict(const std::string& s) {
    // Ultralytics has emitted all of the following forms over time:
    //   {0: 'person', 1: 'car'}  (Python repr)
    //   {"0": "person", "1": "car"} (JSON object)
    //   ["person", "car"] (JSON list)
    // Extract key/value pairs rather than every quoted token: the latter
    // mistakenly turns JSON's numeric keys into class names and shifts the
    // class ABI by one position.
    std::map<int, std::string> keyed;
    std::vector<std::string> listed;
    const auto skip_ws = [&](size_t& p) {
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    };
    const auto quoted = [&](size_t& p, std::string& value) -> bool {
        skip_ws(p);
        if (p >= s.size() || (s[p] != '\'' && s[p] != '"')) return false;
        const char q = s[p++];
        value.clear();
        while (p < s.size()) {
            const char c = s[p++];
            if (c == '\\' && p < s.size()) { value.push_back(s[p++]); continue; }
            if (c == q) return true;
            value.push_back(c);
        }
        return false;
    };
    size_t p = 0;
    skip_ws(p);
    if (p < s.size() && s[p] == '[') {
        ++p;
        while (p < s.size()) {
            skip_ws(p);
            if (p < s.size() && s[p] == ']') break;
            std::string value;
            if (!quoted(p, value)) break;
            listed.push_back(std::move(value));
            skip_ws(p);
            if (p < s.size() && s[p] == ',') ++p;
        }
        return listed;
    }
    if (p >= s.size() || s[p] != '{') return {};
    ++p;
    while (p < s.size()) {
        skip_ws(p);
        if (p < s.size() && s[p] == '}') break;
        int key = -1;
        size_t key_start = p;
        if (p < s.size() && (s[p] == '\'' || s[p] == '"')) {
            std::string key_text;
            if (!quoted(p, key_text)) break;
            try { key = std::stoi(key_text); } catch (...) { key = -1; }
        } else {
            while (p < s.size() && (std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '-')) ++p;
            if (p == key_start) break;
            try { key = std::stoi(s.substr(key_start, p - key_start)); } catch (...) { key = -1; }
        }
        skip_ws(p);
        if (p >= s.size() || s[p] != ':') break;
        ++p;
        std::string value;
        if (!quoted(p, value)) break;
        if (key >= 0) keyed[key] = std::move(value);
        skip_ws(p);
        if (p < s.size() && s[p] == ',') ++p;
    }
    if (keyed.empty()) return {};
    const int max_key = keyed.rbegin()->first;
    std::vector<std::string> names(static_cast<size_t>(max_key + 1));
    for (const auto& item : keyed) names[static_cast<size_t>(item.first)] = item.second;
    return names;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

bool read_ncnn_yaml(const std::string& path, std::vector<std::string>& names, int& imgsz,
                    std::string* input_blob, std::string* output_blob,
                    std::string* proto_blob) {
    std::ifstream f(fs::u8path(path));
    if (!f) return false;
    std::map<int, std::string> nm;
    imgsz = 0;
    if (input_blob) input_blob->clear();
    if (output_blob) output_blob->clear();
    if (proto_blob) proto_blob->clear();
    std::string line;
    enum { NONE, NAMES, IMGSZ } sec = NONE;
    auto scalar = [](const std::string& value) {
        std::string out = trim(value);
        if (out.size() >= 2 && ((out.front() == '"' && out.back() == '"') ||
                                (out.front() == '\'' && out.back() == '\''))) {
            out = out.substr(1, out.size() - 2);
        }
        return out;
    };
    while (std::getline(f, line)) {
        const bool indented = !line.empty() && (line[0] == ' ' || line[0] == '\t' || line[0] == '-');
        if (!indented) {                                   // top-level key -> switch/close section
            if (line.rfind("names:", 0) == 0) { sec = NAMES; continue; }
            if (line.rfind("imgsz:", 0) == 0) {
                sec = IMGSZ;
                auto p = line.find('[');                   // inline "imgsz: [640, 640]"
                if (p != std::string::npos) imgsz = std::atoi(line.c_str() + p + 1);
                continue;
            }
            if (line.rfind("input_blob:", 0) == 0) {
                if (input_blob) *input_blob = scalar(line.substr(std::string("input_blob:").size()));
                sec = NONE; continue;
            }
            if (line.rfind("output_blob:", 0) == 0) {
                if (output_blob) *output_blob = scalar(line.substr(std::string("output_blob:").size()));
                sec = NONE; continue;
            }
            if (line.rfind("proto_blob:", 0) == 0) {
                if (proto_blob) *proto_blob = scalar(line.substr(std::string("proto_blob:").size()));
                sec = NONE; continue;
            }
            sec = NONE; continue;
        }
        if (sec == NAMES) {                                // "  0: pedestrian"
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                const int idx = std::atoi(scalar(line.substr(0, colon)).c_str());
                // Exporters commonly emit JSON-quoted YAML scalars.  Strip
                // the surrounding quotes so class labels are not displayed
                // as part of the name (and remain stable across backends).
                nm[idx] = scalar(line.substr(colon + 1));
            }
        } else if (sec == IMGSZ && imgsz == 0) {           // "- 640"
            auto d = line.find_first_of("0123456789");
            if (d != std::string::npos) imgsz = std::atoi(line.c_str() + d);
        }
    }
    names.clear();
    for (auto& kv : nm) names.push_back(kv.second);
    return !names.empty() ||
           imgsz > 0 ||
           (input_blob && !input_blob->empty()) ||
           (output_blob && !output_blob->empty()) ||
           (proto_blob && !proto_blob->empty());
}

} // namespace meta

// ---------------- source ----------------
static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
// Keep this list in sync with imread_bgr/stb_image.  stb_image does not decode
// TIFF or WebP; advertising those suffixes would make a directory run count
// unreadable files as inputs and produce misleading acceptance summaries.
static const std::set<std::string> kImageExt = {".jpg", ".jpeg", ".png", ".bmp"};
static const std::set<std::string> kVideoExt = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

SourceKind classify_source(const std::string& src) {
    std::error_code ec;
    if (fs::is_directory(src, ec)) return SourceKind::Dir;
    const std::string ext = lower(fs::path(src).extension().string());
    if (ext == ".yaml" || ext == ".yml") return SourceKind::Dataset;
    if (ext == ".txt" || ext == ".list") return SourceKind::List;
    if (kVideoExt.count(ext)) return SourceKind::Video;
    if (kImageExt.count(ext)) return SourceKind::Image;
    return SourceKind::Unknown;
}

static void collect_dir(const std::string& dir, std::vector<std::string>& out) {
    std::error_code ec;
    fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        const auto& e = *it;
        if (!e.is_regular_file(ec)) continue;
        if (kImageExt.count(lower(e.path().extension().string())))
            out.push_back(e.path().lexically_normal().string());
    }
    // Directory traversal order is not guaranteed by the filesystem.  Use a
    // separator-normalized, case-folded key so diagnostic runs are stable on
    // Windows and Linux; callers requiring a publication-grade order should
    // still provide an explicit frozen image list.
    std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
        const std::string ak = lower(fs::u8path(a).generic_string());
        const std::string bk = lower(fs::u8path(b).generic_string());
        return ak == bk ? a < b : ak < bk;
    });
}

static std::string trim_source_line(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static void validate_source_stems(const std::vector<std::string>& images) {
    std::string error;
    if (!validate_unique_stems(images, error)) throw std::runtime_error(error);
}

// Read a frozen, newline-delimited image list. Unlike directory traversal,
// this path preserves the experiment's declared order, which is part of the
// Issue #51 evidence manifest. Relative paths are resolved against the list
// file, matching the convention used by the Python evaluators.
static std::vector<std::string> resolve_image_list(const std::string& list_path) {
    std::ifstream f(fs::u8path(list_path));
    if (!f) throw std::runtime_error("failed to open image list: " + list_path);
    const fs::path absolute_list = fs::absolute(fs::u8path(list_path));
    const fs::path base = absolute_list.parent_path();
    std::vector<std::string> out;
    std::string line;
    size_t line_number = 0;
    while (std::getline(f, line)) {
        ++line_number;
        if (line_number == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xef &&
            static_cast<unsigned char>(line[1]) == 0xbb &&
            static_cast<unsigned char>(line[2]) == 0xbf) {
            line.erase(0, 3);  // UTF-8 BOM from a Windows-generated list
        }
        line = trim_source_line(line);
        if (line.empty() || line.front() == '#') continue;
        // Accept either YAML/Python-style single quotes or JSON-style double
        // quotes. The Python evaluators apply the same rule; keeping the
        // parser symmetric matters when one frozen list is shared by C++ and
        // the metric scripts. Only strip a matching pair so an apostrophe in
        // an unquoted filename remains part of the path.
        if (line.size() >= 2 && line.front() == line.back() &&
            (line.front() == '\'' || line.front() == '"'))
            line = trim_source_line(line.substr(1, line.size() - 2));
        fs::path image = fs::u8path(line);
        if (image.is_relative()) image = base / image;
        image = image.lexically_normal();
        std::error_code ec;
        if (!fs::is_regular_file(image, ec)) {
            throw std::runtime_error("image list line " + std::to_string(line_number) +
                                     " does not name a regular file: " + image.u8string());
        }
        if (!kImageExt.count(lower(image.extension().string()))) {
            throw std::runtime_error("unsupported image extension at list line " +
                                     std::to_string(line_number) + ": " + image.u8string());
        }
        out.push_back(image.u8string());
    }
    if (out.empty()) throw std::runtime_error("image list is empty: " + list_path);
    validate_source_stems(out);
    return out;
}

// Keep dataset YAML list entries on the same frozen-list validation path.
static std::vector<std::string> resolve_image_list_entry(const fs::path& c) {
    return resolve_image_list(c.string());
}

// Resolve the small, intentionally supported subset of Ultralytics dataset
// YAML needed by the runner.  `val` may be a scalar, an inline sequence, or a
// block sequence.  We do not pull in a YAML library for this command-line
// utility, but malformed/unsupported entries fail closed with a useful error.
static std::vector<std::string> resolve_dataset(const std::string& yaml) {
    std::ifstream f(fs::u8path(yaml));
    if (!f) throw std::runtime_error("failed to open dataset YAML: " + yaml);
    std::string path, line;
    std::vector<std::string> val_entries;
    bool val_block = false;
    int val_indent = -1;
    auto yaml_scalar = [](std::string value) {
        value = trim_source_line(value);
        if (value.size() >= 2 &&
            ((value.front() == '\"' && value.back() == '\"') ||
             (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        return trim_source_line(value);
    };
    auto strip_yaml_comment = [](const std::string& value) {
        bool single = false, doubled = false;
        for (size_t i = 0; i < value.size(); ++i) {
            const char c = value[i];
            if (c == '\'' && !doubled) single = !single;
            else if (c == '"' && !single) doubled = !doubled;
            else if (c == '#' && !single && !doubled &&
                     (i == 0 || std::isspace(static_cast<unsigned char>(value[i - 1]))))
                return value.substr(0, i);
        }
        return value;
    };
    auto inline_sequence = [&](const std::string& value) {
        std::vector<std::string> entries;
        std::string body = trim_source_line(strip_yaml_comment(value));
        if (body.size() < 2 || body.front() != '[' || body.back() != ']') return entries;
        body = body.substr(1, body.size() - 2);
        size_t start = 0;
        bool single = false, doubled = false;
        for (size_t i = 0; i <= body.size(); ++i) {
            const char c = (i < body.size()) ? body[i] : ',';
            if (c == '\'' && !doubled) single = !single;
            else if (c == '"' && !single) doubled = !doubled;
            if (c == ',' && !single && !doubled) {
                std::string item = yaml_scalar(body.substr(start, i - start));
                if (!item.empty()) entries.push_back(std::move(item));
                start = i + 1;
            }
        }
        return entries;
    };
    size_t line_number = 0;
    while (std::getline(f, line)) {
        ++line_number;
        if (line_number == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xef &&
            static_cast<unsigned char>(line[1]) == 0xbb &&
            static_cast<unsigned char>(line[2]) == 0xbf)
            line.erase(0, 3);  // UTF-8 BOM from a Windows-generated YAML
        const std::string raw = line;
        const size_t first = raw.find_first_not_of(" \t");
        const int indent = (first == std::string::npos) ? 0 : static_cast<int>(first);
        const std::string content = (first == std::string::npos) ? "" : raw.substr(first);

        if (!content.empty() && content.front() != '#') {
            if (indent == 0 && content.rfind("path:", 0) == 0) {
                path = yaml_scalar(strip_yaml_comment(content.substr(5)));
                val_block = false;
                continue;
            }
            if (indent == 0 && content.rfind("val:", 0) == 0) {
                const std::string value = trim_source_line(strip_yaml_comment(content.substr(4)));
                val_entries.clear();
                if (value.empty()) {
                    val_block = true;
                    // `val` is a top-level key in the supported dataset
                    // schema; list items must therefore be more indented.
                    val_indent = indent;
                } else {
                    const auto listed = inline_sequence(value);
                    if (!listed.empty() || trim_source_line(value) == "[]")
                        val_entries = listed;
                    else
                        val_entries.push_back(yaml_scalar(value));
                    val_block = false;
                }
                continue;
            }
        }

        if (val_block) {
            if (first == std::string::npos || content.empty() || content.front() == '#') continue;
            if (val_indent < 0) val_indent = indent;
            // YAML also permits an indentationless block sequence:
            //   val:
            //   - images/val
            // Accept a dash at the key indentation as long as
            // it is clearly a sequence item; any other top-level key closes
            // the block.
            if (content.front() == '-' && indent >= val_indent) {
                const std::string item = yaml_scalar(
                    strip_yaml_comment(content.size() > 1 ? content.substr(1) : ""));
                if (!item.empty()) val_entries.push_back(item);
                continue;
            }
            if (indent <= val_indent) {
                // A new top-level key closes a block sequence.  It is handled
                // on its own line above; no val item is inferred from it.
                val_block = false;
                continue;
            }
        }
    }
    if (val_entries.empty())
        throw std::runtime_error("dataset YAML has no supported non-empty 'val' split");
    // Resolve a relative dataset `path:` against the YAML file first, as
    // Ultralytics does. Falling back to the process working directory is
    // retained for legacy manifests that intentionally use cwd-relative
    // paths. Canonicalising the YAML location also handles a bare filename
    // (`data.yaml`) without introducing an empty path component.
    const fs::path yaml_abs = fs::absolute(fs::u8path(yaml)).lexically_normal();
    const fs::path ydir = yaml_abs.parent_path();
    const fs::path dataset_root = path.empty()
        ? ydir
        : (fs::path(path).is_absolute() ? fs::path(path)
                                        : (ydir / fs::path(path)).lexically_normal());
    std::vector<std::string> out;
    for (const std::string& entry : val_entries) {
        const fs::path val_path = fs::u8path(entry);
        // For relative values the YAML-defined root is authoritative.  The
        // process-working-directory candidate is retained only as a
        // compatibility fallback for older manifests written with cwd paths.
        std::vector<fs::path> cands;
        if (val_path.is_absolute()) {
            cands.push_back(val_path);
        } else {
            cands.push_back(dataset_root / val_path);
            cands.push_back(ydir / val_path);
            cands.push_back(val_path);
        }
        cands.push_back(fs::path("/data/datasets") / path / val_path);
        std::error_code ec;
        bool resolved = false;
        for (const auto& c : cands) {
            if (fs::is_directory(c, ec)) {
                std::vector<std::string> images;
                collect_dir(c.string(), images);
                if (!images.empty()) {
                    out.insert(out.end(), images.begin(), images.end());
                    resolved = true;
                    break;
                }
            }
            const std::string extension = lower(c.extension().string());
            if (fs::is_regular_file(c, ec) && (extension == ".txt" || extension == ".list")) {
                const auto images = resolve_image_list_entry(c);
                out.insert(out.end(), images.begin(), images.end());
                resolved = true;
                break;
            }
            if (fs::is_regular_file(c, ec) && kImageExt.count(extension)) {
                out.push_back(c.lexically_normal().u8string());
                resolved = true;
                break;
            }
        }
        if (!resolved)
            throw std::runtime_error("dataset val entry does not resolve to an image directory or .txt/.list: " + entry);
    }
    if (out.empty()) throw std::runtime_error("dataset val split contains no supported images");
    validate_source_stems(out);
    return out;
}

std::vector<std::string> gather_images(const std::string& src, int limit) {
    std::vector<std::string> out;
    switch (classify_source(src)) {
        case SourceKind::Image:   out = {src}; break;
        case SourceKind::Dir:     collect_dir(src, out); break;
        case SourceKind::Dataset: out = resolve_dataset(src); break;
        case SourceKind::List:    out = resolve_image_list(src); break;
        default: break;
    }
    if (limit > 0 && static_cast<int>(out.size()) > limit) out.resize(limit);
    return out;
}

bool validate_unique_stems(const std::vector<std::string>& images, std::string& error) {
    std::map<std::string, std::string> seen;
    for (const std::string& image : images) {
        const fs::path path(image);
        const std::string stem = path.stem().string();
        const std::string key = lower(stem);
        const auto it = seen.find(key);
        if (it != seen.end()) {
            error = "duplicate image stems: '" + stem + "' in " + it->second +
                    " and " + image;
            return false;
        }
        seen.emplace(key, image);
    }
    error.clear();
    return true;
}

} // namespace yolomaster
