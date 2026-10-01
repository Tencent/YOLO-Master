// Shared, backend/model-agnostic ops: class tables, letterbox, decode+NMS,
// drawing, model-metadata parsing, and versatile source resolution.
#include "yolomaster.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <filesystem>
#include <sstream>
#include <cstdlib>
#include <stdexcept>

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

void letterbox_params(int w, int h, int imgsz, bool stretch, LetterboxInfo& info, int& out_w, int& out_h) {
    info.orig_w = w;
    info.orig_h = h;
    if (stretch) {
        // resize straight to imgsz x imgsz, ignoring aspect -> per-axis scale, no pad.
        info.scale_x = imgsz / static_cast<float>(w);
        info.scale_y = imgsz / static_cast<float>(h);
        info.scale   = info.scale_x;   // ambiguous under stretch; keep x for any legacy reader
        info.pad_x = info.pad_y = 0;
        out_w = out_h = imgsz;
        return;
    }
    // letterbox: min-scale aspect-preserving, 114-gray padded, centered.
    const float r = std::min(imgsz / static_cast<float>(w), imgsz / static_cast<float>(h));
    out_w = static_cast<int>(std::round(w * r));
    out_h = static_cast<int>(std::round(h * r));
    info.scale = info.scale_x = info.scale_y = r;
    info.pad_x = (imgsz - out_w) / 2;
    info.pad_y = (imgsz - out_h) / 2;
}

cv::Mat preprocess(const cv::Mat& img, int imgsz, bool stretch, LetterboxInfo& info) {
    int nw = 0, nh = 0;
    letterbox_params(img.cols, img.rows, imgsz, stretch, info, nw, nh);
    if (stretch) {
        cv::Mat out;
        cv::resize(img, out, cv::Size(imgsz, imgsz));
        return out;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat out(imgsz, imgsz, img.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, nw, nh)));
    return out;
}

void preprocess_nchw(const cv::Mat& bgr, int imgsz, bool stretch, float* dst, LetterboxInfo& info) {
    const cv::Mat padded = preprocess(bgr, imgsz, stretch, info);   // imgsz x imgsz, CV_8UC3 BGR
    const int sz = imgsz, hw = sz * sz;
    float* r = dst; float* g = dst + hw; float* b = dst + 2 * hw;
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;          // BGR -> RGB /255, planar
            const int idx = y * sz + x;
            r[idx] = px[2] * (1.0f / 255);
            g[idx] = px[1] * (1.0f / 255);
            b[idx] = px[0] * (1.0f / 255);
        }
    }
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
        if (scores[i] >= conf) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
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

// Decode raw output -> pre-NMS candidates (feat_dim = 4 + nc [+ nm mask coeffs]). Box in orig px.
std::vector<RawDet> decode_candidates(const float* out, int feat_dim, int num_anchors,
                                      const Config& cfg, const LetterboxInfo& lb) {
    const int nc = cfg.num_classes() > 0 ? cfg.num_classes() : (feat_dim - 4);
    const int nm = feat_dim - 4 - nc;                // mask-coeff count (0 = detection)
    std::vector<RawDet> cands;
    auto make = [&](int a, int cls, float score) {
        const float cx = out[0 * num_anchors + a];
        const float cy = out[1 * num_anchors + a];
        const float w  = out[2 * num_anchors + a];
        const float h  = out[3 * num_anchors + a];
        const float x0 = (cx - 0.5f * w - lb.pad_x) / lb.scale_x;
        const float y0 = (cy - 0.5f * h - lb.pad_y) / lb.scale_y;
        RawDet d; d.box = cv::Rect2f(x0, y0, w / lb.scale_x, h / lb.scale_y); d.score = score; d.cls = cls;
        if (nm > 0) { d.mask_coeffs.resize(nm);
            for (int k = 0; k < nm; ++k) d.mask_coeffs[k] = out[(4 + nc + k) * num_anchors + a]; }
        cands.push_back(std::move(d));
    };
    for (int a = 0; a < num_anchors; ++a) {
        int best = -1; float bestv = 0.f; bool any = false;
        for (int c = 0; c < nc; ++c) {
            const float v = out[(4 + c) * num_anchors + a];
            if (v > bestv) { bestv = v; best = c; }
            if (cfg.multi_label && v >= cfg.conf_thresh) any = true;
        }
        if (!(cfg.multi_label ? any : (bestv >= cfg.conf_thresh))) continue;
        if (cfg.multi_label) {                       // one candidate per class >= conf
            for (int c = 0; c < nc; ++c) {
                const float v = out[(4 + c) * num_anchors + a];
                if (v >= cfg.conf_thresh) make(a, c, v);
            }
        } else make(a, best, bestv);                 // single best class
    }
    return cands;
}

std::vector<RawDet> decode_end2end(const float* out, int num_det,
                                   const Config& cfg, const LetterboxInfo& lb) {
    std::vector<RawDet> cands;
    for (int i = 0; i < num_det; ++i) {
        const float* r = out + static_cast<size_t>(i) * 6;
        const float score = r[4];
        if (score < cfg.conf_thresh) continue;       // also drops max_det zero-padding rows
        const float x0 = (r[0] - lb.pad_x) / lb.scale_x;
        const float y0 = (r[1] - lb.pad_y) / lb.scale_y;
        const float x1 = (r[2] - lb.pad_x) / lb.scale_x;
        const float y1 = (r[3] - lb.pad_y) / lb.scale_y;
        if (x1 <= x0 || y1 <= y0) continue;
        RawDet d;
        d.box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
        d.score = score;
        d.cls = static_cast<int>(r[5]);
        cands.push_back(std::move(d));
    }
    return cands;
}

// Per-class NMS (ultralytics agnostic=False via class offset) + max_det cap + clip-to-frame.
// ---- Backend defaults (the shared forward_raw / infer contract) ----
void Backend::forward_raw(const cv::Mat&, const Config&, bool) {
    throw std::runtime_error(std::string(runtime_name()) + " backend has no forward_raw (infer() only)");
}

std::vector<Detection> Backend::infer(const cv::Mat& bgr, const Config& cfg) {
    forward_raw(bgr, cfg, /*decode=*/true);
    // post_ms keeps its historical meaning for infer(): decode + NMS.
    const auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<Detection> dets = nms_and_cap(candidates, cfg, cand_orig_w, cand_orig_h);
    post_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t3).count();
    return dets;
}

std::vector<Detection> nms_and_cap(const std::vector<RawDet>& cands, const Config& cfg,
                                   int orig_w, int orig_h) {
    std::vector<cv::Rect2d> boxes; std::vector<float> scores; std::vector<int> idx;
    boxes.reserve(cands.size()); scores.reserve(cands.size()); idx.reserve(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].score < cfg.conf_thresh) continue;
        boxes.emplace_back(cands[i].box.x, cands[i].box.y, cands[i].box.width, cands[i].box.height);
        scores.push_back(cands[i].score); idx.push_back(static_cast<int>(i));
    }
    std::vector<int> keep;
    // Per-class stratification: translate each class into its own disjoint stratum so one
    // greedy pass does agnostic=False NMS. Dynamic offset: candidates are UNCLIPPED (the
    // letterbox inverse can overshoot the frame), so leave generous margin beyond the
    // largest dimension; IoU is translation-invariant, results match any big-enough offset.
    const double OFF = 2.0 * std::max(orig_w, orig_h) + 8192.0;
    std::vector<cv::Rect2d> off = boxes;
    for (size_t k = 0; k < off.size(); ++k) {
        const int cls = cands[idx[k]].cls;
        off[k].x += cls * OFF; off[k].y += cls * OFF;
    }
    nms_greedy(off, scores, cfg.conf_thresh, cfg.iou_thresh, keep);

    // Cluster-Weighted refinement (ultralytics cluster branch): each greedy survivor's box
    // becomes the score-and-proximity weighted average of its cluster - every same-class
    // candidate overlapping it above iou_thresh, weighted w = s * exp(-(1-IoU)^2 / sigma).
    // One-shot over the ORIGINAL coords; survivor set/order/scores/classes are untouched.
    std::vector<cv::Rect2d> refined;
    if (cfg.nms_mode == NmsMode::ClusterWeighted && !keep.empty()) {
        // pool = top-3000 of the conf-filtered candidates by score (upstream 3000-cap)
        std::vector<int> pool(boxes.size());
        for (size_t i = 0; i < pool.size(); ++i) pool[i] = static_cast<int>(i);
        auto by_score = [&](int a, int b) { return scores[a] > scores[b]; };
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
            d.cand_index = idx[k];
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
    return seg_overlay(dets, proto, pc, ph, pw, lb, imgsz, orig_w, orig_h, orig_w, orig_h, mask_alpha);
}

cv::Mat seg_overlay(const std::vector<Detection>& dets, const std::vector<float>& proto,
                    int pc, int ph, int pw, const LetterboxInfo& lb, int imgsz,
                    int orig_w, int orig_h, int out_w, int out_h, int mask_alpha) {
    cv::Mat rgba(std::max(out_h, 0), std::max(out_w, 0), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    if (proto.empty() || pc <= 0 || imgsz <= 0 || out_w <= 0 || out_h <= 0 || orig_w <= 0 || orig_h <= 0)
        return rgba;
    const size_t plane = static_cast<size_t>(ph) * pw;
    const float sx = lb.scale_x * pw / imgsz, sy = lb.scale_y * ph / imgsz;  // orig px -> mask space
    const float ox0 = lb.pad_x * (float)pw / imgsz, oy0 = lb.pad_y * (float)ph / imgsz;
    // output px -> orig px. fx == fy == 1 when out == orig, which keeps every expression below
    // identical to the unsized rendering (x / 1.0f is exact).
    const float fx = (float)out_w / orig_w, fy = (float)out_h / orig_h;
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
        const int x0 = std::max(0, (int)std::floor(d.box.x * fx));
        const int y0 = std::max(0, (int)std::floor(d.box.y * fy));
        const int x1 = std::min(out_w, (int)std::ceil((d.box.x + d.box.width) * fx));
        const int y1 = std::min(out_h, (int)std::ceil((d.box.y + d.box.height) * fy));
        for (int oy = y0; oy < y1; ++oy) {
            const float my = (oy / fy) * sy + oy0;
            const int myi = std::clamp((int)my, 0, ph - 1);
            const int myi2 = std::min(myi + 1, ph - 1);
            const float fy = std::clamp(my - myi, 0.f, 1.f);
            uint8_t* dst = rgba.ptr<uint8_t>(oy);
            for (int ox = x0; ox < x1; ++ox) {
                const float mx = (ox / fx) * sx + ox0;
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

bool read_ncnn_yaml(const std::string& path, std::vector<std::string>& names, int& imgsz) {
    bool e2e_unused = false;
    return read_ncnn_yaml(path, names, imgsz, e2e_unused);
}

bool read_ncnn_yaml(const std::string& path, std::vector<std::string>& names, int& imgsz,
                    bool& end2end) {
    end2end = false;
    std::ifstream f(path);
    if (!f) return false;
    std::map<int, std::string> nm;
    imgsz = 0;
    std::string line;
    enum { NONE, NAMES, IMGSZ } sec = NONE;
    while (std::getline(f, line)) {
        const bool indented = !line.empty() && (line[0] == ' ' || line[0] == '\t' || line[0] == '-');
        if (!indented) {                                   // top-level key -> switch/close section
            if (line.rfind("names:", 0) == 0) { sec = NAMES; continue; }
            if (line.rfind("end2end:", 0) == 0) {
                end2end = line.find("rue") != std::string::npos;   // true/True
                sec = NONE; continue;
            }
            if (line.rfind("imgsz:", 0) == 0) {
                sec = IMGSZ;
                auto p = line.find('[');                   // inline "imgsz: [640, 640]"
                if (p != std::string::npos) imgsz = std::atoi(line.c_str() + p + 1);
                continue;
            }
            sec = NONE; continue;
        }
        if (sec == NAMES) {                                // "  0: pedestrian"
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                const int idx = std::atoi(trim(line.substr(0, colon)).c_str());
                nm[idx] = trim(line.substr(colon + 1));
            }
        } else if (sec == IMGSZ && imgsz == 0) {           // "- 640"
            auto d = line.find_first_of("0123456789");
            if (d != std::string::npos) imgsz = std::atoi(line.c_str() + d);
        }
    }
    names.clear();
    for (auto& kv : nm) names.push_back(kv.second);
    return !names.empty();
}

std::string NcnnParamScan::reason() const {
    if (!ok) return "param file unreadable";
    std::string r;
    if (router_emulated())
        r = "emulated-router fingerprint (1e-9 nudge x" + std::to_string(nudge_1e9) + ", 1e30 mask x" +
            std::to_string(mask_1e30) + ", amax x" + std::to_string(router_amax) + ")";
    if (fp16_overflow > 0) {
        if (!r.empty()) r += "; ";
        r += std::to_string(fp16_overflow) + " literal(s) overflow fp16";
    }
    return r;
}

NcnnParamScan scan_ncnn_param(const std::string& param_path) {
    NcnnParamScan s;
    std::ifstream f(param_path);
    if (!f) return s;
    std::string line;
    if (!std::getline(f, line) || line.find("7767517") == std::string::npos) return s;   // magic
    if (!std::getline(f, line)) return s;                                              // "layers blobs"
    s.ok = true;
    // ncnn writes floats with %e (always a '.' or an exponent); ints never do. Evaluating only
    // float-looking tokens avoids false overflow hits on weight counts such as "6=147456".
    enum Hazard { NONE = 0, BIG = 1, TINY = 2 };
    struct Rec {
        std::string type, name;
        std::vector<std::string> ins, outs;
        int seed = NONE;          // hazard this layer introduces through its own literals
        bool propagate = false;   // only the router mask (1e30) and nudge (1e-9) seeds spread downstream;
                                  // a FLT_MAX Clip bound or similar pins its own layer only (output <= input)
        int unary_op = -1;        // UnaryOp "0=" code
        double clip_max = 0;      // Clip "1=" literal
        bool has_clip_max = false;
    };
    std::vector<Rec> recs;
    auto eval_float = [&s](const std::string& v, int& seed, bool& propagate) {
        if (v.find_first_of(".eE") == std::string::npos) return;
        if (v == "1.000000e-9") { s.nudge_1e9++; seed |= TINY; propagate = true; return; }
        if (v == "1.000000e30") { s.mask_1e30++; seed |= BIG; propagate = true; return; }
        char* end = nullptr;
        const double d = std::strtod(v.c_str(), &end);
        if (end == v.c_str()) return;
        const double a = std::fabs(d);
        if (a > 65504.0) { s.fp16_overflow++; seed |= BIG; }
        else if (a > 0.0 && a < 6.1035e-5) s.fp16_flush++;
    };
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        Rec r;
        int nin = 0, nout = 0;
        if (!(ls >> r.type >> r.name >> nin >> nout)) continue;
        s.layers++;
        std::string tok;
        for (int i = 0; i < nin && (ls >> tok); ++i) r.ins.push_back(tok);
        for (int i = 0; i < nout && (ls >> tok); ++i) r.outs.push_back(tok);
        if (r.type == "Reduction" && r.name.rfind("amax_", 0) == 0) s.router_amax++;
        const bool quantizable =
            (r.type == "Convolution" || r.type == "ConvolutionDepthWise" || r.type == "InnerProduct");
        while (ls >> tok) {
            const auto eq = tok.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
            if (quantizable && key == "8" && val != "0") s.int8_layers++;
            if (r.type == "UnaryOp" && key == "0") r.unary_op = std::atoi(val.c_str());
            if (r.type == "Clip" && key == "1") { r.clip_max = std::strtod(val.c_str(), nullptr); r.has_clip_max = true; }
            if (!key.empty() && key[0] == '-') {           // array param "-23303=1,0" -> elements
                std::istringstream vs(val);
                std::string el;
                while (std::getline(vs, el, ',')) eval_float(el, r.seed, r.propagate);
            } else {
                eval_float(val, r.seed, r.propagate);
            }
        }
        recs.push_back(std::move(r));
    }
    if (s.fp16_safe()) return s;

    // ---- hazard propagation: which layers must keep fp32 storage/arithmetic ----
    // A blob carries BIG (>65504) after a 1e30 mask multiply and through every consumer until a
    // layer that bounds it (Clip with a representable max, Softmax, Sigmoid) or integerises it
    // (UnaryOp floor/ceil/round/trunc). A blob carries TINY (the 1e-9 nudge that must survive
    // ceil()) through every consumer until an integerising UnaryOp, Softmax or Sigmoid. Clip does
    // NOT stop TINY: clamp(1e-9, 0, 1) must stay 1e-9. The stopping layer itself is pinned (it
    // reads the hazardous blob); its outputs are safe. Split/Reshape/Permute only move storage,
    // but pinned storage is exactly what keeps -1e30 alive between two arithmetic layers.
    std::map<std::string, std::vector<int>> consumers;
    for (int i = 0; i < static_cast<int>(recs.size()); ++i)
        for (const auto& b : recs[i].ins) consumers[b].push_back(i);
    std::vector<int> pinned(recs.size(), NONE);         // hazard mask a layer must run in fp32 for
    std::vector<std::pair<int, int>> work;
    for (int i = 0; i < static_cast<int>(recs.size()); ++i)
        if (recs[i].seed) { pinned[i] |= recs[i].seed; if (recs[i].propagate) work.emplace_back(i, recs[i].seed); }
    auto stops = [](const Rec& r, int hz) -> int {
        const bool integerising = r.type == "UnaryOp" &&
            (r.unary_op == 2 || r.unary_op == 3 || r.unary_op == 18 || r.unary_op == 19);
        if (integerising || r.type == "Softmax" || r.type == "Sigmoid") return hz;      // stops both
        if ((hz & BIG) && r.type == "Clip" && r.has_clip_max && std::fabs(r.clip_max) <= 65504.0) return static_cast<int>(BIG);
        return static_cast<int>(NONE);
    };
    std::set<std::pair<int, int>> seen;
    while (!work.empty()) {
        auto [li, hz] = work.back(); work.pop_back();
        if (!seen.insert({li, hz}).second) continue;
        for (const auto& b : recs[li].outs)
            for (int ci : consumers[b]) {
                const int already = pinned[ci];
                pinned[ci] |= hz;
                const int passed = hz & ~stops(recs[ci], hz);
                if (passed && (pinned[ci] != already || !seen.count({ci, passed})))
                    work.emplace_back(ci, passed);
            }
    }
    for (int i = 0; i < static_cast<int>(recs.size()); ++i)
        if (pinned[i]) { s.pin_layer_idx.push_back(i); s.pin_layer_names.push_back(recs[i].name); }
    return s;
}

bool read_ncnn_yaml_scalar(const std::string& path, const std::string& key, std::string& value) {
    std::ifstream f(path);
    if (!f) return false;
    const std::string k = key + ":";
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ' ' || line[0] == '\t' || line[0] == '-') continue;   // top-level only
        if (line.rfind(k, 0) == 0) { value = trim(line.substr(k.size())); return true; }
    }
    return false;
}

} // namespace meta

// ---------------- source ----------------
static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}
static const std::set<std::string> kImageExt = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
static const std::set<std::string> kVideoExt = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

SourceKind classify_source(const std::string& src) {
    std::error_code ec;
    if (fs::is_directory(src, ec)) return SourceKind::Dir;
    const std::string ext = lower(fs::path(src).extension().string());
    if (ext == ".yaml" || ext == ".yml") return SourceKind::Dataset;
    if (kVideoExt.count(ext)) return SourceKind::Video;
    if (kImageExt.count(ext)) return SourceKind::Image;
    return SourceKind::Unknown;
}

static void collect_dir(const std::string& dir, std::vector<std::string>& out) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        if (kImageExt.count(lower(e.path().extension().string())))
            out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
}

// best-effort dataset.yaml -> val image dir
static std::vector<std::string> resolve_dataset(const std::string& yaml) {
    std::ifstream f(yaml);
    std::string path, val, line;
    while (std::getline(f, line)) {
        auto kv = [&](const char* k, std::string& dst) {
            if (line.rfind(k, 0) == 0) {
                std::string v = line.substr(std::strlen(k));
                auto h = v.find('#'); if (h != std::string::npos) v = v.substr(0, h);
                size_t a = v.find_first_not_of(" \t"); size_t b = v.find_last_not_of(" \t\r");
                dst = (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
            }
        };
        kv("path:", path);
        kv("val:", val);
    }
    if (val.empty()) return {};
    const fs::path ydir = fs::path(yaml).parent_path();
    std::vector<fs::path> cands = {
        fs::path(val),                       // absolute val
        fs::path(path) / val,                // path/val (path absolute)
        ydir / path / val,                   // yaml_dir/path/val
        ydir / val,                          // yaml_dir/val
        fs::path("/data/datasets") / path / val,
    };
    std::error_code ec;
    for (auto& c : cands) {
        if (fs::is_directory(c, ec)) { std::vector<std::string> v; collect_dir(c.string(), v); if (!v.empty()) return v; }
        if (fs::is_regular_file(c, ec) && lower(c.extension().string()) == ".txt") {
            std::vector<std::string> v; std::ifstream tf(c); std::string l;
            while (std::getline(tf, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (!l.empty()) v.push_back(l); }
            if (!v.empty()) return v;
        }
    }
    return {};
}

std::vector<std::string> gather_images(const std::string& src, int limit) {
    std::vector<std::string> out;
    switch (classify_source(src)) {
        case SourceKind::Image:   out = {src}; break;
        case SourceKind::Dir:     collect_dir(src, out); break;
        case SourceKind::Dataset: out = resolve_dataset(src); break;
        default: break;
    }
    if (limit > 0 && static_cast<int>(out.size()) > limit) out.resize(limit);
    return out;
}

} // namespace yolomaster
