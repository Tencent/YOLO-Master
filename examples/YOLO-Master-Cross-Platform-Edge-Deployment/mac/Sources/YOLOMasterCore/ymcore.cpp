// C shim over the portable core (see include/ymcore.h).
#include "ymcore.h"
#include "bench_stats.hpp"
#include "metrics_core.hpp"
#include "tracker_core.hpp"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace yolomaster;

extern "C" {

int ym_core_version(void) { return 1; }

// ---- mAP -------------------------------------------------------------------------------------
void ym_map_evaluate(const YmImageEval* images, int n, YmMapResult* out) {
    std::vector<metrics::ImageEval> evals(static_cast<size_t>(n > 0 ? n : 0));
    for (int i = 0; i < n; ++i) {
        auto& e = evals[static_cast<size_t>(i)];
        e.preds.reserve(static_cast<size_t>(images[i].n_preds));
        for (int k = 0; k < images[i].n_preds; ++k) {
            const YmPredBox& p = images[i].preds[k];
            e.preds.push_back({p.x1, p.y1, p.x2, p.y2, p.conf, p.cls});
        }
        e.gts.reserve(static_cast<size_t>(images[i].n_gts));
        for (int k = 0; k < images[i].n_gts; ++k) {
            const YmGtBox& g = images[i].gts[k];
            e.gts.push_back({g.x1, g.y1, g.x2, g.y2, g.cls});
        }
    }
    const metrics::MapResult r = metrics::evaluate(evals);
    out->images = r.images; out->map50 = r.map50; out->map5095 = r.map5095;
    out->n_classes = static_cast<int>(r.per_class.size());
    out->per_class = out->n_classes ? static_cast<YmClassAP*>(std::calloc(static_cast<size_t>(out->n_classes), sizeof(YmClassAP))) : nullptr;
    for (int i = 0; i < out->n_classes; ++i) {
        const auto& c = r.per_class[static_cast<size_t>(i)];
        out->per_class[i].cls = c.cls; out->per_class[i].n_gt = c.n_gt; out->per_class[i].n_pred = c.n_pred;
        for (int k = 0; k < 10; ++k) out->per_class[i].ap[k] = c.ap[k];
    }
}
void ym_map_free(YmMapResult* r) {
    if (!r) return;
    std::free(r->per_class); r->per_class = nullptr; r->n_classes = 0;
}
double ym_round6(double v) { return metrics::round6(v); }

int ym_load_yolo_labels(const char* path, int img_w, int img_h, YmGtBox** out, int* n) {
    std::vector<metrics::GtBox> boxes;
    const bool ok = metrics::load_yolo_labels(path ? path : "", img_w, img_h, boxes);
    *n = static_cast<int>(boxes.size());
    *out = boxes.empty() ? nullptr : static_cast<YmGtBox*>(std::malloc(boxes.size() * sizeof(YmGtBox)));
    for (size_t i = 0; i < boxes.size(); ++i)
        (*out)[i] = YmGtBox{boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].cls};
    return ok ? 1 : 0;
}
void ym_gt_free(YmGtBox* boxes) { std::free(boxes); }

size_t ym_label_path_for(const char* image_path, const char* labels_dir, char* buf, size_t cap) {
    const std::string p = metrics::label_path_for(image_path ? image_path : "", labels_dir ? labels_dir : "");
    if (!buf || cap == 0 || p.size() + 1 > cap) return 0;
    std::memcpy(buf, p.c_str(), p.size() + 1);
    return p.size();
}

// ---- bench statistics --------------------------------------------------------------------------
void ym_stats_reduce(const double* samples, int n, YmStageStats* out) {
    const bench::StageStats s = bench::reduce(std::vector<double>(samples, samples + (n > 0 ? n : 0)));
    *out = YmStageStats{s.n, s.mean, s.median, s.p90, s.p95, s.p99, s.min, s.max};
}
void ym_sustained_summary(const double* samples, int n, int cold_iters, YmSustained* out) {
    const bench::SustainedSummary s = bench::summarize_sustained(std::vector<double>(samples, samples + (n > 0 ? n : 0)), cold_iters);
    *out = YmSustained{s.cold_median_ms, s.sustained_median_ms, s.throttle_pct};
}
void ym_sha256_hex(const uint8_t* data, size_t len, char out[65]) {
    const std::string h = bench::sha256_hex(data && len ? std::string(reinterpret_cast<const char*>(data), len) : std::string());
    std::memcpy(out, h.c_str(), 65);
}
void ym_image_list_sha256(const char* const* paths, int n, char out[65]) {
    std::vector<std::string> v;
    for (int i = 0; paths && i < n; ++i) v.emplace_back(paths[i] ? paths[i] : "");
    const std::string h = bench::image_list_sha256(v);
    std::memcpy(out, h.c_str(), 65);
}
void ym_timestamp_utc(char out[32]) {
    const std::string t = bench::timestamp_utc();
    std::snprintf(out, 32, "%s", t.c_str());
}

// ---- tracking ----------------------------------------------------------------------------------
struct YmTracker {
    track::core::CoreTracker core;
    std::vector<YmTrack> last;
    explicit YmTracker(const track::core::CoreConfig& c) : core(c) {}
};

void ym_tracker_default_config(YmTrackerConfig* out) {
    const track::core::CoreConfig d;
    *out = YmTrackerConfig{d.botsort ? 1 : 0, d.track_high_thresh, d.track_low_thresh, d.new_track_thresh,
                           d.match_thresh, d.track_buffer, d.fuse_score ? 1 : 0, d.fps};
}
YmTracker* ym_tracker_new(const YmTrackerConfig* cfg) {
    track::core::CoreConfig c;
    if (cfg) {
        c.botsort = cfg->botsort != 0;
        c.track_high_thresh = cfg->track_high_thresh; c.track_low_thresh = cfg->track_low_thresh;
        c.new_track_thresh = cfg->new_track_thresh; c.match_thresh = cfg->match_thresh;
        c.track_buffer = cfg->track_buffer; c.fuse_score = cfg->fuse_score != 0; c.fps = cfg->fps;
    }
    return new YmTracker(c);
}
void ym_tracker_update(YmTracker* t, const YmTrackInput* dets, int n, const YmMotion* motion,
                       const YmTrack** out, int* n_out) {
    std::vector<track::core::TrackInput> in(static_cast<size_t>(n > 0 ? n : 0));
    for (int i = 0; i < n; ++i) {
        auto& d = in[static_cast<size_t>(i)];
        d.box = track::core::Box{dets[i].box.x, dets[i].box.y, dets[i].box.width, dets[i].box.height};
        d.conf = dets[i].conf; d.class_id = dets[i].class_id;
        if (dets[i].mask_coeffs && dets[i].n_mask_coeffs > 0)
            d.mask_coeffs.assign(dets[i].mask_coeffs, dets[i].mask_coeffs + dets[i].n_mask_coeffs);
    }
    track::core::Motion m;
    if (motion) {
        m.R[0][0] = motion->r00; m.R[0][1] = motion->r01; m.R[1][0] = motion->r10; m.R[1][1] = motion->r11;
        m.tx = motion->tx; m.ty = motion->ty;
    }
    const std::vector<track::core::CoreTrack> tr = t->core.update(in, motion ? &m : nullptr);
    t->last.clear();
    t->last.reserve(tr.size());
    for (const auto& k : tr)
        t->last.push_back(YmTrack{k.id, k.class_id, k.conf, YmBox{k.box.x, k.box.y, k.box.width, k.box.height},
                                  k.age, k.hits, k.time_since_update, static_cast<int>(k.state), k.det_index});
    *out = t->last.data();
    *n_out = static_cast<int>(t->last.size());
}
void ym_tracker_reset(YmTracker* t) { t->core.reset(); }
int ym_tracker_frame_count(const YmTracker* t) { return t->core.frame_count(); }
void ym_tracker_free(YmTracker* t) { delete t; }

} // extern "C"
