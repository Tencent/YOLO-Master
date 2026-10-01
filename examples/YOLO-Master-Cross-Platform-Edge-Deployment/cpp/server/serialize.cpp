#include "serialize.hpp"
#include <sstream>
#include <iomanip>
#ifdef HAVE_CUDART
#include <cuda_runtime_api.h>
#endif

namespace yolomaster::server {

int coco80_to_91(int c) {
    static const int m[80] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,
                              37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,
                              72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
    return (c >= 0 && c < 80) ? m[c] : c;
}

static double r3(double v) { return std::round(v * 1000.0) / 1000.0; }

nlohmann::json result_json(const InferResult& r, const std::string& model_id, const std::string& request_id,
                           bool include_names) {
    nlohmann::json dets = nlohmann::json::array();
    for (size_t i = 0; i < r.dets.size(); ++i) {
        const auto& d = r.dets[i];
        // full float precision: the JSON must round-trip to the CLI's txt numbers for parity checks
        nlohmann::json o = {{"class_id", d.class_id}, {"conf", d.conf},
                            {"box", {d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height}}};
        if (i < r.track_ids.size()) o["track_id"] = r.track_ids[i];
        if (include_names && d.class_id >= 0 && d.class_id < int(r.cfg_used.class_names.size()))
            o["name"] = r.cfg_used.class_names[d.class_id];
        if (!d.mask_coeffs.empty()) o["mask_coeffs"] = d.mask_coeffs;
        dets.push_back(std::move(o));
    }
    nlohmann::json j = {
        {"request_id", request_id}, {"model", model_id}, {"backend", r.active_ep},
        {"image", {{"width", r.orig_w}, {"height", r.orig_h}}},
        {"params", {{"conf", r.cfg_used.conf_thresh}, {"iou", r.cfg_used.iou_thresh}, {"max_det", r.cfg_used.max_det},
                    {"imgsz", r.cfg_used.imgsz}, {"multi_label", r.cfg_used.multi_label}}},
        {"count", r.dets.size()}, {"detections", std::move(dets)},
        {"timings_ms", {{"queue", r3(r.queue_ms)}, {"decode", r3(r.decode_ms)}, {"pre", r3(r.pre_ms)},
                        {"infer", r3(r.infer_ms)}, {"post", r3(r.post_ms)}, {"encode", r3(r.encode_ms)},
                        {"total", r3(r.total_ms)}}},
        {"seg", r.is_seg}, {"worker", r.worker_id}};
    if (r.tiles_total > 0) j["slicing"] = {{"tiles_run", r.tiles_run}, {"tiles_total", r.tiles_total}};
    return j;
}

std::string result_txt(const InferResult& r) {
    std::ostringstream s;
    for (const auto& d : r.dets)
        s << d.class_id << ' ' << d.conf << ' ' << d.box.x << ' ' << d.box.y << ' '
          << (d.box.x + d.box.width) << ' ' << (d.box.y + d.box.height) << '\n';
    return s.str();
}

nlohmann::json result_coco(const InferResult& r, int image_id, bool coco91) {
    nlohmann::json a = nlohmann::json::array();
    for (const auto& d : r.dets)
        a.push_back({{"image_id", image_id}, {"category_id", coco91 ? coco80_to_91(d.class_id) : d.class_id},
                     {"bbox", {r3(d.box.x), r3(d.box.y), r3(d.box.width), r3(d.box.height)}}, {"score", r3(double(d.conf))}});
    return a;
}

std::string prometheus_text(ModelRegistry& reg, const ServerConfig& cfg, double uptime_s, const GlobalCounters* gc) {
    std::string o;
    o += "# HELP yolomaster_uptime_seconds Seconds since the server started.\n# TYPE yolomaster_uptime_seconds gauge\n";
    o += "yolomaster_uptime_seconds " + std::to_string(uptime_s) + "\n";
    o += "# HELP yolomaster_rate_limited_total Requests answered 429 by the token-bucket limiter.\n# TYPE yolomaster_rate_limited_total counter\n";
    o += "yolomaster_rate_limited_total " + std::to_string(gc ? gc->rate_limited.load() : 0) + "\n";
    o += "# HELP yolomaster_auth_failed_total Requests answered 401 (missing or invalid API key).\n# TYPE yolomaster_auth_failed_total counter\n";
    o += "yolomaster_auth_failed_total " + std::to_string(gc ? gc->auth_failed.load() : 0) + "\n";
    o += "# HELP yolomaster_requests_total Inference requests by model and HTTP status.\n# TYPE yolomaster_requests_total counter\n";
    for (WorkerPool* p : reg.pools()) {
        std::lock_guard<std::mutex> g(p->metrics.codes_m);
        for (auto& kv : p->metrics.codes)
            o += "yolomaster_requests_total{model=\"" + p->spec().id + "\",code=\"" + std::to_string(kv.first) + "\"} " + std::to_string(kv.second) + "\n";
    }
    o += "# HELP yolomaster_queue_depth Jobs waiting per model.\n# TYPE yolomaster_queue_depth gauge\n";
    for (WorkerPool* p : reg.pools())
        o += "yolomaster_queue_depth{model=\"" + p->spec().id + "\"} " + std::to_string(p->metrics.queue_depth.load()) + "\n";
    o += "# HELP yolomaster_busy_workers Workers currently inferring per model.\n# TYPE yolomaster_busy_workers gauge\n";
    for (WorkerPool* p : reg.pools())
        o += "yolomaster_busy_workers{model=\"" + p->spec().id + "\"} " + std::to_string(p->metrics.busy_workers.load()) + "\n";
    o += "# HELP yolomaster_workers Worker threads per model.\n# TYPE yolomaster_workers gauge\n";
    for (WorkerPool* p : reg.pools())
        o += "yolomaster_workers{model=\"" + p->spec().id + "\"} " + std::to_string(p->workers()) + "\n";
    o += "# HELP yolomaster_max_queue Queue capacity before 503.\n# TYPE yolomaster_max_queue gauge\nyolomaster_max_queue " + std::to_string(cfg.max_queue) + "\n";
    o += "# HELP yolomaster_stage_seconds Per-stage latency histograms (seconds).\n# TYPE yolomaster_stage_seconds histogram\n";
    for (WorkerPool* p : reg.pools()) {
        const std::string id = "model=\"" + p->spec().id + "\"";
        auto& m = p->metrics;
        m.queue.render(o, "yolomaster_stage_seconds", id + ",stage=\"queue\"");
        m.decode.render(o, "yolomaster_stage_seconds", id + ",stage=\"decode\"");
        m.pre.render(o, "yolomaster_stage_seconds", id + ",stage=\"pre\"");
        m.infer.render(o, "yolomaster_stage_seconds", id + ",stage=\"infer\"");
        m.post.render(o, "yolomaster_stage_seconds", id + ",stage=\"post\"");
        m.encode.render(o, "yolomaster_stage_seconds", id + ",stage=\"encode\"");
        m.total.render(o, "yolomaster_stage_seconds", id + ",stage=\"total\"");
    }
    o += "# HELP yolomaster_ws_dropped_frames_total Stale WebSocket frames dropped by keep-latest backpressure.\n# TYPE yolomaster_ws_dropped_frames_total counter\n";
    for (WorkerPool* p : reg.pools())
        o += "yolomaster_ws_dropped_frames_total{model=\"" + p->spec().id + "\"} " + std::to_string(p->metrics.dropped_frames.load()) + "\n";
#ifdef HAVE_CUDART
    size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) == cudaSuccess) {
        o += "# HELP yolomaster_gpu_memory_bytes GPU memory (device 0).\n# TYPE yolomaster_gpu_memory_bytes gauge\n";
        o += "yolomaster_gpu_memory_bytes{kind=\"used\"} " + std::to_string(totalb - freeb) + "\n";
        o += "yolomaster_gpu_memory_bytes{kind=\"total\"} " + std::to_string(totalb) + "\n";
    }
#endif
    return o;
}

nlohmann::json stats_json(ModelRegistry& reg, double uptime_s, const GlobalCounters* gc) {
    nlohmann::json models = nlohmann::json::object();
    for (WorkerPool* p : reg.pools()) {
        auto& m = p->metrics;
        models[p->spec().id] = {
            {"ready", p->ready()}, {"workers", p->workers()}, {"queue_depth", m.queue_depth.load()},
            {"busy_workers", m.busy_workers.load()}, {"requests_ok", m.requests_ok.load()}, {"requests_err", m.requests_err.load()},
            {"images_decoded", m.images_decoded.load()}, {"ws_dropped_frames", m.dropped_frames.load()},
            {"latency_ms", {{"1m", m.total_ring.percentiles(60)}, {"5m", m.total_ring.percentiles(300)}}},
            {"infer_ms", {{"1m", m.infer_ring.percentiles(60)}, {"5m", m.infer_ring.percentiles(300)}}},
            {"mean_ms", {{"queue", m.queue.count ? m.queue.sum / m.queue.count : 0.0},
                         {"decode", m.decode.count ? m.decode.sum / m.decode.count : 0.0},
                         {"pre", m.pre.count ? m.pre.sum / m.pre.count : 0.0},
                         {"infer", m.infer.count ? m.infer.sum / m.infer.count : 0.0},
                         {"post", m.post.count ? m.post.sum / m.post.count : 0.0},
                         {"total", m.total.count ? m.total.sum / m.total.count : 0.0}}}};
    }
    return {{"uptime_s", uptime_s}, {"models", models},
            {"server", {{"rate_limited", gc ? gc->rate_limited.load() : 0}, {"auth_failed", gc ? gc->auth_failed.load() : 0}}}};
}

} // namespace yolomaster::server
