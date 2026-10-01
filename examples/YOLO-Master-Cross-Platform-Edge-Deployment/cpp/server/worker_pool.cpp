#include "worker_pool.hpp"
#include "backend_factory.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <thread>

namespace yolomaster::server {

static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

cv::Mat decode_image(const std::string& bytes, int max_pixels, std::string& err) {
    int w = 0, h = 0, n = 0;
    if (!stbi_info_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), int(bytes.size()), &w, &h, &n)) {
        err = "unsupported or corrupt image"; return {};
    }
    if (w <= 0 || h <= 0 || int64_t(w) * h > max_pixels) {
        err = "image too large: " + std::to_string(w) + "x" + std::to_string(h) + " (max_pixels=" + std::to_string(max_pixels) + ")";
        return {};
    }
    unsigned char* d = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), int(bytes.size()), &w, &h, &n, 3);
    if (!d) { err = std::string("decode failed: ") + (stbi_failure_reason() ? stbi_failure_reason() : "?"); return {}; }
    cv::Mat bgr;
    cv::cvtColor(cv::Mat(h, w, CV_8UC3, d), bgr, cv::COLOR_RGB2BGR);
    stbi_image_free(d);
    return bgr;
}

std::vector<unsigned char> encode_jpg(const cv::Mat& bgr, int quality) {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    std::vector<unsigned char> out; out.reserve(size_t(rgb.total()) / 4);
    auto sink = [](void* ctx, void* data, int size) {
        auto* v = static_cast<std::vector<unsigned char>*>(ctx);
        v->insert(v->end(), static_cast<unsigned char*>(data), static_cast<unsigned char*>(data) + size);
    };
    stbi_write_jpg_to_func(sink, &out, rgb.cols, rgb.rows, 3, rgb.data, std::max(1, std::min(100, quality)));
    return out;
}

WorkerPool::WorkerPool(const ModelSpec& spec, const ServerConfig& cfg) : spec_(spec), cfg_(cfg) {
    int n = spec_.workers;
    if (n <= 0) {
        const bool gpu = spec_.backend == "trt" || spec_.device == "cuda" || spec_.device == "gpu" ||
                         spec_.device == "vulkan" || spec_.device == "opencl";
        const int hw = int(std::thread::hardware_concurrency());
        n = gpu ? 1 : std::max(1, std::min(4, hw / std::max(1, spec_.threads)));
    }
    for (int i = 0; i < n; ++i) threads_.emplace_back([this, i] { worker_main(i); });
}

WorkerPool::~WorkerPool() { shutdown(0); }

std::unique_ptr<Backend> WorkerPool::make(std::string& err) const {
    std::string resolved;
    Precision prec = Precision::Auto;
    if (!parse_precision(spec_.precision, prec)) { err = "bad precision: " + spec_.precision; return nullptr; }
    auto be = make_backend(spec_.path, spec_.backend, spec_.threads, spec_.device, resolved, err,
                           prec, cfg_.engine_cache_dir, spec_.preproc != "cpu", spec_.cuda_graph);
    if (be) const_cast<WorkerPool*>(this)->resolved_backend_ = resolved;
    return be;
}

Config WorkerPool::base_config(const Backend& be) const {
    Config c;
    c.conf_thresh = spec_.conf; c.iou_thresh = spec_.iou; c.max_det = spec_.max_det;
    c.multi_label = spec_.multi_label; c.stretch = spec_.stretch;
    int want = spec_.imgsz > 0 ? spec_.imgsz : (be.meta_imgsz > 0 ? be.meta_imgsz : 640);
    if (be.fixed_imgsz > 0) want = be.fixed_imgsz;
    c.imgsz = want;
    if (spec_.classes == "visdrone") c.class_names = visdrone_classes();
    else if (spec_.classes == "sku" || spec_.classes == "sku110k") c.class_names = sku110k_classes();
    else if (!be.meta_names.empty()) c.class_names = be.meta_names;
    else c.class_names = visdrone_classes();
    return c;
}

void WorkerPool::worker_main(int wid) {
    std::string err;
    std::unique_ptr<Backend> be = make(err);
    if (!be) {
        std::lock_guard<std::mutex> g(m_);
        if (init_error_.empty()) init_error_ = err;
        std::cerr << "[pool " << spec_.id << "] worker " << wid << " init failed: " << err << "\n";
        cv_.notify_all();
        return;
    }
    Config base = base_config(*be);
    // warm-up: one gray forward absorbs lazy allocations / TRT context setup
    try {
        cv::Mat gray(base.imgsz, base.imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
        (void)be->infer(gray, base);
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> g(m_);
        if (init_error_.empty()) init_error_ = std::string("warm-up failed: ") + e.what();
        std::cerr << "[pool " << spec_.id << "] worker " << wid << " warm-up failed: " << e.what() << "\n";
        cv_.notify_all();
        return;
    }
    {
        std::lock_guard<std::mutex> g(m_);
        if (active_ep_.empty()) {
            active_ep_ = be->active_ep; ep_note_ = be->ep_note; imgsz_ = base.imgsz;
            nc_ = base.num_classes(); is_seg_ = be->is_seg(); names_ = base.class_names;
        }
    }
    if (++ready_workers_ == 1) {
        ready_ = true;
        std::cerr << "[pool " << spec_.id << "] ready: backend=" << resolved_backend_ << " ep=" << be->active_ep
                  << " imgsz=" << base.imgsz << " nc=" << base.num_classes()
                  << (be->ep_note.empty() ? "" : " note=" + be->ep_note) << "\n";
    }

    int consecutive_failures = 0;
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_.load() || !queue_.empty(); });
            if (queue_.empty()) { if (stop_) return; else continue; }
            job = std::move(queue_.front()); queue_.pop_front();
            metrics.queue_depth.fetch_sub(1);
        }
        metrics.busy_workers.fetch_add(1);
        InferResult r;
        try {
            r = run_job(*be, base, wid, job);
            consecutive_failures = 0;
        } catch (const std::exception& e) {
            r.http_status = 500; r.error = std::string("inference error: ") + e.what();
            if (++consecutive_failures >= 3) {
                // a wedged backend (CUDA context lost, etc.) gets rebuilt once
                std::cerr << "[pool " << spec_.id << "] worker " << wid << ": 3 consecutive failures, rebuilding backend\n";
                std::string e2; auto nb = make(e2);
                if (nb) { be = std::move(nb); base = base_config(*be); consecutive_failures = 0; }
                else std::cerr << "[pool " << spec_.id << "] rebuild failed: " << e2 << "\n";
            }
        }
        metrics.busy_workers.fetch_sub(1);
        metrics.count_code(r.http_status);
        if (job.done) job.done(std::move(r));
    }
}

InferResult WorkerPool::run_job(Backend& be, const Config& base, int wid, Job& job) {
    InferResult r; r.worker_id = wid;
    const auto t_start = Clock::now();
    r.queue_ms = ms_between(job.enqueued, t_start);
    metrics.queue.observe(r.queue_ms);
    if (t_start > job.deadline) {
        r.http_status = 504; r.error = "deadline exceeded while queued (" + std::to_string(int(r.queue_ms)) + " ms)";
        return r;
    }
    // ---- bench job: probe run on this worker's backend, no image ----
    if (job.bench) {
        bench::BenchResult br;
        br.tool = "server";
        br.timestamp = bench::timestamp_utc();
        br.has_cold = true;
        br.cold = bench::cold_run(be, base, job.bench->warmup, job.bench->iters);
        br.model = bench::model_info(be, spec_.path, spec_.backend, spec_.precision, base);
        br.model.id = spec_.id;
        br.env = bench::collect_env(be, spec_.threads);
        br.protocol.mode = "cold"; br.protocol.conf = base.conf_thresh; br.protocol.iou = base.iou_thresh;
        br.protocol.max_det = base.max_det; br.protocol.multi_label = base.multi_label;
        br.protocol.slicing = spec_.slicing.empty() ? "off" : spec_.slicing; br.protocol.tile_size = spec_.tile_size;
        br.protocol.warmup = job.bench->warmup; br.protocol.iters = job.bench->iters;
        br.protocol.probe_mode = br.cold.probe_mode;
        r.bench_json = bench::to_json(br);
        r.bench_json["worker"] = wid;
        r.active_ep = be.active_ep; r.cfg_used = base;
        r.total_ms = ms_between(t_start, Clock::now());
        return r;
    }
    // ---- decode ----
    cv::Mat bgr;
    if (job.raw_w > 0) {
        if (job.image.size() != size_t(job.raw_w) * job.raw_h * 3) { r.http_status = 400; r.error = "raw frame size mismatch"; return r; }
        bgr = cv::Mat(job.raw_h, job.raw_w, CV_8UC3, job.image.data()).clone();
    } else {
        std::string err;
        bgr = decode_image(job.image, cfg_.max_pixels, err);
        if (bgr.empty()) { r.http_status = 400; r.error = err; return r; }
    }
    metrics.images_decoded.fetch_add(1);
    const auto t_dec = Clock::now();
    r.decode_ms = ms_between(t_start, t_dec);
    metrics.decode.observe(r.decode_ms);
    r.orig_w = bgr.cols; r.orig_h = bgr.rows;

    // ---- per-request config ----
    Config cfg = base;
    const InferParams& p = job.params;
    if (p.conf >= 0) cfg.conf_thresh = std::min(1.0f, std::max(0.0f, p.conf));
    if (p.iou >= 0) cfg.iou_thresh = std::min(1.0f, std::max(0.0f, p.iou));
    if (p.max_det > 0) cfg.max_det = std::min(3000, p.max_det);
    if (p.multi_label >= 0) cfg.multi_label = p.multi_label != 0;
    // tracking wants the low-score detections for its second association (the CLI does the same)
    if (job.tracker && p.conf < 0) cfg.conf_thresh = job.tracker->config().track_low_thresh;
    std::string slicing = p.slicing.empty() ? spec_.slicing : p.slicing;
    SliceConfig sc;
    sc.mode = slicing == "dense" ? SliceMode::Dense : slicing == "sparse" ? SliceMode::Sparse : SliceMode::Off;
    sc.tile_size = p.tile_size > 0 ? p.tile_size : spec_.tile_size;
    sc.keep_global_masks = p.mask_overlay;

    // ---- forward + post ----
    if (sc.mode != SliceMode::Off) {
        const SliceOutput so = sliced_candidates(be, bgr, cfg, sc);
        const auto t_nms = Clock::now();
        r.dets = nms_and_cap(be.candidates, cfg, bgr.cols, bgr.rows);
        r.tiles_run = so.tiles_run; r.tiles_total = so.tiles_total;
        r.pre_ms = be.pre_ms; r.infer_ms = so.infer_ms;
        r.post_ms = be.post_ms + ms_between(t_nms, Clock::now());
    } else {
        r.dets = be.infer(bgr, cfg);
        r.pre_ms = be.pre_ms; r.infer_ms = be.infer_ms; r.post_ms = be.post_ms;
    }
    r.active_ep = be.active_ep; r.is_seg = be.is_seg(); r.cfg_used = cfg;
    // ---- tracking: detections -> confirmed tracks (ids parallel to dets) ----
    std::vector<track::Track> tracks;
    if (job.tracker) {
        tracks = job.tracker->update(r.dets, &bgr);
        std::vector<Detection> tracked;
        tracked.reserve(tracks.size());
        for (const auto& t : tracks) {
            Detection d;
            d.class_id = t.class_id; d.conf = t.conf; d.box = t.box; d.mask_coeffs = t.mask_coeffs;
            tracked.push_back(std::move(d));
            r.track_ids.push_back(t.id);
        }
        r.dets.swap(tracked);
    }
    if (!p.mask_coeffs) for (auto& d : r.dets) d.mask_coeffs.clear();
    metrics.pre.observe(r.pre_ms); metrics.infer.observe(r.infer_ms); metrics.post.observe(r.post_ms);
    metrics.infer_ring.add(r.infer_ms);

    // ---- optional annotated image ----
    if (p.annotated) {
        const auto t_enc = Clock::now();
        cv::Mat vis = bgr.clone();
        if (p.mask_overlay && be.is_seg()) {
            cv::Mat ov = seg_overlay(r.dets, be.proto, be.proto_c, be.proto_h, be.proto_w, be.cand_lb, cfg.imgsz, bgr.cols, bgr.rows);
            for (int y = 0; y < vis.rows; ++y) {
                const uint8_t* o = ov.ptr<uint8_t>(y); uint8_t* v = vis.ptr<uint8_t>(y);
                for (int x = 0; x < vis.cols; ++x) {
                    const int a = o[x * 4 + 3]; if (!a) continue;
                    for (int c = 0; c < 3; ++c) v[x * 3 + c] = uint8_t((v[x * 3 + c] * (255 - a) + o[x * 4 + (2 - c)] * a) / 255);
                }
            }
        }
        if (job.tracker) track::draw_tracks(vis, tracks, cfg); else draw(vis, r.dets, cfg);
        r.annotated_jpg = encode_jpg(vis, p.jpeg_quality);
        r.encode_ms = ms_between(t_enc, Clock::now());
        metrics.encode.observe(r.encode_ms);
    }
    r.total_ms = ms_between(t_start, Clock::now());   // server-side, excluding queue wait
    metrics.total.observe(r.total_ms);
    metrics.total_ring.add(r.total_ms + r.queue_ms);
    return r;
}

bool WorkerPool::submit(Job&& job) {
    std::lock_guard<std::mutex> g(m_);
    if (stop_ || int(queue_.size()) >= cfg_.max_queue) return false;
    queue_.push_back(std::move(job));
    metrics.queue_depth.fetch_add(1);
    cv_.notify_one();
    return true;
}

void WorkerPool::shutdown(int drain_timeout_ms) {
    if (stop_.exchange(true)) { for (auto& t : threads_) if (t.joinable()) t.join(); return; }
    // drain: wait for queued work up to the timeout, then fail the rest fast
    const auto until = Clock::now() + std::chrono::milliseconds(drain_timeout_ms);
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_until(lk, until, [&] { return queue_.empty(); });
        std::deque<Job> rest; rest.swap(queue_);
        metrics.queue_depth.store(0);
        lk.unlock();
        for (auto& j : rest) if (j.done) { InferResult r; r.http_status = 503; r.error = "server shutting down"; j.done(std::move(r)); }
    }
    cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
}

nlohmann::json WorkerPool::info() const {
    std::lock_guard<std::mutex> g(m_);
    nlohmann::json j = {{"id", spec_.id}, {"path", spec_.path}, {"backend", resolved_backend_.empty() ? spec_.backend : resolved_backend_},
                        {"device", spec_.device}, {"precision", spec_.precision}, {"threads", spec_.threads},
                        {"workers", int(threads_.size())}, {"ready", ready_.load()}, {"ep", active_ep_},
                        {"ep_note", ep_note_}, {"imgsz", imgsz_}, {"nc", nc_}, {"seg", is_seg_},
                        {"conf", spec_.conf}, {"iou", spec_.iou}, {"max_det", spec_.max_det},
                        {"queue_depth", metrics.queue_depth.load()}, {"busy_workers", metrics.busy_workers.load()},
                        {"requests_ok", metrics.requests_ok.load()}, {"requests_err", metrics.requests_err.load()}};
    if (!init_error_.empty()) j["error"] = init_error_;
    if (!names_.empty()) j["names"] = names_;
    return j;
}

} // namespace yolomaster::server
