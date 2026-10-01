// One WorkerPool per loaded model: W worker threads, each owning ONE Backend instance
// (Backend::infer keeps per-call state in members, so an instance never leaves its thread).
// Jobs are image bytes + per-request parameters; the pool decodes, infers, serializes and
// hands an InferResult to the completion callback on the worker thread. HTTP code re-posts
// it to the event loop with uWS::Loop::defer.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "yolomaster.hpp"
#include "slicing.hpp"
#include "tracker.hpp"
#include "bench.hpp"
#include "server_config.hpp"
#include "metrics.hpp"

namespace yolomaster::server {

using Clock = std::chrono::steady_clock;

struct InferParams {
    float conf = -1, iou = -1;          // <0 = model default
    int max_det = -1;
    int multi_label = -1;               // -1 default, 0/1 override
    std::string slicing;                // "" = model default; off|dense|sparse
    int tile_size = -1;
    bool annotated = false;             // encode an annotated JPEG
    bool mask_overlay = false;          // seg: composite masks into the annotated image
    bool mask_coeffs = false;           // seg: include raw mask coefficients in JSON
    int jpeg_quality = 90;
    std::string track;                  // "" | botsort | bytetrack (video / stream jobs carry a Tracker)
};

struct BenchRequest { int warmup = 10, iters = 50; };

struct InferResult {
    int http_status = 200;
    std::string error;                  // non-empty on failure
    std::vector<Detection> dets;
    int orig_w = 0, orig_h = 0;
    double queue_ms = 0, decode_ms = 0, pre_ms = 0, infer_ms = 0, post_ms = 0, encode_ms = 0, total_ms = 0;
    std::string active_ep;
    bool is_seg = false;
    int tiles_run = 0, tiles_total = 0;
    int worker_id = -1;
    std::vector<unsigned char> annotated_jpg;
    Config cfg_used;                    // conf/iou/imgsz/class names actually applied
    std::vector<int> track_ids;         // parallel to dets when the job carried a tracker
    nlohmann::json bench_json;          // bench jobs: the yolomaster-bench/v1 document
};

struct Job {
    std::string request_id;
    std::string image;                  // encoded bytes (jpg/png/bmp/...) or raw BGR when raw_w > 0
    int raw_w = 0, raw_h = 0;           // raw BGR8 frame (WS video path) instead of an encoded image
    InferParams params;
    std::shared_ptr<track::Tracker> tracker;   // owned by the video request / WS session; one job in flight
    std::shared_ptr<BenchRequest> bench;       // bench job: no image, runs bench::cold_run on this worker
    Clock::time_point enqueued;
    Clock::time_point deadline;
    std::function<void(InferResult&&)> done;
};

class WorkerPool {
public:
    WorkerPool(const ModelSpec& spec, const ServerConfig& cfg);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // Returns false when the queue is full (caller answers 503) or the pool is stopping.
    bool submit(Job&& job);
    void shutdown(int drain_timeout_ms);

    bool ready() const { return ready_.load(); }
    bool failed() const { return !init_error_.empty(); }
    std::string init_error() const { std::lock_guard<std::mutex> g(m_); return init_error_; }
    const ModelSpec& spec() const { return spec_; }
    int workers() const { return int(threads_.size()); }
    nlohmann::json info() const;        // model card: id, backend, ep, imgsz, nc, workers, ...
    ModelMetrics metrics;

private:
    void worker_main(int wid);
    InferResult run_job(Backend& be, const Config& base_cfg, int wid, Job& job);
    std::unique_ptr<Backend> make(std::string& err) const;
    Config base_config(const Backend& be) const;

    ModelSpec spec_;
    ServerConfig cfg_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_{false}, ready_{false};
    std::atomic<int> ready_workers_{0};
    std::string init_error_;
    // model card (filled by the first worker that loads)
    std::string active_ep_, ep_note_, resolved_backend_;
    int imgsz_ = 0, nc_ = 0; bool is_seg_ = false;
    std::vector<std::string> names_;
};

// stb-based decoders shared with the CLI semantics (3-channel, BGR)
cv::Mat decode_image(const std::string& bytes, int max_pixels, std::string& err);
std::vector<unsigned char> encode_jpg(const cv::Mat& bgr, int quality);

} // namespace yolomaster::server
