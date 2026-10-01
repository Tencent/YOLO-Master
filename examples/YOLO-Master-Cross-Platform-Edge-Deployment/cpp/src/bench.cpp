// Backend-facing half of the benchmark mode. The statistics, hashes and timestamp are the portable
// mac/Sources/YOLOMasterCore/bench_stats.* shared with the Swift package.
#include "bench.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numeric>
#include <thread>
#ifndef _WIN32
#include <sys/utsname.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace yolomaster::bench {

using clk = std::chrono::steady_clock;

nlohmann::json to_json(const StageStats& s) {
    return {{"n", s.n}, {"mean", s.mean}, {"median", s.median}, {"p90", s.p90}, {"p95", s.p95},
            {"p99", s.p99}, {"min", s.min}, {"max", s.max}};
}

void Samples::add(const Backend& be, size_t ndets) {
    pre.push_back(be.pre_ms); infer.push_back(be.infer_ms); post.push_back(be.post_ms);
    total.push_back(be.pre_ms + be.infer_ms + be.post_ms);
    frames++; total_dets += static_cast<long>(ndets);
}

static cv::Mat probe_image(int imgsz) {
    return cv::Mat(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
}

// forward_raw(decode=false) is the phone Bench tab's inferOnly; backends without it (TensorRT,
// MNN) fall back to a full infer() and say so.
static bool probe_once(Backend& be, const cv::Mat& probe, const Config& cfg, bool& infer_only) {
    if (infer_only) {
        try { be.forward_raw(probe, cfg, /*decode=*/false); return true; }
        catch (const std::exception&) { infer_only = false; }
    }
    try { (void)be.infer(probe, cfg); return true; } catch (const std::exception&) { return false; }
}

ColdResult cold_run(Backend& be, const Config& cfg, int warmup, int iters) {
    ColdResult r;
    const cv::Mat probe = probe_image(cfg.imgsz);
    bool infer_only = true;
    for (int i = 0; i < warmup; ++i) if (!probe_once(be, probe, cfg, infer_only)) break;
    std::vector<double> v; v.reserve(std::max(iters, 0));
    for (int i = 0; i < iters; ++i) {
        if (!probe_once(be, probe, cfg, infer_only)) break;
        v.push_back(be.infer_ms);
    }
    r.infer_ms = reduce(v);
    r.probe_mode = infer_only ? "infer_only" : "full";
    return r;
}

SustainedResult sustained_loop(Backend& be, const Config& cfg, int warmup, double minutes,
                               int cold_iters, const std::atomic<bool>* cancel) {
    SustainedResult r;
    const cv::Mat probe = probe_image(cfg.imgsz);
    bool infer_only = true;
    for (int i = 0; i < warmup; ++i) if (!probe_once(be, probe, cfg, infer_only)) break;
    std::vector<double> all, second;
    const auto t0 = clk::now();
    auto sec_start = t0;
    const double budget_s = minutes * 60.0;
    while (true) {
        if (cancel && cancel->load()) break;
        const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();
        if (elapsed >= budget_s && static_cast<int>(all.size()) >= cold_iters) break;
        if (!probe_once(be, probe, cfg, infer_only)) break;
        all.push_back(be.infer_ms); second.push_back(be.infer_ms);
        if (std::chrono::duration<double>(clk::now() - sec_start).count() >= 1.0) {
            r.sparkline.push_back(reduce(second).median);
            second.clear(); sec_start = clk::now();
        }
    }
    if (!second.empty()) r.sparkline.push_back(reduce(second).median);
    r.duration_s = std::chrono::duration<double>(clk::now() - t0).count();
    r.infer_ms = reduce(all);
    const SustainedSummary s = summarize_sustained(all, cold_iters);
    r.cold_median_ms = s.cold_median_ms; r.sustained_median_ms = s.sustained_median_ms; r.throttle_pct = s.throttle_pct;
    r.probe_mode = infer_only ? "infer_only" : "full";
    return r;
}

static std::string read_cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
            const auto c = line.find(':');
            if (c != std::string::npos) {
                std::string v = line.substr(c + 1);
                const auto b = v.find_first_not_of(" \t");
                return b == std::string::npos ? "" : v.substr(b);
            }
        }
    }
    return "";
}

static std::string popen_line(const char* cmd) {
    std::string out;
#ifndef _WIN32
    FILE* p = popen(cmd, "r");
    if (!p) return out;
    char buf[256];
    if (fgets(buf, sizeof(buf), p)) out = buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
#endif
    return out;
}

EnvInfo collect_env(const Backend& be, int threads) {
    EnvInfo e;
#ifndef _WIN32
    char h[256] = {0};
    if (gethostname(h, sizeof(h) - 1) == 0) e.host = h;
    struct utsname u{};
    if (uname(&u) == 0) e.os = std::string(u.sysname) + " " + u.release + " " + u.machine;
#endif
    e.cpu_model = read_cpu_model();
    e.cpu_count = static_cast<int>(std::thread::hardware_concurrency());
#ifdef __linux__
    struct sysinfo si{};
    if (sysinfo(&si) == 0) e.mem_bytes = static_cast<long long>(si.totalram) * si.mem_unit;
#endif
    e.gpu_name = be.device_name();
    if (e.gpu_name.empty()) {
        const std::string ep = be.active_ep;
        if (ep.find("CUDA") != std::string::npos || ep.find("cuda") != std::string::npos ||
            ep.find("TRT") != std::string::npos || ep.find("trt") != std::string::npos ||
            ep.find("TensorRT") != std::string::npos)
            e.gpu_name = popen_line("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null");
    }
    e.threads = threads;
#ifdef YM_GIT_COMMIT
    e.git_commit = YM_GIT_COMMIT;
#endif
#ifdef YM_BUILD_FLAGS
    e.build_flags = YM_BUILD_FLAGS;
#endif
#ifdef YM_VERSION
    e.version = YM_VERSION;
#endif
    return e;
}

ModelInfo model_info(const Backend& be, const std::string& path, const std::string& backend,
                     const std::string& precision, const Config& cfg) {
    ModelInfo m;
    std::string p = path;
    while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    const auto slash = p.find_last_of("/\\");
    std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
    const auto dot = base.find_last_of('.');
    m.id = (dot == std::string::npos || dot == 0) ? base : base.substr(0, dot);
    m.path = path; m.backend = backend; m.runtime = be.runtime_name();
    m.execution_provider = be.active_ep; m.ep_note = be.ep_note; m.precision = precision;
    m.nc = cfg.num_classes(); m.is_seg = be.is_seg(); m.imgsz = cfg.imgsz;
    return m;
}

nlohmann::json to_json(const BenchResult& r) {
    nlohmann::json j;
    j["schema_version"] = kSchema;
    j["timestamp"] = r.timestamp.empty() ? timestamp_utc() : r.timestamp;
    j["tool"] = r.tool;
    j["model"] = {{"id", r.model.id}, {"path", r.model.path}, {"backend", r.model.backend},
                  {"runtime", r.model.runtime}, {"execution_provider", r.model.execution_provider},
                  {"ep_note", r.model.ep_note}, {"precision", r.model.precision}, {"nc", r.model.nc},
                  {"is_seg", r.model.is_seg}, {"imgsz", r.model.imgsz}};
    j["environment"] = {{"host", r.env.host}, {"os", r.env.os}, {"cpu_model", r.env.cpu_model},
                        {"cpu_count", r.env.cpu_count}, {"mem_bytes", r.env.mem_bytes}, {"gpu_name", r.env.gpu_name},
                        {"threads", r.env.threads}, {"git_commit", r.env.git_commit},
                        {"build_flags", r.env.build_flags}, {"version", r.env.version}};
    j["protocol"] = {{"mode", r.protocol.mode}, {"conf", r.protocol.conf}, {"iou", r.protocol.iou},
                     {"max_det", r.protocol.max_det}, {"multi_label", r.protocol.multi_label},
                     {"slicing", r.protocol.slicing}, {"tile_size", r.protocol.tile_size},
                     {"warmup", r.protocol.warmup}, {"iters", r.protocol.iters}, {"minutes", r.protocol.minutes},
                     {"probe", r.protocol.probe}, {"probe_mode", r.protocol.probe_mode},
                     {"dataset", r.protocol.dataset}, {"image_count", r.protocol.image_count},
                     {"image_list_sha256", r.protocol.image_list_sha256}};
    j["stats_convention"] = "floor_rank";
    if (r.has_cold) j["cold"] = {{"infer_ms", to_json(r.cold.infer_ms)}, {"probe_mode", r.cold.probe_mode}};
    if (r.has_sustained)
        j["sustained"] = {{"infer_ms", to_json(r.sustained.infer_ms)}, {"cold_median_ms", r.sustained.cold_median_ms},
                          {"sustained_median_ms", r.sustained.sustained_median_ms},
                          {"throttle_pct", r.sustained.throttle_pct}, {"sparkline", r.sustained.sparkline},
                          {"duration_s", r.sustained.duration_s}, {"probe_mode", r.sustained.probe_mode}};
    if (r.has_dataset)
        j["dataset"] = {{"frames", r.dataset.frames}, {"total_dets", r.dataset.total_dets},
                        {"pre_ms", to_json(r.dataset.pre_ms)}, {"infer_ms", to_json(r.dataset.infer_ms)},
                        {"post_ms", to_json(r.dataset.post_ms)}, {"total_ms", to_json(r.dataset.total_ms)},
                        {"model_fps", r.dataset.model_fps}, {"wall_s", r.dataset.wall_s}};
    if (r.accuracy.present) {
        nlohmann::json pc = nlohmann::json::array();
        for (const auto& c : r.accuracy.map.per_class)
            pc.push_back({{"class_id", c.cls}, {"n_gt", c.n_gt}, {"n_pred", c.n_pred},
                          {"ap50", c.ap50()}, {"ap5095", c.ap5095()}});
        j["accuracy"] = {{"protocol", {{"conf", r.accuracy.conf}, {"iou", r.accuracy.iou},
                                       {"max_det", r.accuracy.max_det}, {"multi_label", r.accuracy.multi_label}}},
                         {"labels", r.accuracy.labels}, {"images", r.accuracy.map.images},
                         {"map50", r.accuracy.map.map50}, {"map5095", r.accuracy.map.map5095},
                         {"per_class", pc}, {"timings", {{"infer_ms", to_json(r.accuracy.infer_ms)}}}};
    }
    return j;
}

} // namespace yolomaster::bench
