// Lock-light latency metrics: fixed-bucket histograms for Prometheus, a sample ring for
// rolling percentiles (/v1/stats), and per-model counters.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include "json.hpp"

namespace yolomaster::server {

struct Histogram {
    static constexpr std::array<double, 21> bounds{0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100,
                                                   150, 200, 300, 500, 1000, 2000, 5000, 10000};
    std::array<std::atomic<uint64_t>, bounds.size() + 1> buckets{};
    std::atomic<uint64_t> count{0};
    std::atomic<double> sum{0};
    void observe(double ms) {
        size_t i = 0;
        while (i < bounds.size() && ms > bounds[i]) ++i;
        buckets[i].fetch_add(1, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
        double cur = sum.load(std::memory_order_relaxed);
        while (!sum.compare_exchange_weak(cur, cur + ms, std::memory_order_relaxed)) {}
    }
    // Prometheus cumulative buckets
    void render(std::string& out, const std::string& name, const std::string& labels) const {
        uint64_t cum = 0;
        for (size_t i = 0; i < bounds.size(); ++i) {
            cum += buckets[i].load(std::memory_order_relaxed);
            out += name + "_bucket{" + labels + ",le=\"" + trim(bounds[i]) + "\"} " + std::to_string(cum) + "\n";
        }
        cum += buckets[bounds.size()].load(std::memory_order_relaxed);
        out += name + "_bucket{" + labels + ",le=\"+Inf\"} " + std::to_string(cum) + "\n";
        out += name + "_sum{" + labels + "} " + std::to_string(sum.load() / 1000.0) + "\n";   // seconds
        out += name + "_count{" + labels + "} " + std::to_string(count.load()) + "\n";
    }
    static std::string trim(double v) {   // "7.5" not "7.500000"; seconds for Prometheus
        char b[32]; std::snprintf(b, sizeof b, "%.4g", v / 1000.0); return b;
    }
};

// Rolling percentile window over the last N samples with timestamps (for /v1/stats).
class SampleRing {
public:
    explicit SampleRing(size_t n = 4096) : buf_(n) {}
    void add(double ms) {
        std::lock_guard<std::mutex> g(m_);
        buf_[head_] = {std::chrono::steady_clock::now(), ms};
        head_ = (head_ + 1) % buf_.size();
        if (size_ < buf_.size()) ++size_;
    }
    // percentiles over samples newer than `window_s` seconds; returns n=0 when empty
    nlohmann::json percentiles(double window_s) const {
        std::vector<double> v;
        {
            std::lock_guard<std::mutex> g(m_);
            const auto now = std::chrono::steady_clock::now();
            for (size_t i = 0; i < size_; ++i) {
                const auto& s = buf_[(head_ + buf_.size() - 1 - i) % buf_.size()];
                if (std::chrono::duration<double>(now - s.first).count() > window_s) break;
                v.push_back(s.second);
            }
        }
        if (v.empty()) return {{"n", 0}};
        std::sort(v.begin(), v.end());
        auto q = [&](double p) { return v[std::min(v.size() - 1, size_t(p * v.size()))]; };
        double mean = 0; for (double x : v) mean += x; mean /= v.size();
        return {{"n", v.size()}, {"p50", q(0.50)}, {"p90", q(0.90)}, {"p95", q(0.95)}, {"p99", q(0.99)},
                {"min", v.front()}, {"max", v.back()}, {"mean", mean}};
    }
private:
    mutable std::mutex m_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, double>> buf_;
    size_t head_ = 0, size_ = 0;
};

// server-wide counters (not per model)
struct GlobalCounters {
    std::atomic<uint64_t> rate_limited{0}, auth_failed{0};
};

struct ModelMetrics {
    Histogram queue, decode, pre, infer, post, encode, total;
    SampleRing total_ring, infer_ring;
    std::atomic<int64_t> queue_depth{0}, busy_workers{0};
    std::atomic<uint64_t> requests_ok{0}, requests_err{0}, dropped_frames{0}, images_decoded{0};
    std::mutex codes_m; std::map<int, uint64_t> codes;   // HTTP status counts
    void count_code(int c) { std::lock_guard<std::mutex> g(codes_m); ++codes[c]; if (c < 400) ++requests_ok; else ++requests_err; }
};

} // namespace yolomaster::server
