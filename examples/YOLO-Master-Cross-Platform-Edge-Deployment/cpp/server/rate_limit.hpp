// Token-bucket rate limiting per client (API key when present, else peer address). One map shared
// by all event-loop threads: SO_REUSEPORT spreads one client's connections across loops, so
// per-thread buckets would make the effective limit depend on connection hashing. The critical
// section is a map lookup and a few multiplications.
#pragma once
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>

namespace yolomaster::server {

class RateLimiter {
public:
    struct Verdict { bool ok = true; int limit = 0; int remaining = 0; int retry_after_s = 0; };
    using Clock = std::chrono::steady_clock;

    // rps <= 0 disables; burst = bucket capacity (defaults to rps rounded up)
    Verdict take(const std::string& key, double rps, int burst) {
        Verdict v;
        if (rps <= 0) return v;
        if (burst <= 0) burst = static_cast<int>(std::ceil(rps));
        v.limit = burst;
        const auto now = Clock::now();
        std::lock_guard<std::mutex> g(m_);
        if (++calls_ % 4096 == 0) purge(now);
        auto& b = buckets_[key];
        if (b.last.time_since_epoch().count() == 0) { b.tokens = burst; b.last = now; }
        const double dt = std::chrono::duration<double>(now - b.last).count();
        b.tokens = std::min<double>(burst, b.tokens + dt * rps);
        b.last = now;
        if (b.tokens >= 1.0) {
            b.tokens -= 1.0;
            v.remaining = static_cast<int>(std::floor(b.tokens));
            return v;
        }
        v.ok = false;
        v.remaining = 0;
        v.retry_after_s = std::max(1, static_cast<int>(std::ceil((1.0 - b.tokens) / rps)));
        return v;
    }
    size_t size() { std::lock_guard<std::mutex> g(m_); return buckets_.size(); }

private:
    struct Bucket { double tokens = 0; Clock::time_point last{}; };
    void purge(Clock::time_point now) {
        for (auto it = buckets_.begin(); it != buckets_.end();) {
            if (std::chrono::duration<double>(now - it->second.last).count() > 600.0) it = buckets_.erase(it);
            else ++it;
        }
    }
    std::mutex m_;
    std::unordered_map<std::string, Bucket> buckets_;
    uint64_t calls_ = 0;
};

} // namespace yolomaster::server
