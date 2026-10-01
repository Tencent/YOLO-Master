// HTTP/WebSocket layer (uWebSockets). One uWS::App per event-loop thread; inference runs on
// the model's WorkerPool threads and results are re-posted to the loop with Loop::defer.
#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "model_registry.hpp"
#include "rate_limit.hpp"

namespace uWS { struct Loop; }

namespace yolomaster::server {

struct ServerState {
    ServerConfig cfg;
    ModelRegistry reg;
    std::atomic<bool> stopping{false};
    std::atomic<uint64_t> request_seq{0};
    RateLimiter limiter;
    GlobalCounters counters;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::mutex loops_m;
    std::vector<std::pair<uWS::Loop*, std::function<void()>>> closers;   // per loop: close the App
    explicit ServerState(const ServerConfig& c) : cfg(c), reg(c) {}
    double uptime_s() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); }
};

// Runs one event loop (blocking) serving all routes on cfg.host:cfg.port (SO_REUSEPORT across threads).
// Returns false when the port could not be bound.
bool run_http_loop(ServerState& st, int loop_index, std::atomic<int>& bound_count);
// Ask every loop to close its App (listen sockets + connections); loops then exit run().
void request_stop(ServerState& st);

} // namespace yolomaster::server
