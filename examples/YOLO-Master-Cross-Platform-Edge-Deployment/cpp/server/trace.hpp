// Request identity and W3C trace context: an inbound X-Request-Id is honoured when it is a safe
// token, an inbound `traceparent` is parsed strictly (else a new trace is minted), and every
// response carries both, so a reverse proxy or an OpenTelemetry collector can stitch the server's
// JSON access log into the caller's trace.
#pragma once
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>

namespace yolomaster::server::trace {

inline bool valid_rid(std::string_view s) {
    if (s.empty() || s.size() > 64) return false;
    for (char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

inline std::string gen_hex(int nbytes) {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::string out;
    out.reserve(static_cast<size_t>(nbytes) * 2);
    char b[3];
    for (int i = 0; i < nbytes; ++i) {
        std::snprintf(b, sizeof b, "%02x", static_cast<unsigned>(rng() & 0xff));
        out += b;
    }
    return out;
}

struct TraceCtx {
    std::string trace_id;     // 32 hex
    std::string parent_span;  // 16 hex, the caller's span ("" when we started the trace)
    std::string span_id;      // 16 hex, our span
    std::string flags = "01";
    bool inbound = false;
};

inline bool all_hex(std::string_view s) {
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}
inline bool all_zero(std::string_view s) {
    for (char c : s) if (c != '0') return false;
    return true;
}

// "00-<32 hex trace>-<16 hex span>-<2 hex flags>"; anything else starts a fresh trace.
inline TraceCtx parse_traceparent(std::string_view h) {
    TraceCtx t;
    if (h.size() == 55 && h.substr(0, 3) == "00-" && h[35] == '-' && h[52] == '-') {
        const auto tid = h.substr(3, 32), sid = h.substr(36, 16), fl = h.substr(53, 2);
        if (all_hex(tid) && all_hex(sid) && all_hex(fl) && !all_zero(tid) && !all_zero(sid)) {
            t.trace_id = std::string(tid); t.parent_span = std::string(sid); t.flags = std::string(fl); t.inbound = true;
        }
    }
    if (!t.inbound) t.trace_id = gen_hex(16);
    t.span_id = gen_hex(8);
    return t;
}

inline std::string render_traceparent(const TraceCtx& t) {
    return "00-" + t.trace_id + "-" + t.span_id + "-" + t.flags;
}

} // namespace yolomaster::server::trace
