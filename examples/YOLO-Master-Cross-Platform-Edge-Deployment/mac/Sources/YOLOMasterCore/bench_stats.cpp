#include "bench_stats.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <numeric>

namespace yolomaster::bench {

StageStats reduce(std::vector<double> v) {
    StageStats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    auto q = [&](double p) { return v[std::min(static_cast<size_t>(p * n), n - 1)]; };
    s.n = n;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    s.median = q(0.5); s.p90 = q(0.9); s.p95 = q(0.95); s.p99 = q(0.99);
    s.min = v.front(); s.max = v.back();
    return s;
}

SustainedSummary summarize_sustained(const std::vector<double>& all, int cold_iters) {
    SustainedSummary r;
    if (all.empty()) return r;
    std::vector<double> head(all.begin(), all.begin() + std::min<size_t>(all.size(), std::max(cold_iters, 1)));
    r.cold_median_ms = reduce(head).median;
    std::vector<double> sorted = all;
    std::sort(sorted.begin(), sorted.end());
    const size_t q = std::max<size_t>(sorted.size() / 4, 1);
    std::vector<double> slowest(sorted.end() - q, sorted.end());
    r.sustained_median_ms = reduce(slowest).median;
    r.throttle_pct = r.cold_median_ms > 0 ? (r.sustained_median_ms - r.cold_median_ms) / r.cold_median_ms * 100.0 : 0;
    return r;
}

// ---- compact SHA-256 (FIPS 180-4), enough for list fingerprints ----
namespace {
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
}

std::string sha256_hex(const std::string& data) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::string msg = data;
    const uint64_t bitlen = static_cast<uint64_t>(data.size()) * 8;
    msg.push_back(static_cast<char>(0x80));
    while (msg.size() % 64 != 56) msg.push_back('\0');
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bitlen >> (i * 8)) & 0xff));
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4])) << 24) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4 + 1])) << 16) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4 + 2])) << 8) |
                   static_cast<uint32_t>(static_cast<uint8_t>(msg[off + i * 4 + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h[i]);
    return std::string(out, 64);
}

std::string image_list_sha256(const std::vector<std::string>& paths) {
    std::vector<std::string> names;
    names.reserve(paths.size());
    for (const auto& p : paths) {
        const auto slash = p.find_last_of("/\\");
        names.push_back(slash == std::string::npos ? p : p.substr(slash + 1));
    }
    std::sort(names.begin(), names.end());
    std::string joined;
    for (size_t i = 0; i < names.size(); ++i) { if (i) joined += '\n'; joined += names[i]; }
    return sha256_hex(joined);
}

std::string timestamp_utc() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

} // namespace yolomaster::bench
