// Statistics of the benchmark mode, shared by every host of the core (Linux / Windows / Jetson
// CLI and server through cpp/include/bench.hpp, the macOS / iOS Kit through the Swift package).
//
// Convention: floor rank. For an ascending sorted vector s of n samples,
// percentile q = s[min(int(q * n), n - 1)]; median is q = 0.5. This is exactly what the Android
// BenchStats and the iOS BenchView compute, so device numbers stay comparable.
// Sustained: median of the slowest quarter of all samples (the phones' lastQuarterMedian);
// throttle_pct = (sustained - cold) / cold * 100.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace yolomaster::bench {

constexpr const char* kSchema = "yolomaster-bench/v1";

struct StageStats {
    size_t n = 0;
    double mean = 0, median = 0, p90 = 0, p95 = 0, p99 = 0, min = 0, max = 0;
};
StageStats reduce(std::vector<double> samples);      // sorts a copy; n == 0 -> all zero

struct SustainedSummary {
    double cold_median_ms = 0;        // median of the first `cold_iters` samples
    double sustained_median_ms = 0;   // median of the slowest quarter
    double throttle_pct = 0;
};
// `all` in acquisition order; an empty vector yields zeros.
SustainedSummary summarize_sustained(const std::vector<double>& all, int cold_iters);

std::string sha256_hex(const std::string& data);
// sha256 over the sorted basenames joined by '\n' (scripts/make_coco_subset.py hashes the same string)
std::string image_list_sha256(const std::vector<std::string>& paths);
std::string timestamp_utc();

} // namespace yolomaster::bench
