// API-key authentication: `X-API-Key: <key>` or `Authorization: Bearer <key>` against the keys in
// ServerConfig::api_keys, compared in constant time over every configured key (no early exit, no
// length leak). An empty key list disables authentication.
#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace yolomaster::server::auth {

inline bool ct_equal(std::string_view a, std::string_view b) {
    const size_t n = a.size() > b.size() ? a.size() : b.size();
    volatile unsigned char diff = static_cast<unsigned char>(a.size() != b.size());
    for (size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff = static_cast<unsigned char>(diff | (x ^ y));
    }
    return diff == 0;
}

// The presented key from the two accepted headers ("" when absent).
inline std::string presented_key(std::string_view x_api_key, std::string_view authorization) {
    if (!x_api_key.empty()) return std::string(x_api_key);
    if (authorization.size() > 7) {
        std::string pre(authorization.substr(0, 7));
        for (auto& c : pre) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (pre == "bearer ") {
            std::string_view k = authorization.substr(7);
            while (!k.empty() && k.front() == ' ') k.remove_prefix(1);
            while (!k.empty() && (k.back() == ' ' || k.back() == '\r')) k.remove_suffix(1);
            return std::string(k);
        }
    }
    return "";
}

// true when auth is off or the presented key matches one of the configured keys
inline bool authorized(const std::vector<std::string>& keys, std::string_view presented) {
    if (keys.empty()) return true;
    bool ok = false;
    for (const auto& k : keys) ok |= ct_equal(k, presented);   // visit every key
    return ok;
}

} // namespace yolomaster::server::auth
