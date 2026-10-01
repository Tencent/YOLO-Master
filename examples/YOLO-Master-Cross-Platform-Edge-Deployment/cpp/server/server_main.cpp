// yolomaster_server - HTTP/WebSocket inference API for YOLO-Master edge models.
// Backends: ONNX Runtime (cpu/cuda), TensorRT (.engine or .onnx built+cached), ncnn (cpu/vulkan), MNN.
#include "http_api.hpp"
#include "CLI11.hpp"
#include <csignal>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace yolomaster::server;

namespace yolomaster::server {
ModelSpec parse_model_arg(const std::string& arg) {
    ModelSpec m;
    std::string rest = arg;
    const auto eq = rest.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("model spec needs id=path[,key=value...]: " + arg);
    m.id = rest.substr(0, eq);
    rest = rest.substr(eq + 1);
    size_t pos = 0; bool first = true;
    while (pos <= rest.size()) {
        auto comma = rest.find(',', pos); if (comma == std::string::npos) comma = rest.size();
        const std::string tok = rest.substr(pos, comma - pos);
        if (first) { m.path = tok; first = false; }
        else if (!tok.empty()) {
            const auto e = tok.find('=');
            if (e == std::string::npos) throw std::invalid_argument("bad key=value in model spec: " + tok);
            const std::string k = tok.substr(0, e), v = tok.substr(e + 1);
            if (k == "backend") m.backend = v; else if (k == "device") m.device = v; else if (k == "precision") m.precision = v;
            else if (k == "threads") m.threads = std::stoi(v); else if (k == "workers") m.workers = std::stoi(v);
            else if (k == "imgsz") m.imgsz = std::stoi(v); else if (k == "conf") m.conf = std::stof(v);
            else if (k == "iou") m.iou = std::stof(v); else if (k == "max_det") m.max_det = std::stoi(v);
            else if (k == "classes") m.classes = v; else if (k == "preload") m.preload = (v == "1" || v == "true");
            else if (k == "slicing") m.slicing = v; else if (k == "tile_size") m.tile_size = std::stoi(v);
        else if (k == "preproc") m.preproc = v;
        else if (k == "cuda_graph") m.cuda_graph = (v == "1" || v == "true");
            else if (k == "multi_label") m.multi_label = (v == "1" || v == "true"); else if (k == "stretch") m.stretch = (v == "1" || v == "true");
            else throw std::invalid_argument("unknown model spec key: " + k);
        }
        pos = comma + 1;
    }
    if (m.path.empty()) throw std::invalid_argument("model spec has no path: " + arg);
    return m;
}
}

static std::atomic<int> g_signal{0};
static void on_signal(int s) { g_signal = s; }

int main(int argc, char** argv) {
    CLI::App app{"yolomaster_server - YOLO-Master inference API (REST + WebSocket)"};
    app.set_version_flag("--version", std::string(YM_SERVER_VERSION) + " (runtime " + YM_VERSION + ", " + YM_GIT_COMMIT + ")");
    std::string config_path, host, engine_cache, log_level, api_key_file, log_format;
    double rate_rps = -1; int rate_burst = -1;
    int port = -1, loop_threads = -1, max_queue = -1, timeout_ms = -1, max_body_mb = -1, drain_ms = 5000;
    bool check_config = false, no_preload = false, print_config = false;
    std::vector<std::string> model_args;
    app.add_option("-c,--config", config_path, "server config JSON (deploy/server.json)");
    app.add_option("-m,--model", model_args, "model spec id=path[,backend=trt,device=cuda,precision=fp16,threads=8,workers=1,...] (repeatable)");
    app.add_option("-p,--port", port, "listen port (default 8080)");
    app.add_option("--host", host, "bind address (default 0.0.0.0)");
    app.add_option("--loop-threads", loop_threads, "HTTP event-loop threads (default 2)");
    app.add_option("--max-queue", max_queue, "pending jobs per model before 503 (default 64)");
    app.add_option("--timeout-ms", timeout_ms, "queue+inference deadline per request (default 10000)");
    app.add_option("--max-body-mb", max_body_mb, "request body cap (default 32)");
    app.add_option("--engine-cache", engine_cache, "directory for TensorRT engines built from .onnx");
    app.add_option("--log-level", log_level, "debug|info|warn|error");
    app.add_option("--drain-ms", drain_ms, "graceful shutdown: wait for queued jobs up to this long");
    app.add_option("--api-key-file", api_key_file, "file with one API key per line (blank / # lines ignored); enables auth. "
                   "Env YM_API_KEYS=k1,k2 is the alternative; precedence: config < env < this flag");
    app.add_option("--rate-rps", rate_rps, "token-bucket rate limit per API key or peer IP, requests per second (0 = off)");
    app.add_option("--rate-burst", rate_burst, "rate limit burst (bucket size; default = rps rounded up)");
    app.add_option("--log-format", log_format, "access log format: plain|json");
    app.add_flag("--no-preload", no_preload, "load models on first request instead of at startup");
    app.add_flag("--check-config", check_config, "validate config + model files and exit");
    app.add_flag("--print-config", print_config, "print the effective config JSON and exit");
    CLI11_PARSE(app, argc, argv);

    ServerConfig cfg;
    try {
        if (!config_path.empty()) {
            std::ifstream f(config_path);
            if (!f) { std::cerr << "cannot open config: " << config_path << "\n"; return 2; }
            cfg = nlohmann::json::parse(f, nullptr, true, true).get<ServerConfig>();   // comments allowed
        }
        for (const auto& a : model_args) {
            ModelSpec m = parse_model_arg(a);
            bool replaced = false;
            for (auto& e : cfg.models) if (e.id == m.id) { e = m; replaced = true; }
            if (!replaced) cfg.models.push_back(m);
        }
    } catch (const std::exception& e) { std::cerr << "config error: " << e.what() << "\n"; return 2; }
    if (port > 0) cfg.port = port;
    if (!host.empty()) cfg.host = host;
    if (loop_threads > 0) cfg.loop_threads = loop_threads;
    if (max_queue > 0) cfg.max_queue = max_queue;
    if (timeout_ms > 0) cfg.request_timeout_ms = timeout_ms;
    if (max_body_mb > 0) cfg.max_body_mb = max_body_mb;
    if (!engine_cache.empty()) cfg.engine_cache_dir = engine_cache;
    if (!log_level.empty()) cfg.log_level = log_level;
    if (!log_format.empty()) cfg.log_format = log_format;
    if (cfg.log_format != "plain" && cfg.log_format != "json") { std::cerr << "log_format must be plain|json\n"; return 2; }
    if (rate_rps >= 0) cfg.rate_limit.rps = rate_rps;
    if (rate_burst >= 0) cfg.rate_limit.burst = rate_burst;
    // API keys: config file < YM_API_KEYS < --api-key-file (each source replaces the previous one)
    auto trim = [](std::string v) {
        const auto b = v.find_first_not_of(" \t\r\n"); if (b == std::string::npos) return std::string();
        const auto e = v.find_last_not_of(" \t\r\n"); return v.substr(b, e - b + 1);
    };
    if (const char* env = std::getenv("YM_API_KEYS")) {
        cfg.api_keys.clear();
        std::stringstream ss(env); std::string k;
        while (std::getline(ss, k, ',')) { k = trim(k); if (!k.empty()) cfg.api_keys.push_back(k); }
    }
    if (!api_key_file.empty()) {
        std::ifstream f(api_key_file);
        if (!f) { std::cerr << "cannot open --api-key-file: " << api_key_file << "\n"; return 2; }
        cfg.api_keys.clear();
        std::string k;
        while (std::getline(f, k)) { k = trim(k); if (!k.empty() && k[0] != '#') cfg.api_keys.push_back(k); }
    }
    for (const auto& k : cfg.api_keys)
        if (k.size() < 16) { std::cerr << "refusing an API key shorter than 16 characters\n"; return 2; }
    if (no_preload) for (auto& m : cfg.models) m.preload = false;
    if (cfg.models.empty()) { std::cerr << "no models configured (use --config or --model id=path)\n"; return 2; }
    if (print_config) { std::cout << nlohmann::json(cfg).dump(2) << "\n"; return 0; }

    // validate model paths up-front
    int bad = 0;
    for (const auto& m : cfg.models) {
        std::error_code ec;
        if (!std::filesystem::exists(m.path, ec)) { std::cerr << "model '" << m.id << "': path not found: " << m.path << "\n"; ++bad; }
    }
    if (bad) return 2;
    if (check_config) { std::cout << "config ok: " << cfg.models.size() << " model(s)\n"; return 0; }

    ServerState st(cfg);
    for (const auto& m : cfg.models) if (m.preload) { std::string err; if (!st.reg.load(m.id, err)) { std::cerr << err << "\n"; return 3; } }

    std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal); std::signal(SIGPIPE, SIG_IGN);

    std::atomic<int> bound{0};
    std::vector<std::thread> loops;
    for (int i = 0; i < std::max(1, cfg.loop_threads); ++i) loops.emplace_back([&st, i, &bound] { run_http_loop(st, i, bound); });

    // supervisor: signal -> stop accepting, drain pools, join
    while (!g_signal) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // all loops failed to bind
        bool all_dead = true; for (auto& t : loops) if (t.joinable()) { all_dead = false; break; }
        if (all_dead) break;
        if (bound == 0) {
            // give loops a moment to bind; if none bound and threads exited, fail
            static int ticks = 0; if (++ticks > 20) { bool any = false; for (auto& t : loops) any |= t.joinable(); if (!any) break; }
        }
    }
    std::cerr << "[server] shutting down (signal " << g_signal.load() << ")\n";
    request_stop(st);
    st.reg.shutdown_all(drain_ms);
    for (auto& t : loops) if (t.joinable()) t.join();
    std::cerr << "[server] bye\n";
    return bound > 0 ? 0 : 4;
}
