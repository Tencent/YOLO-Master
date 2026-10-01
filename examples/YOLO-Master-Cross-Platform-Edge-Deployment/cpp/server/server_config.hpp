// Server configuration: JSON file (deploy/server.json) merged with CLI overrides.
#pragma once
#include <string>
#include <vector>
#include "json.hpp"

namespace yolomaster::server {

struct ModelSpec {
    std::string id;                 // route key: /v1/infer?model=<id>
    std::string path;               // .onnx | .engine | .mnn | <dir>_ncnn | model.ncnn.param
    std::string backend = "auto";   // auto|onnx|trt|ncnn|mnn
    std::string device = "cpu";     // cpu|cuda|vulkan|opencl (trt is always CUDA)
    std::string precision = "auto"; // auto|fp32|fp16|int8 (ncnn: int8 sibling; trt: engine build flag)
    int threads = 4;                // intra-op threads per worker (CPU backends)
    int workers = 0;                // 0 = auto (GPU: 1, CPU: min(4, hw/threads))
    int imgsz = 0;                  // 0 = model metadata / 640
    float conf = 0.25f, iou = 0.5f;
    int max_det = 300;
    std::string classes = "auto";   // auto|visdrone|sku|coco
    bool multi_label = false, stretch = false;
    bool preload = true;            // load at startup (readyz waits for it)
    std::string slicing = "off";    // off|dense|sparse (per-request override allowed)
    int tile_size = 0;
    std::string preproc = "gpu";    // gpu|cpu: TensorRT / ORT-CUDA preprocessing (gpu needs USE_CUDA_PREPROC)
    bool cuda_graph = false;        // TensorRT: CUDA graph replay
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int loop_threads = 2;           // HTTP event-loop threads (SO_REUSEPORT)
    int max_body_mb = 32;
    int max_pixels = 50'000'000;    // decoded image cap (w*h)
    int max_queue = 64;             // pending jobs per model before 503
    int request_timeout_ms = 10000; // queue + inference deadline
    int ws_max_payload_mb = 16;
    std::string engine_cache_dir;   // TensorRT engines built from .onnx land here
    std::string log_level = "info"; // debug|info|warn|error
    bool cors = true;
    bool access_log = true;
    std::string log_format = "plain";        // plain | json (one JSON object per request on stderr)
    std::vector<std::string> api_keys;       // empty = authentication off
    std::vector<std::string> auth_exempt{"/healthz", "/readyz"};   // paths served without a key
    struct RateLimit { double rps = 0; int burst = 0; } rate_limit;  // rps <= 0 = off; per API key or peer IP
    std::vector<ModelSpec> models;
};

inline void from_json(const nlohmann::json& j, ModelSpec& m) {
    j.at("id").get_to(m.id); j.at("path").get_to(m.path);
    m.backend = j.value("backend", m.backend); m.device = j.value("device", m.device);
    m.precision = j.value("precision", m.precision); m.threads = j.value("threads", m.threads);
    m.workers = j.value("workers", m.workers); m.imgsz = j.value("imgsz", m.imgsz);
    m.conf = j.value("conf", m.conf); m.iou = j.value("iou", m.iou); m.max_det = j.value("max_det", m.max_det);
    m.classes = j.value("classes", m.classes); m.multi_label = j.value("multi_label", m.multi_label);
    m.stretch = j.value("stretch", m.stretch); m.preload = j.value("preload", m.preload);
    m.slicing = j.value("slicing", m.slicing); m.tile_size = j.value("tile_size", m.tile_size);
    m.preproc = j.value("preproc", m.preproc); m.cuda_graph = j.value("cuda_graph", m.cuda_graph);
}
inline void to_json(nlohmann::json& j, const ModelSpec& m) {
    j = {{"id", m.id}, {"path", m.path}, {"backend", m.backend}, {"device", m.device},
         {"precision", m.precision}, {"threads", m.threads}, {"workers", m.workers}, {"imgsz", m.imgsz},
         {"conf", m.conf}, {"iou", m.iou}, {"max_det", m.max_det}, {"classes", m.classes},
         {"multi_label", m.multi_label}, {"stretch", m.stretch}, {"preload", m.preload},
         {"slicing", m.slicing}, {"tile_size", m.tile_size}, {"preproc", m.preproc}, {"cuda_graph", m.cuda_graph}};
}
inline void from_json(const nlohmann::json& j, ServerConfig& c) {
    c.host = j.value("host", c.host); c.port = j.value("port", c.port);
    c.loop_threads = j.value("loop_threads", c.loop_threads); c.max_body_mb = j.value("max_body_mb", c.max_body_mb);
    c.max_pixels = j.value("max_pixels", c.max_pixels); c.max_queue = j.value("max_queue", c.max_queue);
    c.request_timeout_ms = j.value("request_timeout_ms", c.request_timeout_ms);
    c.ws_max_payload_mb = j.value("ws_max_payload_mb", c.ws_max_payload_mb);
    c.engine_cache_dir = j.value("engine_cache_dir", c.engine_cache_dir);
    c.log_level = j.value("log_level", c.log_level); c.cors = j.value("cors", c.cors);
    c.access_log = j.value("access_log", c.access_log);
    c.log_format = j.value("log_format", c.log_format);
    if (j.contains("api_keys")) c.api_keys = j.at("api_keys").get<std::vector<std::string>>();
    if (j.contains("auth_exempt")) c.auth_exempt = j.at("auth_exempt").get<std::vector<std::string>>();
    if (j.contains("rate_limit")) {
        c.rate_limit.rps = j["rate_limit"].value("rps", c.rate_limit.rps);
        c.rate_limit.burst = j["rate_limit"].value("burst", c.rate_limit.burst);
    }
    if (j.contains("models")) c.models = j.at("models").get<std::vector<ModelSpec>>();
}
inline void to_json(nlohmann::json& j, const ServerConfig& c) {
    j = {{"host", c.host}, {"port", c.port}, {"loop_threads", c.loop_threads}, {"max_body_mb", c.max_body_mb},
         {"max_pixels", c.max_pixels}, {"max_queue", c.max_queue}, {"request_timeout_ms", c.request_timeout_ms},
         {"ws_max_payload_mb", c.ws_max_payload_mb}, {"engine_cache_dir", c.engine_cache_dir},
         {"log_level", c.log_level}, {"cors", c.cors}, {"access_log", c.access_log},
         {"log_format", c.log_format},
         {"api_keys_count", c.api_keys.size()},   // never the keys themselves (--print-config is not a secret store)
         {"auth_exempt", c.auth_exempt},
         {"rate_limit", {{"rps", c.rate_limit.rps}, {"burst", c.rate_limit.burst}}},
         {"models", c.models}};
}

// Parse a CLI model spec "id=path[,backend=trt][,device=cuda][,precision=fp16][,threads=8][,workers=1]...".
// Throws std::invalid_argument on malformed input.
ModelSpec parse_model_arg(const std::string& arg);

} // namespace yolomaster::server
