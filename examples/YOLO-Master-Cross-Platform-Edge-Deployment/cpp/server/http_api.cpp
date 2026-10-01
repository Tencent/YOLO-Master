#include "http_api.hpp"
#include "serialize.hpp"
#include "auth.hpp"
#include "routes.hpp"
#include "trace.hpp"
#include <functional>
#include "App.h"
#include "Multipart.h"
#include <future>
#include <map>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>
#include <cmath>
#ifdef HAVE_VIDEOIO
#include <opencv2/videoio.hpp>
#endif

namespace yolomaster::server {

namespace {

using Res = uWS::HttpResponse<false>;
using Req = uWS::HttpRequest;

std::string now_iso() {
    const auto t = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(t);
    char b[40]; std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", std::gmtime(&tt));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count() % 1000;
    char o[48]; std::snprintf(o, sizeof o, "%s.%03dZ", b, int(ms)); return o;
}

std::string make_rid(ServerState& st) {
    char b[32]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long)(++st.request_seq)); return b;
}

// A response that may be finished from another thread's completion via loop->defer.
struct Pending {
    Res* res; uWS::Loop* loop; std::atomic<bool> aborted{false};
    std::string rid, method, path; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ServerState* st;
    trace::TraceCtx trace;                                   // W3C trace context (inbound or minted)
    std::string client;                                      // peer address ("k:<hash>" once a key authenticated)
    std::vector<std::pair<std::string, std::string>> hdrs;   // extra response headers (rate limit, auth challenge)
    std::string model; int worker = -1;                      // filled by respond_result for the access log
    double stages[7] = {0, 0, 0, 0, 0, 0, 0}; bool has_stages = false;   // queue decode pre infer post encode total
    Pending(Res* r, uWS::Loop* l, ServerState* s) : res(r), loop(l), st(s) {}
};

void access_log(ServerState& st, const Pending& p, int status, size_t bytes) {
    if (!st.cfg.access_log) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p.t0).count();
    if (st.cfg.log_format == "json") {
        // one object per line: an OpenTelemetry collector's filelog receiver (json_parser + timestamp
        // on "ts") turns these into log records joined to the caller's trace by trace_id / span_id
        nlohmann::json j = {{"ts", now_iso()}, {"rid", p.rid}, {"trace_id", p.trace.trace_id}, {"span_id", p.trace.span_id},
                            {"parent_span", p.trace.parent_span}, {"method", p.method}, {"path", p.path}, {"status", status},
                            {"bytes", bytes}, {"ms", std::round(ms * 100.0) / 100.0}, {"client", p.client}};
        if (!p.model.empty()) j["model"] = p.model;
        if (p.worker >= 0) j["worker"] = p.worker;
        if (p.has_stages)
            j["stages"] = {{"queue", p.stages[0]}, {"decode", p.stages[1]}, {"pre", p.stages[2]}, {"infer", p.stages[3]},
                           {"post", p.stages[4]}, {"encode", p.stages[5]}, {"total", p.stages[6]}};
        const std::string line = j.dump() + "\n";
        std::fwrite(line.data(), 1, line.size(), stderr);
        return;
    }
    std::fprintf(stderr, "%s access rid=%s %s %s %d %zuB %.2fms\n", now_iso().c_str(), p.rid.c_str(),
                 p.method.c_str(), p.path.c_str(), status, bytes, ms);
}

void cors_headers(Res* res, const ServerConfig& cfg) {
    if (!cfg.cors) return;
    res->writeHeader("Access-Control-Allow-Origin", "*");
    res->writeHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res->writeHeader("Access-Control-Allow-Headers", "Content-Type, X-Request-Id, X-API-Key, Authorization, traceparent");
    res->writeHeader("Access-Control-Expose-Headers", "X-Request-Id, traceparent, X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After, X-Infer-Ms, X-Detections");
}

const char* status_text(int code) {
    switch (code) {
        case 200: return "200 OK"; case 202: return "202 Accepted"; case 204: return "204 No Content";
        case 400: return "400 Bad Request"; case 401: return "401 Unauthorized"; case 403: return "403 Forbidden";
        case 404: return "404 Not Found"; case 405: return "405 Method Not Allowed";
        case 409: return "409 Conflict"; case 413: return "413 Payload Too Large"; case 415: return "415 Unsupported Media Type";
        case 422: return "422 Unprocessable Entity"; case 429: return "429 Too Many Requests";
        case 500: return "500 Internal Server Error"; case 501: return "501 Not Implemented";
        case 503: return "503 Service Unavailable"; case 504: return "504 Gateway Timeout";
        default: return "500 Internal Server Error";
    }
}

// Must run on the loop thread.
void send(Pending& p, int code, const std::string& body, const char* ctype, const std::string& extra_hdr = "", const std::string& extra_val = "") {
    if (p.aborted) return;
    p.res->cork([&] {
        p.res->writeStatus(status_text(code));
        p.res->writeHeader("Content-Type", ctype);
        p.res->writeHeader("X-Request-Id", p.rid);
        p.res->writeHeader("traceparent", trace::render_traceparent(p.trace));
        for (const auto& h : p.hdrs) p.res->writeHeader(h.first, h.second);
        if (!extra_hdr.empty()) p.res->writeHeader(extra_hdr, extra_val);
        cors_headers(p.res, p.st->cfg);
        p.res->end(body);
    });
    access_log(*p.st, p, code, body.size());
}
void send_json(Pending& p, int code, const nlohmann::json& j, const std::string& eh = "", const std::string& ev = "") {
    send(p, code, j.dump(), "application/json", eh, ev);
}
void send_error(Pending& p, int code, const std::string& msg, const std::string& eh = "", const std::string& ev = "") {
    send_json(p, code, {{"error", msg}, {"code", code}, {"request_id", p.rid}}, eh, ev);
}
// Deliver from any thread.
void deliver(const std::shared_ptr<Pending>& p, std::function<void(Pending&)> fn) {
    if (p->aborted) return;
    p->loop->defer([p, fn = std::move(fn)]() { if (!p->aborted) fn(*p); });
}

float qf(Req* req, const char* k, float def) { auto v = req->getQuery(k); if (v.empty()) return def; try { return std::stof(std::string(v)); } catch (...) { return def; } }
int qi(Req* req, const char* k, int def) { auto v = req->getQuery(k); if (v.empty()) return def; try { return std::stoi(std::string(v)); } catch (...) { return def; } }
std::string qs(Req* req, const char* k, const std::string& def = "") { auto v = req->getQuery(k); return v.empty() ? def : std::string(v); }
bool qb(Req* req, const char* k, bool def) { auto v = req->getQuery(k); if (v.empty()) return def; return v == "1" || v == "true" || v == "yes"; }

InferParams params_from(Req* req, const std::string& ret) {
    InferParams p;
    p.conf = qf(req, "conf", -1); p.iou = qf(req, "iou", -1); p.max_det = qi(req, "max_det", -1);
    auto ml = req->getQuery("multi_label"); if (!ml.empty()) p.multi_label = (ml == "1" || ml == "true") ? 1 : 0;
    p.slicing = qs(req, "slicing"); p.tile_size = qi(req, "tile_size", -1);
    p.annotated = ret == "annotated"; p.mask_overlay = qs(req, "masks") == "overlay"; p.mask_coeffs = qb(req, "mask_coeffs", false);
    p.jpeg_quality = qi(req, "quality", 90);
    p.track = qs(req, "track");
    if (p.track == "off") p.track.clear();
    return p;
}

// Collect the request body (bounded), then call `cb(body)` on the loop thread.
// Oversized bodies are drained (discarded) to the end and then answered 413: ending the response
// while the client is still uploading makes the kernel RST the socket and the client never sees
// the status.
void read_body(const std::shared_ptr<Pending>& p, size_t max_bytes, std::function<void(std::string&&)> cb) {
    auto buf = std::make_shared<std::string>();
    auto too_big = std::make_shared<bool>(false);
    p->res->onAborted([p] { p->aborted = true; });
    p->res->onData([p, buf, too_big, max_bytes, cb = std::move(cb)](std::string_view chunk, bool last) mutable {
        if (p->aborted) return;
        if (!*too_big) {
            if (buf->size() + chunk.size() > max_bytes) { *too_big = true; std::string().swap(*buf); }
            else buf->append(chunk.data(), chunk.size());
        }
        if (!last) return;
        if (*too_big) send_error(*p, 413, "body exceeds max_body_mb=" + std::to_string(p->st->cfg.max_body_mb));
        else cb(std::move(*buf));
    });
}

// First file part of a multipart body (or the raw body when not multipart).
bool extract_image(const std::string& content_type, const std::string& body, std::string& out, std::string& filename) {
    if (content_type.rfind("multipart/form-data", 0) == 0) {
        uWS::MultipartParser mp(content_type);
        mp.setBody(body);
        std::pair<std::string_view, std::string_view> headers[10];
        while (auto part = mp.getNextPart(headers)) {
            std::string disp;
            for (auto& h : headers) { if (h.first.empty()) break; if (h.first == "content-disposition") disp = std::string(h.second); }
            const auto fn = disp.find("filename=\"");
            if (fn != std::string::npos) { const auto e = disp.find('"', fn + 10); filename = disp.substr(fn + 10, e == std::string::npos ? std::string::npos : e - fn - 10); }
            if (fn != std::string::npos || disp.find("name=\"file\"") != std::string::npos || disp.find("name=\"image\"") != std::string::npos) {
                out.assign(part->data(), part->size()); return true;
            }
        }
        return false;
    }
    out = body; return !out.empty();
}

// Every file part of a multipart body, in order.
std::vector<std::pair<std::string, std::string>> extract_all(const std::string& content_type, const std::string& body) {
    std::vector<std::pair<std::string, std::string>> v;
    if (content_type.rfind("multipart/form-data", 0) != 0) return v;
    uWS::MultipartParser mp(content_type);
    mp.setBody(body);
    std::pair<std::string_view, std::string_view> headers[10];
    int i = 0;
    while (auto part = mp.getNextPart(headers)) {
        std::string disp, name = "part" + std::to_string(i++);
        for (auto& h : headers) { if (h.first.empty()) break; if (h.first == "content-disposition") disp = std::string(h.second); }
        const auto fn = disp.find("filename=\"");
        if (fn == std::string::npos) continue;
        const auto e = disp.find('"', fn + 10);
        name = disp.substr(fn + 10, e == std::string::npos ? std::string::npos : e - fn - 10);
        v.emplace_back(name, std::string(part->data(), part->size()));
    }
    return v;
}

WorkerPool* resolve_pool(ServerState& st, Req* req, std::shared_ptr<Pending>& p, std::string& model_id) {
    model_id = qs(req, "model");
    if (model_id.empty()) {
        // single-model convenience: the only configured model is the default
        auto specs = st.reg.specs();
        if (specs.size() == 1) model_id = specs[0].id;
        else { send_error(*p, 400, "missing ?model=<id>"); return nullptr; }
    }
    WorkerPool* pool = st.reg.get(model_id);
    if (!pool) {
        if (!st.reg.has_spec(model_id)) { send_error(*p, 404, "unknown model: " + model_id); return nullptr; }
        std::string err;
        if (!st.reg.load(model_id, err)) { send_error(*p, 500, err); return nullptr; }
        pool = st.reg.get(model_id);
    }
    if (pool->failed()) { send_error(*p, 503, "model failed to load: " + pool->init_error()); return nullptr; }
    if (!pool->ready()) { send_error(*p, 503, "model warming up: " + model_id, "Retry-After", "2"); return nullptr; }
    return pool;
}

// Serialize one result per the `return` mode; runs on the loop thread.
void respond_result(Pending& p, InferResult& r, const std::string& model_id, const std::string& ret, int image_id, bool coco91, bool names) {
    p.model = model_id; p.worker = r.worker_id; p.has_stages = true;
    const double st7[7] = {r.queue_ms, r.decode_ms, r.pre_ms, r.infer_ms, r.post_ms, r.encode_ms, r.total_ms};
    for (int i = 0; i < 7; ++i) p.stages[i] = std::round(st7[i] * 1000.0) / 1000.0;
    if (r.http_status != 200) { send_error(p, r.http_status, r.error, r.http_status == 503 ? "Retry-After" : "", r.http_status == 503 ? "1" : ""); return; }
    p.res->writeHeader("X-Infer-Ms", std::to_string(r.infer_ms));
    if (ret == "txt") send(p, 200, result_txt(r), "text/plain");
    else if (ret == "coco") send_json(p, 200, result_coco(r, image_id, coco91));
    else if (ret == "annotated") send(p, 200, std::string(r.annotated_jpg.begin(), r.annotated_jpg.end()), "image/jpeg",
                                      "X-Detections", std::to_string(r.dets.size()));
    else send_json(p, 200, result_json(r, model_id, p.rid, names));
}

Job make_job(ServerState& st, const std::shared_ptr<Pending>& p, std::string&& image, const InferParams& params) {
    Job j; j.request_id = p->rid; j.image = std::move(image); j.params = params;
    j.enqueued = std::chrono::steady_clock::now();
    j.deadline = j.enqueued + std::chrono::milliseconds(st.cfg.request_timeout_ms);
    return j;
}

// tracker factory for video / stream requests ("" or unknown -> none)
static std::shared_ptr<track::Tracker> make_tracker(const std::string& mode, double fps, std::string* err = nullptr) {
    if (mode.empty()) return nullptr;
    track::TrackerConfig tc;
    if (!track::parse_tracker_kind(mode, tc.kind)) { if (err) *err = "unknown track mode: " + mode + " (botsort|bytetrack|off)"; return nullptr; }
    tc.fps = (fps > 1.0 && fps < 1000.0) ? fps : 30.0;
    return std::make_shared<track::Tracker>(tc);
}

// ---- WebSocket stream: keep-latest backpressure per connection ----
struct WsSession {
    std::string model_id; InferParams params; std::string ret = "json";
    std::shared_ptr<track::Tracker> tracker;   // set by ?track= at upgrade or a text {"track": ...} message
    std::atomic<bool> busy{false}, closed{false};
    std::string pending; bool has_pending = false; uint64_t seq = 0, pending_seq = 0, dropped = 0;
    uWS::WebSocket<false, true, WsSession*>* ws = nullptr; uWS::Loop* loop = nullptr;
};
using Ws = uWS::WebSocket<false, true, WsSession*>;

void ws_submit(ServerState& st, std::shared_ptr<WsSession> s, std::string&& frame, uint64_t seq);

void ws_on_result(ServerState& st, std::shared_ptr<WsSession> s, uint64_t seq, InferResult&& r) {
    // on the loop thread
    if (s->closed) return;
    s->busy = false;
    nlohmann::json j = r.http_status == 200 ? result_json(r, s->model_id, std::to_string(seq), true)
                                            : nlohmann::json{{"error", r.error}, {"code", r.http_status}};
    j["seq"] = seq; j["dropped"] = s->dropped;
    if (s->ret == "annotated" && r.http_status == 200 && !r.annotated_jpg.empty())
        s->ws->send(std::string_view(reinterpret_cast<const char*>(r.annotated_jpg.data()), r.annotated_jpg.size()), uWS::OpCode::BINARY);
    s->ws->send(j.dump(), uWS::OpCode::TEXT);
    if (s->has_pending) { s->has_pending = false; ws_submit(st, s, std::move(s->pending), s->pending_seq); }
}

void ws_submit(ServerState& st, std::shared_ptr<WsSession> s, std::string&& frame, uint64_t seq) {
    WorkerPool* pool = st.reg.get(s->model_id);
    if (!pool || !pool->ready()) { s->ws->send(nlohmann::json{{"error", "model not ready"}, {"seq", seq}}.dump(), uWS::OpCode::TEXT); return; }
    s->busy = true;
    Job j; j.request_id = "ws-" + std::to_string(seq); j.image = std::move(frame); j.params = s->params;
    j.tracker = s->tracker;
    j.enqueued = std::chrono::steady_clock::now(); j.deadline = j.enqueued + std::chrono::milliseconds(st.cfg.request_timeout_ms);
    uWS::Loop* loop = s->loop; ServerState* stp = &st;
    j.done = [stp, s, seq, loop](InferResult&& r) mutable {
        auto rp = std::make_shared<InferResult>(std::move(r));
        loop->defer([stp, s, seq, rp]() { ws_on_result(*stp, s, seq, std::move(*rp)); });
    };
    if (!pool->submit(std::move(j))) {
        s->busy = false; pool->metrics.dropped_frames.fetch_add(1);
        s->ws->send(nlohmann::json{{"error", "queue full"}, {"code", 503}, {"seq", seq}}.dump(), uWS::OpCode::TEXT);
    }
}

} // namespace

void request_stop(ServerState& st) {
    st.stopping = true;
    std::lock_guard<std::mutex> g(st.loops_m);
    for (auto& c : st.closers) { auto fn = c.second; c.first->defer([fn]() { fn(); }); }
}

namespace {
bool path_exempt(const ServerConfig& cfg, const std::string& path) {
    for (const auto& e : cfg.auth_exempt) if (path == e) return true;
    return false;
}

// Auth + rate-limit gate shared by the HTTP routes and the WebSocket upgrade. Returns false after
// answering the request (401 / 429) itself.
bool gate_request(ServerState& st, Pending& p, Req* req, const RouteSpec& spec) {
    const bool exempt = path_exempt(st.cfg, p.path);
    if (spec.auth && !st.cfg.api_keys.empty() && !exempt) {
        const std::string key = auth::presented_key(req->getHeader("x-api-key"), req->getHeader("authorization"));
        if (!auth::authorized(st.cfg.api_keys, key)) {
            st.counters.auth_failed.fetch_add(1);
            p.hdrs.emplace_back("WWW-Authenticate", "Bearer realm=\"yolomaster\"");
            send_error(p, 401, "missing or invalid API key (X-API-Key or Authorization: Bearer)");
            return false;
        }
        p.client = "k:" + std::to_string(std::hash<std::string>{}(key));   // one bucket per key, never the key itself
    }
    if (spec.rate_limited && st.cfg.rate_limit.rps > 0 && !exempt) {
        const auto v = st.limiter.take(p.client, st.cfg.rate_limit.rps, st.cfg.rate_limit.burst);
        p.hdrs.emplace_back("X-RateLimit-Limit", std::to_string(v.limit));
        p.hdrs.emplace_back("X-RateLimit-Remaining", std::to_string(v.remaining));
        if (!v.ok) {
            st.counters.rate_limited.fetch_add(1);
            p.hdrs.emplace_back("Retry-After", std::to_string(v.retry_after_s));
            send_error(p, 429, "rate limit exceeded");
            return false;
        }
    }
    return true;
}

std::shared_ptr<Pending> begin_request(ServerState& st, uWS::Loop* loop, Res* res, Req* req, const RouteSpec& spec) {
    auto p = std::make_shared<Pending>(res, loop, &st);
    const auto in_rid = req->getHeader("x-request-id");
    p->rid = trace::valid_rid(in_rid) ? std::string(in_rid) : make_rid(st);
    p->trace = trace::parse_traceparent(req->getHeader("traceparent"));
    p->method = std::string(req->getMethod()); p->path = std::string(req->getUrl());
    p->client = std::string(res->getRemoteAddressAsText());
    res->onAborted([p] { p->aborted = true; });
    if (!gate_request(st, *p, req, spec)) return nullptr;
    return p;
}

RouteSpec R(const char* method, const char* path, const char* op, const char* summary) {
    RouteSpec r; r.method = method; r.path = path; r.op_id = op; r.summary = summary; return r;
}
QueryParam Q(const char* name, const char* type, const char* desc, std::vector<std::string> en = {}) {
    QueryParam q; q.name = name; q.type = type; q.desc = desc; q.enum_values = std::move(en); return q;
}
const std::vector<QueryParam>& infer_params() {
    static const std::vector<QueryParam> v = {
        Q("model", "string", "model id (optional when exactly one model is configured)"),
        Q("conf", "number", "confidence threshold"), Q("iou", "number", "NMS IoU threshold"),
        Q("max_det", "integer", "detections cap"), Q("multi_label", "boolean", "one detection per class above conf per anchor"),
        Q("slicing", "string", "Sparse SAHI per request", {"off", "dense", "sparse"}), Q("tile_size", "integer", "tile edge in source px"),
        Q("masks", "string", "seg: composite masks into the annotated image", {"overlay"}),
        Q("mask_coeffs", "boolean", "seg: include raw mask coefficients"), Q("names", "boolean", "include class names (default true)"),
        Q("quality", "integer", "annotated JPEG quality")};
    return v;
}
} // namespace

bool run_http_loop(ServerState& st, int loop_index, std::atomic<int>& bound_count) {
    const ServerConfig& cfg = st.cfg;
    const size_t max_body = size_t(cfg.max_body_mb) << 20;
    uWS::App app;
    uWS::Loop* loop = uWS::Loop::get();

    // ---- the route table: registration and GET /openapi.json read the same entries ----
    std::vector<RouteSpec> table;
    using Handler = std::function<void(std::shared_ptr<Pending>, Req*)>;
    auto add_route = [&](RouteSpec spec, Handler fn) {
        table.push_back(spec);
        auto h = [&st, loop, spec, fn](Res* res, Req* req) {
            auto p = begin_request(st, loop, res, req, spec);
            if (!p) return;
            fn(p, req);
        };
        if (spec.method == "GET") app.get(spec.path, h);
        else if (spec.method == "POST") app.post(spec.path, h);
        else if (spec.method == "OPTIONS") app.options(spec.path, h);
        else if (spec.method == "ANY") app.any(spec.path, h);
    };

    add_route([]{ auto r = R("GET", "/", "serviceInfo", "service, versions and the endpoint list"); r.auth = false; r.responses = {{200, "ServiceInfo"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        nlohmann::json eps = nlohmann::json::array();
        for (const auto& r : table) if (!r.hidden && r.method != "OPTIONS" && r.method != "ANY") eps.push_back(r.method == "WS" ? r.path + " (ws)" : r.path);
        send_json(*p, 200, {{"service", "yolomaster-edge api"}, {"version", YM_SERVER_VERSION}, {"runtime", YM_VERSION}, {"commit", YM_GIT_COMMIT},
                            {"docs", "/docs/API.md"}, {"openapi", "/openapi.json"}, {"auth", !st.cfg.api_keys.empty()}, {"endpoints", eps}});
    });
    add_route([]{ auto r = R("GET", "/healthz", "healthz", "process liveness"); r.auth = false; r.rate_limited = false; r.responses = {{200, "Health"}, {503, "Health"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        send_json(*p, st.stopping ? 503 : 200, {{"status", st.stopping ? "stopping" : "ok"}, {"uptime_s", st.uptime_s()}});
    });
    add_route([]{ auto r = R("GET", "/readyz", "readyz", "all preloaded models warmed up"); r.auth = false; r.rate_limited = false; r.responses = {{200, "Ready"}, {503, "Ready"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        std::string why; const bool ok = !st.stopping && st.reg.ready(why);
        send_json(*p, ok ? 200 : 503, {{"ready", ok}, {"reason", ok ? "" : why}});
    });
    add_route([]{ auto r = R("GET", "/v1/models", "listModels", "model cards"); r.responses = {{200, "ModelsList"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& s : st.reg.specs()) {
            WorkerPool* pool = st.reg.get(s.id);
            nlohmann::json j = pool ? pool->info() : nlohmann::json(s);
            j["loaded"] = pool != nullptr;
            arr.push_back(j);
        }
        send_json(*p, 200, {{"models", arr}});
    });
    add_route([]{ auto r = R("POST", "/v1/models/:id/load", "loadModel", "start a model's worker pool"); r.responses = {{202, "LoadResult"}, {404, "Error"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        const std::string id(req->getParameter("id"));
        std::string err;
        if (!st.reg.load(id, err)) { send_error(*p, 404, err); return; }
        send_json(*p, 202, {{"model", id}, {"loaded", true}, {"ready", st.reg.get(id)->ready()}});
    });
    add_route([]{ auto r = R("POST", "/v1/models/:id/unload", "unloadModel", "stop a model's worker pool"); r.responses = {{200, "LoadResult"}, {404, "Error"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        const std::string id(req->getParameter("id"));
        if (!st.reg.unload(id)) { send_error(*p, 404, "model not loaded: " + id); return; }
        send_json(*p, 200, {{"model", id}, {"loaded", false}});
    });
    add_route([]{ auto r = R("GET", "/v1/stats", "stats", "rolling latency percentiles per model and server counters"); r.responses = {{200, "Stats"}}; return r; }(),
              [&](std::shared_ptr<Pending> p, Req*) { send_json(*p, 200, stats_json(st.reg, st.uptime_s(), &st.counters)); });
    add_route([]{ auto r = R("GET", "/metrics", "metrics", "Prometheus text exposition"); r.response_ctype = "text/plain"; r.responses = {{200, "Prometheus text exposition (version 0.0.4)"}}; return r; }(),
              [&](std::shared_ptr<Pending> p, Req*) { send(*p, 200, prometheus_text(st.reg, st.cfg, st.uptime_s(), &st.counters), "text/plain; version=0.0.4"); });
    add_route([]{ auto r = R("GET", "/openapi.json", "openapi", "this API as an OpenAPI 3.1 document, rendered from the route table"); r.auth = false; r.responses = {{200, "OpenAPI 3.1 document"}}; return r; }(),
              [&](std::shared_ptr<Pending> p, Req*) { send_json(*p, 200, openapi_json(table, YM_SERVER_VERSION, YM_VERSION, !st.cfg.api_keys.empty())); });
    add_route([]{ auto r = R("OPTIONS", "/*", "preflight", "CORS preflight"); r.auth = false; r.rate_limited = false; r.hidden = true; return r; }(),
              [&](std::shared_ptr<Pending> p, Req*) {
        p->res->writeHeader("Access-Control-Max-Age", "86400");
        send(*p, 204, "", "text/plain");
    });

    // ---- single image ----
    add_route([]{ auto r = R("POST", "/v1/infer", "infer", "one image, one result"); r.query = infer_params(); r.query.insert(r.query.begin() + 1, Q("return", "string", "response format", {"json", "txt", "coco", "annotated"})); r.query.push_back(Q("image_id", "integer", "coco: image_id")); r.query.push_back(Q("coco91", "boolean", "coco: 91-id category map")); r.request_ctypes = {"image/*", "application/octet-stream", "multipart/form-data"}; r.responses = {{200, "InferResult"}, {400, "Error"}, {404, "Error"}, {413, "Error"}, {503, "Error"}, {504, "Error"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        const std::string ctype(req->getHeader("content-type"));
        const std::string ret = qs(req, "return", "json");
        if (ret != "json" && ret != "txt" && ret != "coco" && ret != "annotated") { send_error(*p, 400, "return must be json|txt|coco|annotated"); return; }
        const InferParams params = params_from(req, ret);
        const int image_id = qi(req, "image_id", 0); const bool coco91 = qb(req, "coco91", false); const bool names = qb(req, "names", true);
        read_body(p, max_body, [&st, p, pool, ctype, ret, params, model_id, image_id, coco91, names](std::string&& body) {
            std::string img, fname;
            if (!extract_image(ctype, body, img, fname)) { send_error(*p, 400, "no image in request body (send raw bytes or multipart field 'file')"); return; }
            Job j = make_job(st, p, std::move(img), params);
            j.done = [p, model_id, ret, image_id, coco91, names](InferResult&& r) {
                auto rp = std::make_shared<InferResult>(std::move(r));
                deliver(p, [rp, model_id, ret, image_id, coco91, names](Pending& pp) { respond_result(pp, *rp, model_id, ret, image_id, coco91, names); });
            };
            if (!pool->submit(std::move(j))) send_error(*p, 503, "queue full for model " + model_id, "Retry-After", "1");
        });
    });

    // ---- multi-file: K independent batch-1 jobs spread over the pool ----
    add_route([]{ auto r = R("POST", "/v1/infer/batch", "inferBatch", "K files in one multipart body, K independent batch-1 jobs"); r.query = infer_params(); r.request_ctypes = {"multipart/form-data"}; r.responses = {{200, "BatchResult"}, {400, "Error"}, {404, "Error"}, {413, "Error"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        const std::string ctype(req->getHeader("content-type"));
        const InferParams params = params_from(req, "json");
        const bool names = qb(req, "names", true);
        read_body(p, max_body, [&st, p, pool, ctype, params, model_id, names](std::string&& body) {
            auto files = extract_all(ctype, body);
            if (files.empty()) { send_error(*p, 400, "multipart/form-data with one or more file parts required"); return; }
            struct Agg { std::mutex m; std::vector<nlohmann::json> out; size_t remaining; };
            auto agg = std::make_shared<Agg>(); agg->out.resize(files.size()); agg->remaining = files.size();
            for (size_t i = 0; i < files.size(); ++i) {
                Job j = make_job(st, p, std::move(files[i].second), params);
                j.request_id = p->rid + "-" + std::to_string(i);
                const std::string fname = files[i].first;
                j.done = [p, agg, i, fname, model_id, names](InferResult&& r) {
                    nlohmann::json one = r.http_status == 200 ? result_json(r, model_id, p->rid + "-" + std::to_string(i), names)
                                                              : nlohmann::json{{"error", r.error}, {"code", r.http_status}};
                    one["file"] = fname;
                    bool last = false;
                    { std::lock_guard<std::mutex> g(agg->m); agg->out[i] = std::move(one); last = (--agg->remaining == 0); }
                    if (last) deliver(p, [agg](Pending& pp) { send_json(pp, 200, {{"request_id", pp.rid}, {"count", agg->out.size()}, {"results", agg->out}}); });
                };
                if (!pool->submit(std::move(j))) {
                    // fail this part immediately; the aggregate still completes
                    InferResult r; r.http_status = 503; r.error = "queue full";
                    j.done(std::move(r));
                }
            }
        });
    });

    // ---- bench: probe run on one worker of a loaded model (yolomaster-bench/v1 JSON) ----
    add_route([]{ auto r = R("POST", "/v1/bench", "bench", "probe run on one worker of a loaded model (yolomaster-bench/v1)"); r.query = {Q("model", "string", "model id"), Q("warmup", "integer", "untimed forwards (default 10)"), Q("iters", "integer", "timed forwards (default 50)")}; r.request_ctypes = {"application/octet-stream"}; r.responses = {{200, "BenchResult"}, {404, "Error"}, {503, "Error"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        auto breq = std::make_shared<BenchRequest>();
        breq->warmup = std::max(0, std::min(1000, qi(req, "warmup", 10)));
        breq->iters = std::max(1, std::min(10000, qi(req, "iters", 50)));
        read_body(p, max_body, [&st, p, pool, breq, model_id](std::string&&) {
            Job j = make_job(st, p, std::string(), InferParams{});
            j.bench = breq;
            j.deadline = j.enqueued + std::chrono::seconds(600);   // a bench run is long by design
            j.done = [p, model_id](InferResult&& r) {
                auto rp = std::make_shared<InferResult>(std::move(r));
                deliver(p, [rp, model_id](Pending& pp) {
                    if (rp->http_status != 200) { send_error(pp, rp->http_status, rp->error); return; }
                    rp->bench_json["request_id"] = pp.rid;
                    rp->bench_json["model"]["id"] = model_id;
                    send_json(pp, 200, rp->bench_json);
                });
            };
            if (!pool->submit(std::move(j))) send_error(*p, 503, "queue full", "Retry-After", "1");
        });
    });

    // ---- video upload -> NDJSON stream of per-frame results ----
    add_route([]{ auto r = R("POST", "/v1/video", "video", "video file upload, NDJSON stream of per-frame results"); r.query = infer_params(); r.query.push_back(Q("every", "integer", "process every Nth frame")); r.query.push_back(Q("max_frames", "integer", "stop after N processed frames")); r.query.push_back(Q("track", "string", "multi-object tracking, track_id per detection", {"off", "botsort", "bytetrack"})); r.request_ctypes = {"application/octet-stream", "video/*"}; r.response_ctype = "application/x-ndjson"; r.responses = {{200, "NDJSON: one InferResult per frame with frame, then {done, frames, decoded, track?}"}, {404, "Error"}, {501, "Error"}}; return r; }(), [&](std::shared_ptr<Pending> p, Req* req) {
#ifndef HAVE_VIDEOIO
        send_error(*p, 501, "server built without OpenCV videoio (PORTABLE build)");
        return;
#else
        std::string model_id;
        WorkerPool* pool = resolve_pool(st, req, p, model_id);
        if (!pool) return;
        const InferParams params = params_from(req, "json");
        const int every = std::max(1, qi(req, "every", 1));
        const int max_frames = qi(req, "max_frames", 0);
        const bool names = qb(req, "names", true);
        read_body(p, max_body, [&st, p, pool, params, every, max_frames, model_id, names](std::string&& body) {
            namespace fs = std::filesystem;
            const fs::path tmp = fs::temp_directory_path() / ("ym-video-" + p->rid + ".bin");
            { std::ofstream f(tmp, std::ios::binary); f.write(body.data(), std::streamsize(body.size())); }
            p->res->writeStatus("200 OK"); p->res->writeHeader("Content-Type", "application/x-ndjson");
            p->res->writeHeader("X-Request-Id", p->rid); cors_headers(p->res, st.cfg);
            // decoder thread: sequential frames, one in flight, order preserved
            std::thread([&st, p, pool, params, every, max_frames, model_id, names, tmp]() {
                cv::VideoCapture cap(tmp.string());
                nlohmann::json tail = {{"done", true}};
                if (!cap.isOpened()) { tail["error"] = "cannot decode video"; }
                std::string terr;
                std::shared_ptr<track::Tracker> tracker = make_tracker(params.track, cap.isOpened() ? cap.get(cv::CAP_PROP_FPS) : 30.0, &terr);
                if (!terr.empty()) tail["error"] = terr;
                if (tracker) tail["track"] = track::tracker_kind_name(tracker->config().kind);
                int idx = 0, sent = 0;
                cv::Mat frame;
                while (cap.isOpened() && terr.empty() && !p->aborted && cap.read(frame)) {
                    const int fi = idx++;
                    if (fi % every) continue;
                    if (max_frames > 0 && sent >= max_frames) break;
                    if (!frame.isContinuous()) frame = frame.clone();
                    Job j; j.request_id = p->rid + "-f" + std::to_string(fi); j.params = params; j.tracker = tracker;
                    j.raw_w = frame.cols; j.raw_h = frame.rows; j.image.assign(reinterpret_cast<const char*>(frame.data), size_t(frame.total()) * 3);
                    j.enqueued = std::chrono::steady_clock::now(); j.deadline = j.enqueued + std::chrono::milliseconds(st.cfg.request_timeout_ms);
                    std::promise<InferResult> pr; auto fut = pr.get_future();
                    j.done = [&pr](InferResult&& r) { pr.set_value(std::move(r)); };
                    if (!pool->submit(std::move(j))) { tail["error"] = "queue full"; break; }
                    InferResult r = fut.get();
                    nlohmann::json line = r.http_status == 200 ? result_json(r, model_id, p->rid, names) : nlohmann::json{{"error", r.error}, {"code", r.http_status}};
                    line["frame"] = fi;
                    const std::string s = line.dump() + "\n";
                    p->loop->defer([p, s]() { if (!p->aborted) p->res->write(s); });
                    ++sent;
                }
                tail["frames"] = sent; tail["decoded"] = idx;
                const std::string s = tail.dump() + "\n";
                p->loop->defer([p, s]() { if (!p->aborted) { p->res->write(s); p->res->end(); access_log(*p->st, *p, 200, 0); } });
                std::error_code ec; fs::remove(tmp, ec);
            }).detach();
        });
#endif
    });

    // ---- WebSocket stream ----
    const RouteSpec ws_spec = []{ auto r = R("WS", "/v1/stream", "stream", "binary frames in, JSON per frame out, keep-latest backpressure");
        r.query = {Q("model", "string", "model id"), Q("conf", "number", ""), Q("iou", "number", ""), Q("max_det", "integer", ""),
                   Q("return", "string", "json or annotated (JPEG binary before the JSON)", {"json", "annotated"}),
                   Q("track", "string", "multi-object tracking for this connection", {"off", "botsort", "bytetrack"}), Q("fps", "number", "source fps for the tracker buffer")};
        r.responses = {{101, "WebSocket upgrade"}, {404, "unknown model"}}; return r; }();
    table.push_back(ws_spec);
    app.ws<WsSession*>("/v1/stream", {
        .compression = uWS::DISABLED,
        .maxPayloadLength = unsigned(cfg.ws_max_payload_mb) << 20,
        .idleTimeout = 120,
        .maxBackpressure = 4u << 20,
        .closeOnBackpressureLimit = false,
        .resetIdleTimeoutOnSend = true,
        .sendPingsAutomatically = true,
        .upgrade = [&](Res* res, Req* req, us_socket_context_t* ctx) {
            {   // the same API-key and rate-limit gate as the HTTP routes (one token per upgrade)
                auto gp = begin_request(st, loop, res, req, ws_spec);
                if (!gp) return;
            }
            std::string model_id = qs(req, "model");
            if (model_id.empty()) { auto specs = st.reg.specs(); if (specs.size() == 1) model_id = specs[0].id; }
            std::string err;
            if (model_id.empty() || !st.reg.has_spec(model_id) || !st.reg.load(model_id, err)) {
                res->writeStatus("404 Not Found")->end("unknown model");
                return;
            }
            auto* s = new WsSession();
            s->model_id = model_id; s->params = params_from(req, qs(req, "return", "json")); s->ret = qs(req, "return", "json");
            s->params.annotated = s->ret == "annotated";
            s->tracker = make_tracker(s->params.track, qf(req, "fps", 30.0f));
            res->template upgrade<WsSession*>(std::move(s), req->getHeader("sec-websocket-key"), req->getHeader("sec-websocket-protocol"),
                                             req->getHeader("sec-websocket-extensions"), ctx);
        },
        .open = [&](Ws* ws) {
            WsSession* s = *ws->getUserData(); s->ws = ws; s->loop = loop;
            ws->send(nlohmann::json{{"hello", "yolomaster stream"}, {"model", s->model_id}, {"protocol", "binary frame in -> json per frame out; text json {conf,iou,max_det,return} updates params"}}.dump(), uWS::OpCode::TEXT);
        },
        .message = [&](Ws* ws, std::string_view msg, uWS::OpCode op) {
            WsSession* raw = *ws->getUserData();
            if (op == uWS::OpCode::TEXT) {
                try {
                    auto j = nlohmann::json::parse(msg);
                    if (j.contains("conf")) raw->params.conf = j["conf"].get<float>();
                    if (j.contains("iou")) raw->params.iou = j["iou"].get<float>();
                    if (j.contains("max_det")) raw->params.max_det = j["max_det"].get<int>();
                    if (j.contains("return")) { raw->ret = j["return"].get<std::string>(); raw->params.annotated = raw->ret == "annotated"; }
                    if (j.contains("track")) {          // "botsort" | "bytetrack" start a fresh tracker, "off" stops it
                        std::string mode = j["track"].get<std::string>();
                        if (mode == "off") mode.clear();
                        std::string terr;
                        raw->params.track = mode;
                        raw->tracker = make_tracker(mode, j.value("fps", 30.0), &terr);
                        if (!terr.empty()) throw std::runtime_error(terr);
                    }
                    ws->send(nlohmann::json{{"ok", true}, {"params", {{"conf", raw->params.conf}, {"iou", raw->params.iou}, {"max_det", raw->params.max_det}, {"return", raw->ret},
                                                              {"track", raw->tracker ? track::tracker_kind_name(raw->tracker->config().kind) : "off"}}}}.dump(), uWS::OpCode::TEXT);
                } catch (const std::exception& e) { ws->send(nlohmann::json{{"error", std::string("bad json: ") + e.what()}}.dump(), uWS::OpCode::TEXT); }
                return;
            }
            // binary frame: keep-latest
            // shared_ptr alias so completions after close are harmless
            static thread_local std::map<WsSession*, std::shared_ptr<WsSession>> owners;
            auto& sp = owners[raw]; if (!sp) sp = std::shared_ptr<WsSession>(raw, [](WsSession*) {});
            const uint64_t seq = ++raw->seq;
            if (raw->busy) {
                if (raw->has_pending) { ++raw->dropped; WorkerPool* pool = st.reg.get(raw->model_id); if (pool) pool->metrics.dropped_frames.fetch_add(1); }
                raw->pending.assign(msg.data(), msg.size()); raw->has_pending = true; raw->pending_seq = seq;
                return;
            }
            ws_submit(st, sp, std::string(msg), seq);
        },
        .close = [&](Ws* ws, int, std::string_view) {
            WsSession* s = *ws->getUserData();
            s->closed = true; s->ws = nullptr;
            // deferred delete: in-flight completions hold the alias and check `closed`
            uWS::Loop::get()->defer([s]() { /* leak-free after all defers drained */ delete s; });
        }
    });

    add_route([]{ auto r = R("ANY", "/*", "notFound", "404"); r.auth = false; r.rate_limited = false; r.hidden = true; return r; }(),
              [&](std::shared_ptr<Pending> p, Req* req) { send_error(*p, 404, "no such route: " + std::string(req->getUrl())); });

    bool ok = false;
    app.listen(cfg.host, cfg.port, [&](us_listen_socket_t* ls) {
        ok = ls != nullptr;
        if (ok) {
            ++bound_count;
            std::lock_guard<std::mutex> g(st.loops_m);
            st.closers.emplace_back(loop, [&app, ls]() { us_listen_socket_close(0, ls); app.close(); });
        }
    });
    if (!ok) { std::cerr << "[http] loop " << loop_index << ": cannot bind " << cfg.host << ":" << cfg.port << "\n"; return false; }
    if (loop_index == 0) std::cerr << now_iso() << " [http] listening on http://" << cfg.host << ":" << cfg.port << " (" << cfg.loop_threads << " loop threads)\n";
    app.run();
    return true;
}

} // namespace yolomaster::server
