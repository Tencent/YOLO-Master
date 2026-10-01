#include "routes.hpp"

namespace yolomaster::server {

namespace {
using nlohmann::json;

json schema_components() {
    json c;
    c["Error"] = {{"type", "object"}, {"properties", {{"error", {{"type", "string"}}}, {"code", {{"type", "integer"}}},
                                                       {"request_id", {{"type", "string"}}}}}, {"required", {"error", "code"}}};
    c["Detection"] = {{"type", "object"},
                      {"properties", {{"class_id", {{"type", "integer"}}}, {"conf", {{"type", "number"}}},
                                      {"box", {{"type", "array"}, {"items", {{"type", "number"}}}, {"minItems", 4}, {"maxItems", 4},
                                               {"description", "x1, y1, x2, y2 in original-image pixels"}}},
                                      {"name", {{"type", "string"}}},
                                      {"track_id", {{"type", "integer"}, {"description", "present when track= is active"}}},
                                      {"mask_coeffs", {{"type", "array"}, {"items", {{"type", "number"}}}}}}},
                      {"required", {"class_id", "conf", "box"}}};
    c["Timings"] = {{"type", "object"}, {"description", "milliseconds per stage, server side"},
                    {"properties", {{"queue", {{"type", "number"}}}, {"decode", {{"type", "number"}}}, {"pre", {{"type", "number"}}},
                                    {"infer", {{"type", "number"}}}, {"post", {{"type", "number"}}}, {"encode", {{"type", "number"}}},
                                    {"total", {{"type", "number"}}}}}};
    c["InferResult"] = {{"type", "object"},
                        {"properties", {{"request_id", {{"type", "string"}}}, {"model", {{"type", "string"}}}, {"backend", {{"type", "string"}}},
                                        {"image", {{"type", "object"}, {"properties", {{"width", {{"type", "integer"}}}, {"height", {{"type", "integer"}}}}}}},
                                        {"params", {{"type", "object"}}}, {"count", {{"type", "integer"}}},
                                        {"detections", {{"type", "array"}, {"items", {{"$ref", "#/components/schemas/Detection"}}}}},
                                        {"timings_ms", {{"$ref", "#/components/schemas/Timings"}}},
                                        {"seg", {{"type", "boolean"}}}, {"worker", {{"type", "integer"}}},
                                        {"slicing", {{"type", "object"}, {"properties", {{"tiles_run", {{"type", "integer"}}}, {"tiles_total", {{"type", "integer"}}}}}}},
                                        {"frame", {{"type", "integer"}, {"description", "/v1/video only"}}}}},
                        {"required", {"request_id", "model", "count", "detections", "timings_ms"}}};
    c["CocoDetection"] = {{"type", "object"}, {"properties", {{"image_id", {{"type", "integer"}}}, {"category_id", {{"type", "integer"}}},
                                                              {"bbox", {{"type", "array"}, {"items", {{"type", "number"}}}}}, {"score", {{"type", "number"}}}}}};
    c["BatchResult"] = {{"type", "object"}, {"properties", {{"request_id", {{"type", "string"}}}, {"count", {{"type", "integer"}}},
                                                            {"results", {{"type", "array"}, {"items", {{"$ref", "#/components/schemas/InferResult"}}}}}}}};
    c["ModelCard"] = {{"type", "object"}, {"properties", {{"id", {{"type", "string"}}}, {"backend", {{"type", "string"}}}, {"ep", {{"type", "string"}}},
                                                          {"imgsz", {{"type", "integer"}}}, {"nc", {{"type", "integer"}}}, {"workers", {{"type", "integer"}}},
                                                          {"loaded", {{"type", "boolean"}}}, {"ready", {{"type", "boolean"}}}}}};
    c["ModelsList"] = {{"type", "object"}, {"properties", {{"models", {{"type", "array"}, {"items", {{"$ref", "#/components/schemas/ModelCard"}}}}}}}};
    c["LoadResult"] = {{"type", "object"}, {"properties", {{"model", {{"type", "string"}}}, {"loaded", {{"type", "boolean"}}}, {"ready", {{"type", "boolean"}}}}}};
    c["Health"] = {{"type", "object"}, {"properties", {{"status", {{"type", "string"}}}, {"uptime_s", {{"type", "number"}}}}}};
    c["Ready"] = {{"type", "object"}, {"properties", {{"ready", {{"type", "boolean"}}}, {"reason", {{"type", "string"}}}}}};
    c["Stats"] = {{"type", "object"}, {"properties", {{"uptime_s", {{"type", "number"}}}, {"models", {{"type", "object"}}}, {"server", {{"type", "object"}}}}}};
    c["BenchResult"] = {{"type", "object"}, {"description", "yolomaster-bench/v1 document (see docs/API.md)"},
                        {"properties", {{"schema_version", {{"type", "string"}}}, {"tool", {{"type", "string"}}}, {"model", {{"type", "object"}}},
                                        {"environment", {{"type", "object"}}}, {"protocol", {{"type", "object"}}}, {"cold", {{"type", "object"}}},
                                        {"request_id", {{"type", "string"}}}, {"worker", {{"type", "integer"}}}}},
                        {"required", {"schema_version", "cold"}}};
    c["ServiceInfo"] = {{"type", "object"}, {"properties", {{"service", {{"type", "string"}}}, {"version", {{"type", "string"}}}, {"runtime", {{"type", "string"}}},
                                                            {"commit", {{"type", "string"}}}, {"endpoints", {{"type", "array"}, {"items", {{"type", "string"}}}}}}}};
    return c;
}

std::string oas_path(std::string p) {   // uWS ":id" -> OpenAPI "{id}"
    std::string out;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] == ':' && (i == 0 || p[i - 1] == '/')) {
            size_t j = i + 1; while (j < p.size() && p[j] != '/') ++j;
            out += "{" + p.substr(i + 1, j - i - 1) + "}"; i = j - 1;
        } else out += p[i];
    }
    return out;
}

json response_for(const std::string& ref, const std::string& ctype) {
    json r;
    if (ref.empty()) { r["description"] = "no content"; return r; }
    const bool is_ref = ref.find(' ') == std::string::npos && ref.find('/') == std::string::npos && !ref.empty() && std::isupper(static_cast<unsigned char>(ref[0]));
    if (is_ref) {
        r["description"] = ref;
        r["content"][ctype]["schema"] = {{"$ref", "#/components/schemas/" + ref}};
    } else {
        r["description"] = ref;
    }
    return r;
}
} // namespace

json openapi_json(const std::vector<RouteSpec>& routes, const std::string& server_version,
                  const std::string& runtime_version, bool auth_enabled) {
    json doc;
    doc["openapi"] = "3.1.0";
    doc["info"] = {{"title", "YOLO-Master Edge inference API"}, {"version", server_version},
                   {"description", "REST + WebSocket inference on ONNX Runtime, TensorRT, ncnn and MNN (runtime " + runtime_version + ")"}};
    doc["components"]["schemas"] = schema_components();
    doc["components"]["securitySchemes"] = {
        {"ApiKeyHeader", {{"type", "apiKey"}, {"in", "header"}, {"name", "X-API-Key"}}},
        {"BearerAuth", {{"type", "http"}, {"scheme", "bearer"}}}};
    json paths = json::object();
    for (const auto& r : routes) {
        if (r.hidden) continue;
        const std::string p = oas_path(r.path);
        json op;
        op["operationId"] = r.op_id;
        op["summary"] = r.summary;
        if (!r.description.empty()) op["description"] = r.description;
        json params = json::array();
        // path parameters
        for (size_t i = 0; i < r.path.size(); ++i) if (r.path[i] == ':') {
            size_t j = i + 1; while (j < r.path.size() && r.path[j] != '/') ++j;
            params.push_back({{"name", r.path.substr(i + 1, j - i - 1)}, {"in", "path"}, {"required", true}, {"schema", {{"type", "string"}}}});
        }
        for (const auto& q : r.query) {
            json s = {{"type", q.type}};
            if (!q.enum_values.empty()) s["enum"] = q.enum_values;
            json pj = {{"name", q.name}, {"in", "query"}, {"required", q.required}, {"schema", s}};
            if (!q.desc.empty()) pj["description"] = q.desc;
            params.push_back(pj);
        }
        if (!params.empty()) op["parameters"] = params;
        if (!r.request_ctypes.empty()) {
            json content = json::object();
            for (const auto& ct : r.request_ctypes) content[ct] = {{"schema", {{"type", "string"}, {"format", "binary"}}}};
            op["requestBody"] = {{"required", true}, {"content", content}};
        }
        json responses = json::object();
        for (const auto& kv : r.responses) responses[std::to_string(kv.first)] = response_for(kv.second, r.response_ctype);
        if (r.auth && auth_enabled) responses["401"] = response_for("Error", "application/json");
        if (r.rate_limited) responses["429"] = response_for("Error", "application/json");
        op["responses"] = responses;
        if (r.auth && auth_enabled) op["security"] = json::array({{{"ApiKeyHeader", json::array()}}, {{"BearerAuth", json::array()}}});
        if (r.method == "WS") {
            paths[p]["get"] = op;
            paths[p]["get"]["x-websocket"] = true;
            continue;
        }
        std::string m = r.method; for (auto& ch : m) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        paths[p][m] = op;
    }
    doc["paths"] = paths;
    if (auth_enabled) doc["security"] = json::array({{{"ApiKeyHeader", json::array()}}, {{"BearerAuth", json::array()}}});
    return doc;
}

} // namespace yolomaster::server
