// The route table: one declarative entry per endpoint. http_api.cpp registers the uWebSockets
// handlers from it and GET /openapi.json renders it, so the OpenAPI document cannot drift from
// what is actually served. Handlers live next to the entries in http_api.cpp; this header carries
// only the description.
#pragma once
#include <map>
#include <string>
#include <vector>
#include "json.hpp"

namespace yolomaster::server {

struct QueryParam {
    std::string name, type = "string", desc;   // type: string | integer | number | boolean
    bool required = false;
    std::vector<std::string> enum_values;
};

struct RouteSpec {
    std::string method;            // GET | POST | OPTIONS | WS
    std::string path;              // uWS form, ":id" path parameters
    std::string op_id, summary, description;
    std::vector<QueryParam> query;
    std::vector<std::string> request_ctypes;      // request body media types (POST)
    std::map<int, std::string> responses;         // code -> schema ref ("Error") or free text
    std::string response_ctype = "application/json";
    bool auth = true;              // API key required when keys are configured
    bool rate_limited = true;
    bool hidden = false;           // not in the document
};

// OpenAPI 3.1 document for the table (the WS entry becomes a path with x-websocket: true).
nlohmann::json openapi_json(const std::vector<RouteSpec>& routes, const std::string& server_version,
                            const std::string& runtime_version, bool auth_enabled);

} // namespace yolomaster::server
