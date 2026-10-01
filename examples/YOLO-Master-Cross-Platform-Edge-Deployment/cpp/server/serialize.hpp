// Response serializers: JSON (default), YOLO txt ("class conf x1 y1 x2 y2", the CLI's --save-txt
// format), COCO results JSON, plus the Prometheus text exposition for /metrics.
#pragma once
#include <string>
#include <vector>
#include "worker_pool.hpp"
#include "model_registry.hpp"

namespace yolomaster::server {

nlohmann::json result_json(const InferResult& r, const std::string& model_id, const std::string& request_id,
                           bool include_names);
std::string result_txt(const InferResult& r);                          // one detection per line
nlohmann::json result_coco(const InferResult& r, int image_id, bool coco91);   // [{image_id,category_id,bbox,score}]
std::string prometheus_text(ModelRegistry& reg, const ServerConfig& cfg, double uptime_s, const GlobalCounters* gc = nullptr);
nlohmann::json stats_json(ModelRegistry& reg, double uptime_s, const GlobalCounters* gc = nullptr);
int coco80_to_91(int c);

} // namespace yolomaster::server
