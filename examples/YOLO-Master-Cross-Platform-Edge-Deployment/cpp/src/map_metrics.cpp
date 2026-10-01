#include "map_metrics.hpp"

namespace yolomaster::metrics {

std::vector<PredBox> from_detections(const std::vector<Detection>& dets, bool txt_rounding) {
    std::vector<PredBox> out;
    out.reserve(dets.size());
    for (const auto& d : dets) {
        PredBox p;
        p.x1 = d.box.x; p.y1 = d.box.y;
        p.x2 = d.box.x + d.box.width;          // float arithmetic, as the txt writer does
        p.y2 = d.box.y + d.box.height;
        p.conf = d.conf;
        p.cls = d.class_id;
        if (txt_rounding) { p.x1 = round6(p.x1); p.y1 = round6(p.y1); p.x2 = round6(p.x2); p.y2 = round6(p.y2); p.conf = round6(p.conf); }
        out.push_back(p);
    }
    return out;
}

} // namespace yolomaster::metrics
