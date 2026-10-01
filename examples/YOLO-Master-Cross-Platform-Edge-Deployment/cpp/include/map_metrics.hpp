// OpenCV-typed entry point of the in-process mAP: Detection -> PredBox. The scorer itself
// (evaluate, labels, rounding) is the portable core in mac/Sources/YOLOMasterCore/metrics_core.hpp,
// shared verbatim with the Swift package.
#pragma once
#include "metrics_core.hpp"
#include "yolomaster.hpp"

namespace yolomaster::metrics {

// Detection -> PredBox in original-image px. `txt_rounding` rounds conf and box edges to the six
// significant digits the CLI's --save-txt writer prints (ostream default), so the in-process score
// equals scoring the txt dump with scripts/eval_map*.py, ties included.
std::vector<PredBox> from_detections(const std::vector<Detection>& dets, bool txt_rounding = false);

} // namespace yolomaster::metrics
