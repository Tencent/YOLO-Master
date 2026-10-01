// In-process mAP (COCO-style 0.50:0.95): a faithful port of scripts/eval_map_standalone.py (itself
// bit-compatible with scripts/eval_map.py, the ultralytics DetMetrics matcher). Same matching rule,
// 101-point interpolated AP, same tie handling, so a device can print the number the Linux scripts
// would print for its own detections.
//
// This is the portable core: plain C++17, no OpenCV, no JSON library. It is compiled by the Linux /
// Windows / Jetson CMake build (cpp/CMakeLists.txt, target yolomaster_ccore) and by the Swift
// package (mac/Package.swift, target YOLOMasterCore) from this one directory. The OpenCV-typed
// entry points (Detection -> PredBox) live in cpp/include/map_metrics.hpp.
#pragma once
#include <array>
#include <string>
#include <vector>

namespace yolomaster::metrics {

struct GtBox { double x1 = 0, y1 = 0, x2 = 0, y2 = 0; int cls = 0; };
struct PredBox { double x1 = 0, y1 = 0, x2 = 0, y2 = 0, conf = 0; int cls = 0; };
struct ImageEval {
    std::vector<PredBox> preds;
    std::vector<GtBox> gts;
};
struct ClassAP {
    int cls = 0;
    double ap[10] = {0};     // IoU 0.50, 0.55, ..., 0.95
    int n_gt = 0, n_pred = 0;
    double ap50() const { return ap[0]; }
    double ap5095() const { double s = 0; for (double v : ap) s += v; return s / 10.0; }
};
struct MapResult {
    int images = 0;
    double map50 = 0, map5095 = 0;
    std::vector<ClassAP> per_class;   // classes present in the ground truth, ascending id
};

// YOLO labels ("cls cx cy w h", normalized) or the original VisDrone comma format; a missing or
// empty file yields no boxes and returns true (an image without objects is valid ground truth).
bool load_yolo_labels(const std::string& path, int img_w, int img_h, std::vector<GtBox>& out);
// ultralytics img2label_paths rule when labels_dir is empty ("/images/" -> "/labels/", ".txt"),
// otherwise "<labels_dir>/<stem>.txt".
std::string label_path_for(const std::string& image_path, const std::string& labels_dir);
// What "std::cout << float" prints (6 significant digits, %g) read back as a double: the rounding
// the CLI's --save-txt writer applies, so an in-process score equals scoring the txt dump.
double round6(double v);
MapResult evaluate(const std::vector<ImageEval>& images);
// per-pred correctness matrix (N x 10) for one image; diagnostics only
std::vector<std::array<bool, 10>> debug_tp(const ImageEval& im);

} // namespace yolomaster::metrics
