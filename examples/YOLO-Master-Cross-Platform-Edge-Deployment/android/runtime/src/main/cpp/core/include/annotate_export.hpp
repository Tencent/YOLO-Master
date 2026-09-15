// Annotation export over the shared runtime types - mask tracing, IR building, and the
// streaming per-format sink used by the CLI and the Windows GUI. The cv-free writers live in
// annotate.hpp; this layer needs OpenCV (imgproc contours + the proto math).
#pragma once
#include "yolomaster.hpp"
#include "annotate.hpp"
#include <set>

namespace yolomaster {

// Mask polygons for ONE detection, ORIGINAL px. Port of the Mac tracer
// (Detector.maskPolygons, mac/Sources/YOLOMasterKit/Detector.swift:613-790) with
// cv::findContours + cv::approxPolyDP standing in for the hand-rolled Moore trace /
// Douglas-Peucker (same threshold-0.5 / box-restrict / outer-only / area-sort / cap
// semantics; a <=2-cell speckle dies at the >=3-points filter either way).
// Returns flat x,y rings sorted by area desc, at most max_polygons; empty when the det has
// no usable coeffs or the mask thresholds to nothing inside the box.
std::vector<std::vector<float>> seg_polygons(
    const Detection& det, const std::vector<float>& proto, int pc, int ph, int pw,
    const LetterboxInfo& lb, int imgsz,
    float threshold = 0.5f, float epsilon_px = 2.0f, int max_polygons = 8);

// Post-NMS detections -> IR instances (WYSIWYG - pass exactly what the preview shows).
// include_polygons = the run's dialect (seg model with mask data available). Instances whose
// mask is unavailable (coeff-less tile detection) or empty in a polygon export fall back to
// the box as a 4-point polygon.
std::vector<annot::Instance> annotation_instances(
    const std::vector<Detection>& dets, bool include_polygons,
    const std::vector<float>& proto, int pc, int ph, int pw,
    const LetterboxInfo& lb, int imgsz);

// Streaming per-format emitter shared by the GUI folder/video export loops and the CLI.
//   YOLO: <labels_dir>/<stem>.txt per add() + classes.txt at finish()
//   VOC : <labels_dir>/<stem>.xml per add()
//   COCO: accumulates; finish() writes ONE json to <coco_path>
// Duplicate stems within one run (e.g. 1.jpg + 1.png in a folder) are suffixed -2, -3, ...
// instead of silently overwriting the earlier file.
// Files are written binary (LF endings). add() returns false (and finish() reports the
// error) on the first failed write.
class AnnotationSink {
public:
    AnnotationSink(annot::Format fmt, std::string labels_dir, std::string coco_path,
                   std::vector<std::string> names, bool seg_dialect);
    // coco_file_name: the file_name recorded in the COCO doc (may carry a relative dir like
    // "frames/clip_000090.jpg"); coco_id: explicit image id (0 = 1-based sequence).
    bool add(const annot::Image& img, const std::string& coco_file_name, int coco_id = 0);
    struct Result { int images = 0, instances = 0; std::string error; };
    Result finish();
private:
    annot::Format fmt_;
    std::string labels_dir_, coco_path_, error_;
    std::vector<std::string> names_;
    bool seg_dialect_;
    int images_ = 0, instances_ = 0;
    std::set<std::string> used_names_;   // stems already written this run (collision guard)
    std::vector<annot::Image> coco_images_;
    std::vector<std::string> coco_names_;
    std::vector<int> coco_ids_;
};

// "stem", or "stem-2", "stem-3", ... if already in `used`; inserts the result. The shared
// collision guard for every per-source-file writer (annotated jpgs, --save-txt, labels):
// two sources with the same stem (1.jpg + 1.png) must never overwrite each other's output.
std::string unique_stem(std::set<std::string>& used, const std::string& stem);

// JPEG writer via stb (quality 90) - shared so the GUI's export paths skip cv::imgcodecs
// exactly like the CLI does.
bool write_jpg(const std::string& path, const cv::Mat& bgr);

} // namespace yolomaster
