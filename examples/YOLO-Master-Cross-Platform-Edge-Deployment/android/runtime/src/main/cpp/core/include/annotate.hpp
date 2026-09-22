// Annotation export - format-neutral IR + pure writers (YOLO TXT / COCO JSON / Pascal VOC
// XML). The C++ port of mac/Sources/YOLOMasterKit/AnnotationExport.swift's AnnotationWriter,
// line-for-line semantics. OpenCV-free on purpose so the writers are testable with a bare
// compiler; the repo vendors no JSON/XML library and these hand-emitters keep it that way.
//
// Dialect rule (WYSIWYG): the FILE dialect (det boxes vs seg polygons) follows what the
// preview shows - seg polygons only when the model is seg AND mask data exists for the run
// (single-pass, or sliced with keep_global_masks). Within a polygon-dialect export, an
// instance with no usable mask (coeff-less tile detection, or a mask that thresholds to
// nothing) emits its BOX as a 4-point polygon so every line/segmentation entry stays
// structurally valid. Pascal VOC has no standard segmentation field and always writes boxes.
//
// Callers write the returned bodies with std::ios::binary so line endings stay LF.
#pragma once
#include <string>
#include <vector>

namespace yolomaster { namespace annot {

enum class Format { YoloTXT, CocoJSON, PascalVOC };

inline const char* label(Format f) {
    switch (f) {
    case Format::YoloTXT:  return "YOLO (TXT)";
    case Format::CocoJSON: return "COCO (JSON)";
    default:               return "Pascal VOC (XML)";
    }
}
inline const char* file_extension(Format f) {
    switch (f) {
    case Format::YoloTXT:  return "txt";
    case Format::CocoJSON: return "json";
    default:               return "xml";
    }
}

// Format-neutral intermediate representation.
struct Instance {
    int cls = 0;
    float score = 0.f;
    float x = 0, y = 0, w = 0, h = 0;            // box, ORIGINAL px, top-left origin
    std::vector<std::vector<float>> polygons;    // flat x,y pairs, ORIGINAL px; empty = box-only
};

struct Image {
    std::string name;                            // stem, no extension
    int width = 0, height = 0;
    std::vector<Instance> instances;
};

// One image's YOLO .txt body. det: "cls cx cy w h" normalized 6dp. seg dialect:
// "cls x1 y1 x2 y2 ..." using the FIRST polygon per instance with >= 3 points (polygons
// arrive area-sorted from the tracer; YOLO-seg has no multi-part syntax), box line when none.
// Empty detections -> empty string (valid negative). Trailing newline iff non-empty.
std::string yolo_lines(const Image& img, bool seg_dialect);

std::string classes_txt(const std::vector<std::string>& names);

// Per-image Pascal VOC XML. Boxes only (VOC has no polygon field); 1-based inclusive ints
// clamped to the image; no score element (strictest-parser compatibility).
std::string voc_xml(const Image& img, const std::vector<std::string>& names,
                    const std::string& folder);

// ONE JSON for the whole set. file_names[i] pairs with images[i] (may carry a relative dir
// like "frames/clip_000090.jpg"). image_ids: optional explicit ids (video: source frame
// index + 1 so ids survive re-export at a different sampling); default 1-based sequence.
// category_id = cls + 1; area = shoelace over the polygons in seg dialect (bbox area when
// zero or in det dialect); "score" is an extension key GT readers ignore. Keys are emitted
// sorted (Mac JSONEncoder .sortedKeys parity).
std::string coco_json(const std::vector<Image>& images,
                      const std::vector<std::string>& file_names,
                      const std::vector<std::string>& names, bool seg_dialect,
                      const std::vector<int>* image_ids = nullptr);

}} // namespace yolomaster::annot
