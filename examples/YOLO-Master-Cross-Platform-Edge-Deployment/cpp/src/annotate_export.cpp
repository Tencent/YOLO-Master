// Annotation export implementation - see annotate_export.hpp. Parity references:
// Detector.maskPolygons (tracer), annotationInstances + the export orchestrators in
// mac/Sources/YOLOMasterKit/AnnotationExport.swift (the sink replaces the Mac orchestrators
// so three consumers - CLI, GUI folder, GUI video - share one emit path).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "annotate_export.hpp"
#include "stb_image_write.h"
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace yolomaster {

std::vector<std::vector<float>> seg_polygons(
    const Detection& det, const std::vector<float>& proto, int pc, int ph, int pw,
    const LetterboxInfo& lb, int imgsz,
    float threshold, float epsilon_px, int max_polygons) {
    std::vector<std::vector<float>> rings;
    if (proto.empty() || pc <= 0 || ph <= 0 || pw <= 0 || imgsz <= 0) return rings;
    if (static_cast<int>(det.mask_coeffs.size()) != pc) return rings;

    // det.box (original px) <-> proto grid, the same mapping seg_overlay rasterizes with:
    //   grid = orig * (lb.scale * gridDim / imgsz) + pad * gridDim / imgsz
    const float sx = lb.scale_x * pw / imgsz, sy = lb.scale_y * ph / imgsz;
    const float ox0 = lb.pad_x * static_cast<float>(pw) / imgsz;
    const float oy0 = lb.pad_y * static_cast<float>(ph) / imgsz;
    if (sx <= 0.f || sy <= 0.f) return rings;
    auto orig_x = [&](float gx) { return (gx - ox0) / sx; };
    auto orig_y = [&](float gy) { return (gy - oy0) / sy; };

    const float bx0 = det.box.x, by0 = det.box.y;
    const float bx1 = det.box.x + det.box.width, by1 = det.box.y + det.box.height;

    // sub-grid covering the box, expanded by 1 cell, clamped to the proto
    const int gx0 = std::max(0, static_cast<int>(std::floor(bx0 * sx + ox0)) - 1);
    const int gy0 = std::max(0, static_cast<int>(std::floor(by0 * sy + oy0)) - 1);
    const int gx1 = std::min(pw, static_cast<int>(std::ceil(bx1 * sx + ox0)) + 1);
    const int gy1 = std::min(ph, static_cast<int>(std::ceil(by1 * sy + oy0)) + 1);
    if (gx1 <= gx0 || gy1 <= gy0) return rings;
    const int sub_w = gx1 - gx0, sub_h = gy1 - gy0;

    // padded (+1 ring each side) with a guaranteed-false border so every contour closes
    cv::Mat bin(sub_h + 2, sub_w + 2, CV_8UC1, cv::Scalar(0));
    const size_t plane = static_cast<size_t>(ph) * pw;
    const float* co = det.mask_coeffs.data();
    for (int i = 0; i < sub_h; ++i) {
        const int pi = gy0 + i;
        const float cy = orig_y(pi + 0.5f);
        if (cy < by0 || cy > by1) continue;          // preview clips to the box
        uint8_t* row = bin.ptr<uint8_t>(i + 1);
        for (int j = 0; j < sub_w; ++j) {
            const int pj = gx0 + j;
            const float cx = orig_x(pj + 0.5f);
            if (cx < bx0 || cx > bx1) continue;
            float acc = 0.f;
            const size_t base = static_cast<size_t>(pi) * pw + pj;
            for (int k = 0; k < pc; ++k) acc += co[k] * proto[k * plane + base];
            if (1.f / (1.f + std::exp(-acc)) > threshold) row[j + 1] = 255;
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const float img_w = static_cast<float>(lb.orig_w), img_h = static_cast<float>(lb.orig_h);
    const float cx_lo = std::max(0.f, bx0), cx_hi = std::min(img_w, bx1);
    const float cy_lo = std::max(0.f, by0), cy_hi = std::min(img_h, by1);
    std::vector<std::pair<double, std::vector<float>>> scored;
    for (const auto& c : contours) {
        if (c.size() < 3) continue;                  // 1-2 cell speckles die here
        // padded-grid cell centers -> original px, clamped to det.box intersect image
        std::vector<cv::Point2f> pts;
        pts.reserve(c.size());
        for (const cv::Point& p : c) {
            const float gx = static_cast<float>(gx0 + p.x - 1) + 0.5f;
            const float gy = static_cast<float>(gy0 + p.y - 1) + 0.5f;
            pts.emplace_back(std::min(std::max(orig_x(gx), cx_lo), cx_hi),
                             std::min(std::max(orig_y(gy), cy_lo), cy_hi));
        }
        std::vector<cv::Point2f> simplified;
        cv::approxPolyDP(pts, simplified, epsilon_px, /*closed=*/true);
        if (simplified.size() < 3) continue;
        std::vector<float> flat;
        flat.reserve(simplified.size() * 2);
        double area = 0;
        for (size_t i = 0; i < simplified.size(); ++i) {
            const cv::Point2f& a = simplified[i];
            const cv::Point2f& b = simplified[(i + 1) % simplified.size()];
            area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
            flat.push_back(a.x);
            flat.push_back(a.y);
        }
        scored.emplace_back(std::abs(area) / 2, std::move(flat));
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        // Polygon areas can tie after quantisation.  A lexical coordinate
        // tie-break prevents platform/library-dependent annotation order.
        return a.second < b.second;
    });
    for (auto& s : scored) {
        if (static_cast<int>(rings.size()) >= max_polygons) break;
        rings.push_back(std::move(s.second));
    }
    return rings;
}

std::vector<annot::Instance> annotation_instances(
    const std::vector<Detection>& dets, bool include_polygons,
    const std::vector<float>& proto, int pc, int ph, int pw,
    const LetterboxInfo& lb, int imgsz) {
    std::vector<annot::Instance> out;
    out.reserve(dets.size());
    for (const Detection& d : dets) {
        annot::Instance inst;
        inst.cls = d.class_id;
        inst.score = d.conf;
        inst.x = d.box.x; inst.y = d.box.y; inst.w = d.box.width; inst.h = d.box.height;
        if (include_polygons) {
            if (!d.mask_coeffs.empty())
                inst.polygons = seg_polygons(d, proto, pc, ph, pw, lb, imgsz);
            if (inst.polygons.empty()) {
                // coeff-less (tile det) or empty mask -> box as a valid polygon
                inst.polygons.push_back({d.box.x, d.box.y,
                                         d.box.x + d.box.width, d.box.y,
                                         d.box.x + d.box.width, d.box.y + d.box.height,
                                         d.box.x, d.box.y + d.box.height});
            }
        }
        out.push_back(std::move(inst));
    }
    return out;
}

static bool write_text(const std::string& path, const std::string& body, std::string& err) {
    // u8path preserves UTF-8 destination names on Windows (where the narrow
    // ofstream constructor otherwise follows the process ANSI code page).
    std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) { err = "cannot write " + path; return false; }
    f << body;
    if (!f) { err = "write failed: " + path; return false; }
    return true;
}

std::string unique_stem(std::set<std::string>& used, const std::string& stem) {
    std::string s = stem;
    for (int i = 2; !used.insert(s).second; ++i) s = stem + "-" + std::to_string(i);
    return s;
}

AnnotationSink::AnnotationSink(annot::Format fmt, std::string labels_dir, std::string coco_path,
                               std::vector<std::string> names, bool seg_dialect)
    : fmt_(fmt), labels_dir_(std::move(labels_dir)), coco_path_(std::move(coco_path)),
      names_(std::move(names)), seg_dialect_(seg_dialect) {}

bool AnnotationSink::add(const annot::Image& img, const std::string& coco_file_name, int coco_id) {
    if (!error_.empty()) return false;
    images_ += 1;
    instances_ += static_cast<int>(img.instances.size());
    switch (fmt_) {
    case annot::Format::YoloTXT:
        return write_text(labels_dir_ + "/" + unique_stem(used_names_, img.name) + ".txt",
                          annot::yolo_lines(img, seg_dialect_), error_);
    case annot::Format::PascalVOC: {
        // folder element = the destination dir's basename (Mac parity)
        std::string folder = labels_dir_;
        const size_t cut = folder.find_last_of("/\\");
        if (cut != std::string::npos) folder = folder.substr(cut + 1);
        return write_text(labels_dir_ + "/" + unique_stem(used_names_, img.name) + ".xml",
                          annot::voc_xml(img, names_, folder), error_);
    }
    case annot::Format::CocoJSON:
    default:
        coco_images_.push_back(img);
        coco_names_.push_back(coco_file_name);
        coco_ids_.push_back(coco_id > 0 ? coco_id : images_);
        return true;
    }
}

AnnotationSink::Result AnnotationSink::finish() {
    if (error_.empty()) {
        if (fmt_ == annot::Format::YoloTXT)
            write_text(labels_dir_ + "/classes.txt", annot::classes_txt(names_), error_);
        else if (fmt_ == annot::Format::CocoJSON)
            write_text(coco_path_, annot::coco_json(coco_images_, coco_names_, names_,
                                                    seg_dialect_, &coco_ids_), error_);
    }
    return {images_, instances_, error_};
}

namespace {

struct JpegFileContext {
    FILE* file = nullptr;
    bool ok = true;
};

void jpeg_file_write(void* opaque, void* data, int size) {
    auto* context = static_cast<JpegFileContext*>(opaque);
    if (!context || !context->file || !context->ok || size <= 0) return;
    context->ok = std::fwrite(data, 1, static_cast<size_t>(size), context->file) ==
                  static_cast<size_t>(size);
}

#ifdef _WIN32
static std::wstring utf8_path(const std::string& value) {
    if (value.empty()) return {};
    const int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                           value.data(), static_cast<int>(value.size()),
                                           nullptr, 0);
    if (needed <= 0) return {};
    std::wstring result(static_cast<size_t>(needed), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                            value.data(), static_cast<int>(value.size()),
                            result.data(), needed) != needed)
        return {};
    return result;
}
#endif

} // namespace

bool write_jpg(const std::string& path, const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
#ifdef _WIN32
    const std::wstring wide = utf8_path(path);
    if (wide.empty()) return false;
    FILE* file = _wfopen(wide.c_str(), L"wb");
#else
    FILE* file = std::fopen(path.c_str(), "wb");
#endif
    if (!file) return false;
    JpegFileContext context{file, true};
    const int encoded = stbi_write_jpg_to_func(jpeg_file_write, &context,
                                               rgb.cols, rgb.rows, 3, rgb.data, 90);
    const bool closed = std::fclose(file) == 0;
    return encoded != 0 && context.ok && closed;
}

} // namespace yolomaster
