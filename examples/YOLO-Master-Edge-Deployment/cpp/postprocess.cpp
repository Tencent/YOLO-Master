#include "postprocess.h"

#include <algorithm>
#include <stdexcept>

static float clamp(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

static float box_iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;

    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

static std::vector<Detection> nms(std::vector<Detection> detections, float iou_threshold) {
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!suppressed[j] && detections[i].class_id == detections[j].class_id &&
                box_iou(detections[i], detections[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

std::vector<Detection> postprocess_yolo_output(
    const Tensor& output,
    int num_classes,
    float conf_threshold,
    float iou_threshold,
    const PreprocessResult& prep) {
    if (output.shape.size() != 3) {
        throw std::invalid_argument("expected YOLO output shape [1, channels, anchors]");
    }
    const int64_t batch = output.shape[0];
    const int64_t channels = output.shape[1];
    const int64_t anchors = output.shape[2];
    if (batch != 1 || channels < 5 || anchors <= 0) {
        throw std::invalid_argument("invalid YOLO output shape");
    }

    const int inferred_classes = static_cast<int>(channels) - 4;
    const int classes = num_classes > 0 ? num_classes : inferred_classes;
    if (classes <= 0 || channels < 4 + classes) {
        throw std::invalid_argument("invalid class count for YOLO output");
    }
    if (output.data.size() < static_cast<size_t>(channels * anchors)) {
        throw std::invalid_argument("YOLO output data is smaller than shape");
    }

    std::vector<Detection> detections;
    for (int64_t anchor = 0; anchor < anchors; ++anchor) {
        float best_score = 0.0f;
        int best_class = -1;
        for (int cls = 0; cls < classes; ++cls) {
            const float score = output.data[static_cast<size_t>((4 + cls) * anchors + anchor)];
            if (score > best_score) {
                best_score = score;
                best_class = cls;
            }
        }
        if (best_score < conf_threshold) {
            continue;
        }

        const float cx = output.data[static_cast<size_t>(0 * anchors + anchor)];
        const float cy = output.data[static_cast<size_t>(1 * anchors + anchor)];
        const float w = output.data[static_cast<size_t>(2 * anchors + anchor)];
        const float h = output.data[static_cast<size_t>(3 * anchors + anchor)];

        Detection det;
        det.class_id = best_class;
        det.confidence = best_score;
        det.x1 = (cx - w * 0.5f - static_cast<float>(prep.pad_w)) / prep.ratio;
        det.y1 = (cy - h * 0.5f - static_cast<float>(prep.pad_h)) / prep.ratio;
        det.x2 = (cx + w * 0.5f - static_cast<float>(prep.pad_w)) / prep.ratio;
        det.y2 = (cy + h * 0.5f - static_cast<float>(prep.pad_h)) / prep.ratio;
        det.x1 = clamp(det.x1, 0.0f, static_cast<float>(prep.original_w));
        det.x2 = clamp(det.x2, 0.0f, static_cast<float>(prep.original_w));
        det.y1 = clamp(det.y1, 0.0f, static_cast<float>(prep.original_h));
        det.y2 = clamp(det.y2, 0.0f, static_cast<float>(prep.original_h));
        detections.push_back(det);
    }

    return nms(std::move(detections), iou_threshold);
}

