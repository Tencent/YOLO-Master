#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

struct OutputView {
    int64_t channels = 0;
    int64_t anchors = 0;
    bool channel_first = true;

    size_t index(int64_t channel, int64_t anchor) const {
        if (channel_first) {
            return static_cast<size_t>(channel * anchors + anchor);
        }
        return static_cast<size_t>(anchor * channels + channel);
    }
};

OutputView make_output_view(const Tensor& output, int num_classes) {
    if (output.shape.size() != 3 || output.shape[0] != 1) {
        throw std::invalid_argument("expected YOLO output shape [1, channels, anchors] or [1, anchors, channels]");
    }
    const int64_t dim1 = output.shape[1];
    const int64_t dim2 = output.shape[2];
    if (dim1 <= 0 || dim2 <= 0) {
        throw std::invalid_argument("YOLO output dimensions must be positive");
    }

    OutputView view;
    if (num_classes > 0) {
        const int64_t expected_channels = static_cast<int64_t>(4 + num_classes);
        if (dim1 == expected_channels) {
            view.channels = dim1; view.anchors = dim2; view.channel_first = true;
        } else if (dim2 == expected_channels) {
            view.channels = dim2; view.anchors = dim1; view.channel_first = false;
        } else {
            throw std::invalid_argument("YOLO output does not contain the requested class dimension");
        }
    } else if (dim1 < 5 && dim2 >= 5) {
        view.channels = dim2; view.anchors = dim1; view.channel_first = false;
    } else if (dim2 < 5 && dim1 >= 5) {
        view.channels = dim1; view.anchors = dim2; view.channel_first = true;
    } else if (dim1 <= 256 && dim2 > dim1) {
        // Typical exported YOLO heads are [1, 4+nc, anchors].
        view.channels = dim1; view.anchors = dim2; view.channel_first = true;
    } else if (dim2 <= 256 && dim1 > dim2) {
        // ONNX exporters commonly transpose the same head to [1, anchors, 4+nc].
        view.channels = dim2; view.anchors = dim1; view.channel_first = false;
    } else {
        // Ambiguous small synthetic tensors retain the historical channel-first
        // interpretation; callers with a known class count take the exact path above.
        view.channels = dim1; view.anchors = dim2; view.channel_first = true;
    }
    if (view.channels < 5 || view.anchors <= 0) {
        throw std::invalid_argument("YOLO output must have at least four box channels and one class");
    }
    const size_t expected = static_cast<size_t>(view.channels) * static_cast<size_t>(view.anchors);
    if (output.data.size() != expected) {
        throw std::invalid_argument("YOLO output data size does not match shape");
    }
    return view;
}

}  // namespace

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
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        if (a.class_id != b.class_id) return a.class_id < b.class_id;
        if (a.x1 != b.x1) return a.x1 < b.x1;
        if (a.y1 != b.y1) return a.y1 < b.y1;
        if (a.x2 != b.x2) return a.x2 < b.x2;
        return a.y2 < b.y2;
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
    const PreprocessResult& prep,
    bool multi_label,
    int max_det) {
    if (!std::isfinite(conf_threshold) || conf_threshold < 0.0f || conf_threshold > 1.0f ||
        !std::isfinite(iou_threshold) || iou_threshold < 0.0f || iou_threshold > 1.0f) {
        throw std::invalid_argument("confidence and IoU thresholds must be finite values in [0,1]");
    }
    if (max_det <= 0) {
        throw std::invalid_argument("max_det must be positive");
    }
    if (!std::isfinite(prep.ratio) || prep.ratio <= 0.0f ||
        prep.original_w <= 0 || prep.original_h <= 0) {
        throw std::invalid_argument("invalid preprocessing geometry");
    }
    const OutputView view = make_output_view(output, num_classes);
    const int64_t channels = view.channels;
    const int64_t anchors = view.anchors;
    const int inferred_classes = static_cast<int>(channels) - 4;
    const int classes = num_classes > 0 ? num_classes : inferred_classes;
    if (classes <= 0 || channels < 4 + classes) {
        throw std::invalid_argument("invalid class count for YOLO output");
    }
    std::vector<Detection> detections;
    for (int64_t anchor = 0; anchor < anchors; ++anchor) {
        float best_score = 0.0f;
        int best_class = -1;
        for (int cls = 0; cls < classes; ++cls) {
            const float score = output.data[view.index(4 + cls, anchor)];
            if (!std::isfinite(score)) continue;
            if (score > best_score) {
                best_score = score;
                best_class = cls;
            }
        }
        const float cx = output.data[view.index(0, anchor)];
        const float cy = output.data[view.index(1, anchor)];
        const float w = output.data[view.index(2, anchor)];
        const float h = output.data[view.index(3, anchor)];
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) ||
            w <= 0.0f || h <= 0.0f) {
            continue;
        }

        auto append_detection = [&](int cls, float score) {
            if (cls < 0 || !std::isfinite(score) || score < conf_threshold) return;
            Detection det;
            det.class_id = cls;
            det.confidence = score;
            det.x1 = (cx - w * 0.5f - static_cast<float>(prep.pad_w)) / prep.ratio;
            det.y1 = (cy - h * 0.5f - static_cast<float>(prep.pad_h)) / prep.ratio;
            det.x2 = (cx + w * 0.5f - static_cast<float>(prep.pad_w)) / prep.ratio;
            det.y2 = (cy + h * 0.5f - static_cast<float>(prep.pad_h)) / prep.ratio;
            if (!std::isfinite(det.x1) || !std::isfinite(det.y1) ||
                !std::isfinite(det.x2) || !std::isfinite(det.y2)) return;
            det.x1 = clamp(det.x1, 0.0f, static_cast<float>(prep.original_w));
            det.x2 = clamp(det.x2, 0.0f, static_cast<float>(prep.original_w));
            det.y1 = clamp(det.y1, 0.0f, static_cast<float>(prep.original_h));
            det.y2 = clamp(det.y2, 0.0f, static_cast<float>(prep.original_h));
            if (det.x2 > det.x1 && det.y2 > det.y1) detections.push_back(det);
        };

        if (multi_label) {
            for (int cls = 0; cls < classes; ++cls) {
                const float score = output.data[view.index(4 + cls, anchor)];
                append_detection(cls, score);
            }
        } else if (best_score >= conf_threshold) {
            append_detection(best_class, best_score);
        }
    }
    auto kept = nms(std::move(detections), iou_threshold);
    if (static_cast<int>(kept.size()) > max_det) {
        kept.resize(static_cast<size_t>(max_det));
    }
    return kept;
}
