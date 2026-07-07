#pragma once

#include <vector>

#include "backends/backend.h"
#include "preprocess_types.h"

struct Detection {
    int class_id = -1;
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

std::vector<Detection> postprocess_yolo_output(
    const Tensor& output,
    int num_classes,
    float conf_threshold,
    float iou_threshold,
    const PreprocessResult& prep);
