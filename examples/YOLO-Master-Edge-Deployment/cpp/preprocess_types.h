#pragma once

#include "backends/backend.h"

struct PreprocessResult {
    Tensor input;
    int original_w = 0;
    int original_h = 0;
    float ratio = 1.0f;
    int pad_w = 0;
    int pad_h = 0;
};
