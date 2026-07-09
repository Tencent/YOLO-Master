#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "preprocess_types.h"

PreprocessResult preprocess_image(const std::string& image_path, int target_h, int target_w);
