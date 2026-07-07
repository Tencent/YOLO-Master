#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

static cv::Mat letterbox(
    const cv::Mat& image,
    int target_h,
    int target_w,
    float& ratio,
    int& pad_w,
    int& pad_h) {
    const int original_h = image.rows;
    const int original_w = image.cols;
    ratio = std::min(
        static_cast<float>(target_h) / static_cast<float>(original_h),
        static_cast<float>(target_w) / static_cast<float>(original_w));

    const int resized_w = static_cast<int>(std::round(static_cast<float>(original_w) * ratio));
    const int resized_h = static_cast<int>(std::round(static_cast<float>(original_h) * ratio));
    pad_w = target_w - resized_w;
    pad_h = target_h - resized_h;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    const int left = pad_w / 2;
    const int right = pad_w - left;
    const int top = pad_h / 2;
    const int bottom = pad_h - top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    pad_w = left;
    pad_h = top;
    return padded;
}

PreprocessResult preprocess_image(const std::string& image_path, int target_h, int target_w) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("failed to read image: " + image_path);
    }

    PreprocessResult result;
    result.original_w = bgr.cols;
    result.original_h = bgr.rows;

    cv::Mat padded = letterbox(bgr, target_h, target_w, result.ratio, result.pad_w, result.pad_h);

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    result.input.shape = {1, 3, target_h, target_w};
    result.input.data.resize(static_cast<size_t>(3 * target_h * target_w));

    const int plane = target_h * target_w;
    for (int y = 0; y < target_h; ++y) {
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < target_w; ++x) {
            const int idx = y * target_w + x;
            result.input.data[idx] = row[x][0];
            result.input.data[plane + idx] = row[x][1];
            result.input.data[2 * plane + idx] = row[x][2];
        }
    }

    return result;
}
