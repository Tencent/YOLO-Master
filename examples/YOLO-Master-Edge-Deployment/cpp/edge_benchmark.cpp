#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/backend_factory.h"
#include "postprocess.h"
#include "preprocess.h"

namespace fs = std::filesystem;

struct Args {
    std::string backend = "onnx";
    std::string model;
    std::string images;
    std::string profile = "visdrone";
    std::string output = "benchmark.csv";
    int imgsz = 960;
    float conf = -1.0f;
    float iou = -1.0f;
    int warmup = 5;
    int runs = 1;
    int limit = 0;
};

struct TimingRow {
    std::string image;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    int detections = 0;
};

static void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " "
        << "--backend onnx|ncnn|mnn "
        << "--model MODEL "
        << "--images IMAGE_LIST "
        << "[--profile visdrone|sku110k] "
        << "[--imgsz 960] "
        << "[--conf 0.20] "
        << "[--iou 0.55] "
        << "[--warmup 5] "
        << "[--runs 1] "
        << "[--limit 500] "
        << "[--output benchmark.csv]\n";
}

static bool require_value(int i, int argc, const char* key) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value for " << key << "\n";
        return false;
    }
    return true;
}

static Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (!require_value(i, argc, argv[i])) {
            print_usage(argv[0]);
            std::exit(2);
        }

        const std::string value = argv[++i];
        if (key == "--backend") {
            args.backend = value;
        } else if (key == "--model") {
            args.model = value;
        } else if (key == "--images") {
            args.images = value;
        } else if (key == "--profile") {
            args.profile = value;
        } else if (key == "--output") {
            args.output = value;
        } else if (key == "--imgsz") {
            args.imgsz = std::stoi(value);
        } else if (key == "--conf") {
            args.conf = std::stof(value);
        } else if (key == "--iou") {
            args.iou = std::stof(value);
        } else if (key == "--warmup") {
            args.warmup = std::stoi(value);
        } else if (key == "--runs") {
            args.runs = std::stoi(value);
        } else if (key == "--limit") {
            args.limit = std::stoi(value);
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }

    if (args.model.empty() || args.images.empty()) {
        std::cerr << "Missing required --model or --images argument\n";
        print_usage(argv[0]);
        std::exit(2);
    }
    if (args.backend != "onnx" && args.backend != "ncnn" && args.backend != "mnn") {
        std::cerr << "Invalid --backend: " << args.backend << "\n";
        std::exit(2);
    }
    if (args.profile != "visdrone" && args.profile != "sku110k") {
        std::cerr << "Invalid --profile: " << args.profile << "\n";
        std::exit(2);
    }
    if (args.imgsz <= 0 || args.warmup < 0 || args.runs <= 0 || args.limit < 0) {
        std::cerr << "Invalid numeric argument\n";
        std::exit(2);
    }

    if (args.conf < 0.0f) {
        args.conf = args.profile == "visdrone" ? 0.20f : 0.25f;
    }
    if (args.iou < 0.0f) {
        args.iou = args.profile == "visdrone" ? 0.55f : 0.60f;
    }
    return args;
}

static std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool is_image_file(const fs::path& path) {
    const std::string ext = lowercase(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static std::vector<std::string> read_image_list_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open image list: " + path);
    }

    std::vector<std::string> images;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            images.push_back(line);
        }
    }
    if (images.empty()) {
        throw std::runtime_error("image list is empty: " + path);
    }
    return images;
}

static std::vector<std::string> read_image_directory(const fs::path& path) {
    std::vector<std::string> images;
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            images.push_back(entry.path().string());
        }
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) {
        throw std::runtime_error("no image files found in directory: " + path.string());
    }
    return images;
}

static std::vector<std::string> collect_images(const std::string& path, int limit) {
    std::vector<std::string> images;
    const fs::path input(path);
    if (fs::is_directory(input)) {
        images = read_image_directory(input);
    } else {
        images = read_image_list_file(path);
    }
    if (limit > 0 && static_cast<size_t>(limit) < images.size()) {
        images.resize(static_cast<size_t>(limit));
    }
    return images;
}

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void write_csv(const std::string& path, const std::vector<TimingRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write benchmark CSV: " + path);
    }
    out << "image,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections\n";
    for (const auto& row : rows) {
        out << row.image << ","
            << row.preprocess_ms << ","
            << row.inference_ms << ","
            << row.postprocess_ms << ","
            << row.total_ms << ","
            << row.detections << "\n";
    }
}

static double percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t idx = std::min(
        values.size() - 1,
        static_cast<size_t>((pct / 100.0) * static_cast<double>(values.size() - 1)));
    return values[idx];
}

static void print_summary(const std::vector<TimingRow>& rows) {
    std::vector<double> totals;
    totals.reserve(rows.size());
    for (const auto& row : rows) {
        totals.push_back(row.total_ms);
    }

    const double sum = std::accumulate(totals.begin(), totals.end(), 0.0);
    const double mean = totals.empty() ? 0.0 : sum / static_cast<double>(totals.size());
    const double fps = mean > 0.0 ? 1000.0 / mean : 0.0;

    std::cout << "count,mean_ms,p50_ms,p95_ms,p99_ms,fps\n"
              << totals.size() << ","
              << mean << ","
              << percentile(totals, 50.0) << ","
              << percentile(totals, 95.0) << ","
              << percentile(totals, 99.0) << ","
              << fps << "\n";
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto images = collect_images(args.images, args.limit);
        auto backend = create_backend(args.backend);
        backend->load(args.model);

        const Tensor warmup_input = preprocess_image(images.front(), args.imgsz, args.imgsz).input;
        for (int i = 0; i < args.warmup; ++i) {
            backend->infer(warmup_input);
        }

        std::vector<TimingRow> rows;
        rows.reserve(images.size() * static_cast<size_t>(args.runs));

        for (int run = 0; run < args.runs; ++run) {
            for (const auto& image : images) {
                const auto total_start = std::chrono::steady_clock::now();

                const auto preprocess_start = std::chrono::steady_clock::now();
                PreprocessResult prep = preprocess_image(image, args.imgsz, args.imgsz);
                const auto preprocess_end = std::chrono::steady_clock::now();

                const auto inference_start = std::chrono::steady_clock::now();
                Tensor output = backend->infer(prep.input);
                const auto inference_end = std::chrono::steady_clock::now();

                const auto postprocess_start = std::chrono::steady_clock::now();
                const auto detections = postprocess_yolo_output(output, 0, args.conf, args.iou, prep);
                const auto postprocess_end = std::chrono::steady_clock::now();

                const auto total_end = std::chrono::steady_clock::now();

                TimingRow row;
                row.image = image;
                row.preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);
                row.inference_ms = elapsed_ms(inference_start, inference_end);
                row.postprocess_ms = elapsed_ms(postprocess_start, postprocess_end);
                row.total_ms = elapsed_ms(total_start, total_end);
                row.detections = static_cast<int>(detections.size());
                rows.push_back(row);
            }
        }

        write_csv(args.output, rows);
        std::cout << "backend=" << backend->name()
                  << " model=" << args.model
                  << " profile=" << args.profile
                  << " imgsz=" << args.imgsz
                  << " conf=" << args.conf
                  << " iou=" << args.iou
                  << " output=" << args.output << "\n";
        print_summary(rows);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
