#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
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
    std::string json_output;
    int imgsz = 0;
    float conf = -1.0f;
    float iou = -1.0f;
    int warmup = 5;
    int runs = 3;
    int limit = 0;
    int min_images = 0;
    int max_det = 300;
    int threads = 4;
    bool multi_label = false;
    bool multi_label_set = false;
    bool imgsz_set = false;
    bool conf_set = false;
    bool iou_set = false;
};

struct TimingRow {
    int run = 0;
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
        << "[--imgsz 640] "
        << "[--conf 0.001] "
        << "[--iou 0.70] "
        << "[--max-det 300] "
        << "[--multi-label|--single-label] "
        << "[--warmup 5] "
        << "[--runs 3] "
        << "[--limit 500] "
        << "[--min-images 500] "
        << "[--threads 4] "
        << "[--output benchmark.csv] "
        << "[--json benchmark.json]\n";
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
        if (key == "--multi-label") {
            args.multi_label = true;
            args.multi_label_set = true;
            continue;
        }
        if (key == "--single-label") {
            args.multi_label = false;
            args.multi_label_set = true;
            continue;
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
        } else if (key == "--json") {
            args.json_output = value;
        } else if (key == "--imgsz") {
            args.imgsz = std::stoi(value);
            args.imgsz_set = true;
        } else if (key == "--conf") {
            args.conf = std::stof(value);
            args.conf_set = true;
        } else if (key == "--iou") {
            args.iou = std::stof(value);
            args.iou_set = true;
        } else if (key == "--warmup") {
            args.warmup = std::stoi(value);
        } else if (key == "--runs") {
            args.runs = std::stoi(value);
        } else if (key == "--limit") {
            args.limit = std::stoi(value);
        } else if (key == "--min-images") {
            args.min_images = std::stoi(value);
        } else if (key == "--max-det") {
            args.max_det = std::stoi(value);
        } else if (key == "--threads") {
            args.threads = std::stoi(value);
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
    std::transform(args.profile.begin(), args.profile.end(), args.profile.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (args.profile != "visdrone" && args.profile != "sku110k") {
        std::cerr << "Invalid --profile: " << args.profile << "\n";
        std::exit(2);
    }
    if (args.imgsz < 0 || args.warmup < 0 || args.runs <= 0 || args.limit < 0 ||
        args.min_images < 0 || args.max_det <= 0 || args.threads <= 0) {
        std::cerr << "Invalid numeric argument\n";
        std::exit(2);
    }

    if (!args.imgsz_set) {
        args.imgsz = args.profile == "visdrone" ? 640 : 1280;
    }
    if (args.imgsz <= 0) {
        std::cerr << "imgsz must be positive\n";
        std::exit(2);
    }

    if (!args.conf_set) {
        args.conf = args.profile == "visdrone" ? 0.001f : 0.25f;
    }
    if (!args.iou_set) {
        args.iou = args.profile == "visdrone" ? 0.70f : 0.60f;
    }
    if (!std::isfinite(args.conf) || args.conf < 0.0f || args.conf > 1.0f ||
        !std::isfinite(args.iou) || args.iou < 0.0f || args.iou > 1.0f) {
        std::cerr << "conf and iou must be finite values in [0,1]\n";
        std::exit(2);
    }
    if (!args.multi_label_set &&
        (args.profile == "visdrone" || args.profile == "sku110k")) {
        args.multi_label = true;
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

static std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::string image_stem_key(const fs::path& path) {
    std::string key = path.stem().string();
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return key;
}

static void validate_unique_stems(const std::vector<std::string>& images) {
    std::set<std::string> stems;
    for (const auto& image : images) {
        const std::string key = image_stem_key(fs::path(image));
        if (key.empty() || !stems.insert(key).second) {
            throw std::runtime_error(
                "image stems are not unique; duplicate output stem: " + key);
        }
    }
}

static std::vector<std::string> read_image_list_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open image list: " + path);
    }

    const fs::path list_path = fs::absolute(fs::path(path));
    const fs::path base = list_path.parent_path();
    std::vector<std::string> images;
    std::string line;
    size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim_copy(line);
        if (line_number == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xef &&
            static_cast<unsigned char>(line[1]) == 0xbb &&
            static_cast<unsigned char>(line[2]) == 0xbf) {
            line.erase(0, 3);
            line = trim_copy(line);
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (line.size() >= 2 && line.front() == line.back() &&
            (line.front() == '"' || line.front() == '\'')) {
            line = trim_copy(line.substr(1, line.size() - 2));
        }
        fs::path image_path(line);
        if (image_path.is_relative()) {
            image_path = base / image_path;
        }
        images.push_back(image_path.lexically_normal().string());
    }
    if (images.empty()) {
        throw std::runtime_error("image list is empty: " + path);
    }
    for (const auto& image : images) {
        if (!fs::is_regular_file(fs::path(image))) {
            throw std::runtime_error("image path does not exist: " + image);
        }
        if (!is_image_file(fs::path(image))) {
            throw std::runtime_error("unsupported image extension: " + image);
        }
    }
    validate_unique_stems(images);
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
    validate_unique_stems(images);
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
    if (limit > 0) {
        validate_unique_stems(images);
    }
    return images;
}

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::string csv_quote(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char c : value) {
        if (c == '\"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += '"';
    return escaped;
}

static void write_csv(const std::string& path, const std::vector<TimingRow>& rows) {
    const fs::path output(path);
    if (!output.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(output.parent_path(), ec);
        if (ec) throw std::runtime_error("failed to create benchmark output directory: " + ec.message());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write benchmark CSV: " + path);
    }
    out << "image,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections,run\n";
    for (const auto& row : rows) {
        out << csv_quote(row.image) << ","
            << row.preprocess_ms << ","
            << row.inference_ms << ","
            << row.postprocess_ms << ","
            << row.total_ms << ","
            << row.detections << ","
            << row.run << "\n";
    }
}

static double percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    if (!std::isfinite(pct) || pct < 0.0 || pct > 100.0) {
        throw std::invalid_argument("percentile must be in [0,100]");
    }
    // Nearest-rank semantics match edge_utils.summarize_latency_ms and make
    // small smoke runs deterministic (P95 of five values is the maximum).
    const size_t rank = std::max<size_t>(1, static_cast<size_t>(std::ceil(pct * values.size() / 100.0)));
    const size_t idx = std::min(values.size() - 1, rank - 1);
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

static std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char c : value) {
        switch (c) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\b': escaped << "\\b"; break;
        case '\f': escaped << "\\f"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (c < 0x20) escaped << "\\u" << std::hex << std::setw(4)
                                   << std::setfill('0') << static_cast<int>(c) << std::dec;
            else escaped << static_cast<char>(c);
        }
    }
    return escaped.str();
}

static std::string host_cpu_model() {
#if defined(__linux__)
    std::ifstream cpu("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) return trim_copy(line.substr(colon + 1));
        }
    }
#endif
    return {};
}

static std::string host_platform() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

static std::string host_compiler() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang_version__)
    return __clang_version__;
#elif defined(__VERSION__)
    return __VERSION__;
#else
    return "unknown";
#endif
}

static void write_json(
    const std::string& path,
    const Args& args,
    const std::vector<std::string>& images,
    const std::vector<TimingRow>& rows) {
    const fs::path output(path);
    if (!output.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(output.parent_path(), ec);
        if (ec) throw std::runtime_error("failed to create benchmark JSON directory: " + ec.message());
    }
    std::vector<double> totals, prep, infer, post;
    totals.reserve(rows.size()); prep.reserve(rows.size()); infer.reserve(rows.size()); post.reserve(rows.size());
    for (const auto& row : rows) {
        totals.push_back(row.total_ms); prep.push_back(row.preprocess_ms);
        infer.push_back(row.inference_ms); post.push_back(row.postprocess_ms);
    }
    auto summary_object = [](const std::vector<double>& values) {
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        const double avg = values.empty() ? 0.0 : sum / static_cast<double>(values.size());
        std::ostringstream s;
        s << std::setprecision(10)
          << "{\"count\":" << values.size()
          << ",\"mean_ms\":" << avg
          << ",\"p50_ms\":" << percentile(values, 50.0)
          << ",\"p95_ms\":" << percentile(values, 95.0)
          << ",\"p99_ms\":" << percentile(values, 99.0)
          << ",\"fps\":" << (avg > 0.0 ? 1000.0 / avg : 0.0) << "}";
        return s.str();
    };
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to write benchmark JSON: " + path);
    out << std::setprecision(10)
        << "{\n  \"schema_version\": 1,\n"
        << "  \"backend\": \"" << json_escape(args.backend) << "\",\n"
        << "  \"model\": \"" << json_escape(args.model) << "\",\n"
        << "  \"images_source\": \"" << json_escape(args.images) << "\",\n"
        << "  \"image_count\": " << images.size() << ",\n"
        << "  \"profile\": \"" << json_escape(args.profile) << "\",\n"
        << "  \"protocol\": {\"imgsz\": " << args.imgsz
        << ", \"conf\": " << args.conf << ", \"iou\": " << args.iou
        << ", \"max_det\": " << args.max_det
        << ", \"multi_label\": " << (args.multi_label ? "true" : "false")
        << ", \"letterbox\": true},\n"
        << "  \"benchmark\": {\"warmup\": " << args.warmup
        << ", \"runs\": " << args.runs << ", \"threads\": " << args.threads
        << ", \"rows\": " << rows.size() << "},\n"
        << "  \"environment\": {\"platform\": \"" << host_platform()
        << "\", \"cpu_model\": \"" << json_escape(host_cpu_model())
        << "\", \"compiler\": \"" << json_escape(host_compiler())
        << "\", \"build_date\": \"" << __DATE__ << "\"},\n"
        << "  \"timing_ms\": {\"preprocess\": " << summary_object(prep)
        << ", \"inference\": " << summary_object(infer)
        << ", \"postprocess\": " << summary_object(post)
        << ", \"total\": " << summary_object(totals) << "}\n}\n";
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto images = collect_images(args.images, args.limit);
        if (args.min_images > 0 && static_cast<int>(images.size()) < args.min_images) {
            throw std::runtime_error(
                "resolved image count " + std::to_string(images.size()) +
                " is below --min-images " + std::to_string(args.min_images));
        }
        auto backend = create_backend(args.backend);
        backend->set_num_threads(args.threads);
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
                const auto detections = postprocess_yolo_output(
                    output, 0, args.conf, args.iou, prep, args.multi_label, args.max_det);
                const auto postprocess_end = std::chrono::steady_clock::now();

                const auto total_end = std::chrono::steady_clock::now();

                TimingRow row;
                row.run = run;
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
                  << " threads=" << args.threads
                  << " conf=" << args.conf
                  << " iou=" << args.iou
                  << " max_det=" << args.max_det
                  << " multi_label=" << (args.multi_label ? "true" : "false")
                  << " warmup=" << args.warmup
                  << " runs=" << args.runs
                  << " output=" << args.output << "\n";
        print_summary(rows);
        if (!args.json_output.empty()) {
            write_json(args.json_output, args, images, rows);
            std::cout << "json=" << args.json_output << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
