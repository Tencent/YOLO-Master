#include "ncnn_backend.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {
#ifdef WITH_NCNN
size_t tensor_size(const std::vector<int64_t>& shape) {
    return std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<size_t>(1),
        [](size_t acc, int64_t value) { return acc * static_cast<size_t>(value > 0 ? value : 1); });
}

void validate_input(const Tensor& input) {
    if (input.shape.size() != 4 || input.shape[0] != 1 || input.shape[1] != 3) {
        throw std::invalid_argument("NCNN input tensor must have shape [1, 3, H, W]");
    }
    if (tensor_size(input.shape) != input.data.size()) {
        throw std::invalid_argument("NCNN input tensor data size does not match shape");
    }
}

std::vector<fs::path> sorted_files_with_extension(const fs::path& dir, const std::string& extension) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::pair<std::string, std::string> resolve_ncnn_model_files(const std::string& model_path) {
    const fs::path path(model_path);
    if (fs::is_regular_file(path) && path.extension() == ".param") {
        const fs::path bin = path.parent_path() / (path.stem().string() + ".bin");
        if (!fs::is_regular_file(bin)) {
            throw std::runtime_error("NCNN .bin file not found next to param: " + bin.string());
        }
        return {path.string(), bin.string()};
    }

    if (fs::is_directory(path)) {
        const auto params = sorted_files_with_extension(path, ".param");
        if (params.empty()) {
            throw std::runtime_error("NCNN model directory contains no .param file: " + path.string());
        }

        for (const auto& param : params) {
            const fs::path bin = param.parent_path() / (param.stem().string() + ".bin");
            if (fs::is_regular_file(bin)) {
                return {param.string(), bin.string()};
            }
        }

        const auto bins = sorted_files_with_extension(path, ".bin");
        if (params.size() == 1 && bins.size() == 1) {
            return {params.front().string(), bins.front().string()};
        }
        throw std::runtime_error("NCNN model directory must contain matching .param/.bin files: " + path.string());
    }

    throw std::runtime_error("NCNN --model must be a .param file or directory containing .param/.bin files: " + model_path);
}

ncnn::Mat tensor_to_ncnn_mat(const Tensor& input) {
    const int h = static_cast<int>(input.shape[2]);
    const int w = static_cast<int>(input.shape[3]);
    ncnn::Mat mat(w, h, 3);
    const size_t plane = static_cast<size_t>(h * w);
    for (int c = 0; c < 3; ++c) {
        float* dst = mat.channel(c);
        const float* src = input.data.data() + static_cast<size_t>(c) * plane;
        std::copy(src, src + plane, dst);
    }
    return mat;
}

Tensor ncnn_2d_mat_to_yolo_tensor(const ncnn::Mat& mat) {
    const int rows = mat.h;
    const int cols = mat.w;
    Tensor output;
    if (rows > cols && cols >= 5) {
        output.shape = {1, static_cast<int64_t>(cols), static_cast<int64_t>(rows)};
        output.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int anchor = 0; anchor < rows; ++anchor) {
            const float* row = mat.row(anchor);
            for (int channel = 0; channel < cols; ++channel) {
                output.data[static_cast<size_t>(channel) * static_cast<size_t>(rows) + static_cast<size_t>(anchor)] =
                    row[channel];
            }
        }
        return output;
    }

    output.shape = {1, static_cast<int64_t>(rows), static_cast<int64_t>(cols)};
    const size_t total = mat.total();
    output.data.assign(static_cast<const float*>(mat), static_cast<const float*>(mat) + total);
    return output;
}

Tensor ncnn_mat_to_tensor(const ncnn::Mat& mat) {
    Tensor output;
    if (mat.dims == 1) {
        output.shape = {1, static_cast<int64_t>(mat.w), 1};
    } else if (mat.dims == 2) {
        return ncnn_2d_mat_to_yolo_tensor(mat);
    } else if (mat.dims == 3) {
        if (mat.c == 1) {
            return ncnn_2d_mat_to_yolo_tensor(mat);
        } else {
            output.shape = {1, static_cast<int64_t>(mat.c), static_cast<int64_t>(mat.h * mat.w)};
        }
    } else if (mat.dims == 4) {
        output.shape = {
            static_cast<int64_t>(mat.c),
            static_cast<int64_t>(mat.d),
            static_cast<int64_t>(mat.h),
            static_cast<int64_t>(mat.w)};
    } else {
        throw std::runtime_error("unsupported NCNN output tensor rank");
    }

    const size_t total = mat.total();
    output.data.resize(total);
    if (mat.dims == 3 && mat.c > 1) {
        size_t offset = 0;
        for (int c = 0; c < mat.c; ++c) {
            const ncnn::Mat channel = mat.channel(c);
            const size_t channel_size = static_cast<size_t>(channel.total());
            std::copy(
                static_cast<const float*>(channel),
                static_cast<const float*>(channel) + channel_size,
                output.data.data() + offset);
            offset += channel_size;
        }
    } else {
        std::copy(static_cast<const float*>(mat), static_cast<const float*>(mat) + total, output.data.begin());
    }
    return output;
}

const std::vector<const char*>& input_name_candidates() {
    static const std::vector<const char*> names = {"images", "in0", "input", "data", "input.1"};
    return names;
}

const std::vector<const char*>& output_name_candidates() {
    static const std::vector<const char*> names = {"output0", "out0", "output", "outputs", "output.1"};
    return names;
}
#endif
}  // namespace

NcnnBackend::NcnnBackend() = default;

NcnnBackend::~NcnnBackend() = default;

void NcnnBackend::load(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::invalid_argument("NCNN model path is empty");
    }
    model_path_ = model_path;

#ifdef WITH_NCNN
    const auto files = resolve_ncnn_model_files(model_path_);
    param_path_ = files.first;
    bin_path_ = files.second;

    net_.reset(new ncnn::Net());
    net_->opt.use_vulkan_compute = false;
    net_->opt.num_threads = 1;

    if (net_->load_param(param_path_.c_str()) != 0) {
        throw std::runtime_error("failed to load NCNN param file: " + param_path_);
    }
    if (net_->load_model(bin_path_.c_str()) != 0) {
        throw std::runtime_error("failed to load NCNN bin file: " + bin_path_);
    }
#else
    // Stub mode keeps the benchmark harness buildable without NCNN.
    // Configure WITH_NCNN=ON for real model execution.
#endif
}

Tensor NcnnBackend::infer(const Tensor& input) {
#ifdef WITH_NCNN
    validate_input(input);
    if (!net_) {
        throw std::runtime_error("NCNN backend used before load()");
    }

    ncnn::Mat input_mat = tensor_to_ncnn_mat(input);
    ncnn::Extractor extractor = net_->create_extractor();

    int input_status = -1;
    for (const char* name : input_name_candidates()) {
        input_status = extractor.input(name, input_mat);
        if (input_status == 0) {
            input_name_ = name;
            break;
        }
    }
    if (input_status != 0) {
        throw std::runtime_error("failed to set NCNN input blob; tried images,in0,input,data,input.1");
    }

    ncnn::Mat output_mat;
    int output_status = -1;
    for (const char* name : output_name_candidates()) {
        output_status = extractor.extract(name, output_mat);
        if (output_status == 0) {
            output_name_ = name;
            break;
        }
    }
    if (output_status != 0) {
        throw std::runtime_error("failed to extract NCNN output blob; tried output0,out0,output,outputs,output.1");
    }

    return ncnn_mat_to_tensor(output_mat);
#else
    Tensor output;
    output.shape = {1, 84, 1};
    output.data.assign(84, input.data.empty() ? 0.0f : input.data.front());
    return output;
#endif
}

std::string NcnnBackend::name() const {
    return "ncnn";
}
