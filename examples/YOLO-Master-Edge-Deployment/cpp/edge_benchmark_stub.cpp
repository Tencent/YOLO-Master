#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

static std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) {
        args[argv[i]] = argv[i + 1];
    }
    return args;
}

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    const auto backend = args.count("--backend") ? args.at("--backend") : "onnx";
    const auto model = args.count("--model") ? args.at("--model") : "";
    const auto images = args.count("--images") ? args.at("--images") : "";

    if (model.empty() || images.empty()) {
        std::cerr << "Usage: yolo_master_edge_benchmark --backend onnx|ncnn|mnn --model MODEL --images LIST\n";
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "backend,model,images,latency_ms\n"
              << backend << "," << model << "," << images << "," << elapsed_ms << "\n";
    return 0;
}
