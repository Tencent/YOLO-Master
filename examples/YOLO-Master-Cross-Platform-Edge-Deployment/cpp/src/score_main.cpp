// yolomaster_score: score a directory of "class conf x1 y1 x2 y2" txt dumps against YOLO labels with
// the in-process mAP (metrics_core.hpp, the portable core). Prints the same line as scripts/eval_map.py so the two can
// be diffed: "images=N  mAP50=X.XXXX  mAP50-95=Y.YYYY".
#include "metrics_core.hpp"
#include "stb_image.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace yolomaster;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: yolomaster_score <preds_dir> <images_dir> <labels_dir> [--per-class]\n");
        return 2;
    }
    const std::string preds = argv[1], images = argv[2], labels = argv[3];
    const bool per_class = argc > 4 && std::string(argv[4]) == "--per-class";
    const bool dump_tp = argc > 4 && std::string(argv[4]) == "--dump-tp";
    std::vector<std::string> imgs;
    for (const auto& e : fs::directory_iterator(images))
        if (e.is_regular_file() && e.path().extension() == ".jpg") imgs.push_back(e.path().string());
    std::sort(imgs.begin(), imgs.end());
    std::vector<metrics::ImageEval> evals;
    evals.reserve(imgs.size());
    for (const auto& p : imgs) {
        int w = 0, h = 0, c = 0;
        if (!stbi_info(p.c_str(), &w, &h, &c)) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); return 1; }
        metrics::ImageEval ev;
        metrics::load_yolo_labels(metrics::label_path_for(p, labels), w, h, ev.gts);
        std::ifstream f(preds + "/" + fs::path(p).stem().string() + ".txt");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            metrics::PredBox b;
            if (ss >> b.cls >> b.conf >> b.x1 >> b.y1 >> b.x2 >> b.y2) ev.preds.push_back(b);
        }
        evals.push_back(std::move(ev));
    }
    if (dump_tp) {
        for (size_t i = 0; i < evals.size(); ++i) {
            std::vector<metrics::ImageEval> one{evals[i]};
            const auto tp = metrics::debug_tp(one[0]);
            int t50 = 0, tall = 0;
            for (const auto& row : tp) { t50 += row[0]; for (bool b : row) tall += b; }
            std::printf("%s preds=%zu gts=%zu tp50=%d tpall=%d\n", fs::path(imgs[i]).stem().string().c_str(),
                        evals[i].preds.size(), evals[i].gts.size(), t50, tall);
            if (std::getenv("YM_DUMP_ROWS"))
                for (size_t k = 0; k < tp.size(); ++k) {
                    std::string bits; for (bool b : tp[k]) bits += b ? '1' : '0';
                    std::printf("  %zu %s\n", k, bits.c_str());
                }
        }
        return 0;
    }
    const metrics::MapResult r = metrics::evaluate(evals);
    std::printf("images=%d  mAP50=%.4f  mAP50-95=%.4f\n", r.images, r.map50, r.map5095);
    if (per_class)
        for (const auto& c : r.per_class)
            { std::printf("  class %d  n_gt=%d n_pred=%d ", c.cls, c.n_gt, c.n_pred);
              for (double v : c.ap) std::printf(" %.5f", v); std::printf("\n"); }
    return 0;
}
