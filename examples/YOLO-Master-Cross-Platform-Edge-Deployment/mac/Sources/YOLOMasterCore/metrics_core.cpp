#include "metrics_core.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace yolomaster::metrics {

static const double IOUV[10] = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95};

bool load_yolo_labels(const std::string& path, int img_w, int img_h, std::vector<GtBox>& out) {
    out.clear();
    std::ifstream f(path);
    if (!f) return true;
    std::vector<std::string> lines;
    std::string l;
    while (std::getline(f, l)) {
        if (l.find_first_not_of(" \t\r") != std::string::npos) lines.push_back(l);
    }
    if (lines.empty()) return true;
    if (lines[0].find(',') != std::string::npos) {
        // original VisDrone: x,y,w,h,score,category,trunc,occ (pixel). Matches ultralytics'
        // visdrone2yolo: skip score==0 (ignored regions), class = category-1, keep classes 0..9.
        for (const auto& s : lines) {
            std::vector<std::string> v;
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) v.push_back(tok);
            if (v.size() < 6 || v[4] == "0") continue;
            const int c = std::atoi(v[5].c_str()) - 1;
            if (c < 0 || c > 9) continue;
            GtBox b;
            b.x1 = std::atof(v[0].c_str()); b.y1 = std::atof(v[1].c_str());
            b.x2 = b.x1 + std::atof(v[2].c_str()); b.y2 = b.y1 + std::atof(v[3].c_str());
            b.cls = c;
            out.push_back(b);
        }
        return true;
    }
    for (const auto& s : lines) {
        std::istringstream ss(s);
        double c, cx, cy, w, h;
        if (!(ss >> c >> cx >> cy >> w >> h)) continue;
        GtBox b;
        b.cls = static_cast<int>(c);
        cx *= img_w; cy *= img_h; w *= img_w; h *= img_h;
        b.x1 = cx - w / 2; b.y1 = cy - h / 2; b.x2 = cx + w / 2; b.y2 = cy + h / 2;
        out.push_back(b);
    }
    return true;
}

std::string label_path_for(const std::string& image_path, const std::string& labels_dir) {
    const auto slash = image_path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? "" : image_path.substr(0, slash + 1);
    std::string base = slash == std::string::npos ? image_path : image_path.substr(slash + 1);
    const auto dot = base.find_last_of('.');
    const std::string stem = dot == std::string::npos ? base : base.substr(0, dot);
    if (!labels_dir.empty()) {
        std::string d = labels_dir;
        if (!d.empty() && d.back() != '/' && d.back() != '\\') d += '/';
        return d + stem + ".txt";
    }
    // ultralytics img2label_paths: replace the LAST "/images/" with "/labels/"
    std::string p = dir;
    const auto pos = p.rfind("/images/");
    if (pos != std::string::npos) p = p.substr(0, pos) + "/labels/" + p.substr(pos + 8);
    return p + stem + ".txt";
}

double round6(double v) {            // what "std::cout << float" prints (6 significant digits)
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<float>(v));
    return std::strtod(buf, nullptr);
}

static double box_iou(const PredBox& a, const GtBox& b) {
    const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const double lx = std::max(a.x1, b.x1), ly = std::max(a.y1, b.y1);
    const double rx = std::min(a.x2, b.x2), ry = std::min(a.y2, b.y2);
    const double w = std::max(rx - lx, 0.0), h = std::max(ry - ly, 0.0);
    const double inter = w * h;
    return inter / (area_a + area_b - inter + 1e-9);
}

// eval_map_standalone.match(): per threshold, all (gt, pred) pairs with iou >= thr and equal class,
// sorted by iou descending; keep each pred's first (best) row, then each gt's first row in
// pred-index order; those preds are correct at that threshold.
static void match_image(const ImageEval& im, std::vector<std::array<bool, 10>>& correct) {
    const size_t N = im.preds.size(), M = im.gts.size();
    correct.assign(N, std::array<bool, 10>{});
    if (N == 0 || M == 0) return;
    std::vector<double> iou(M * N, 0.0);   // (M_gt, N_pred), zero where the class differs
    for (size_t g = 0; g < M; ++g)
        for (size_t p = 0; p < N; ++p)
            if (im.gts[g].cls == im.preds[p].cls) iou[g * N + p] = box_iou(im.preds[p], im.gts[g]);
    struct Row { size_t g, p; double v; };
    for (int k = 0; k < 10; ++k) {
        std::vector<Row> rows;
        for (size_t g = 0; g < M; ++g)                 // np.nonzero order: row-major
            for (size_t p = 0; p < N; ++p)
                if (iou[g * N + p] >= IOUV[k]) rows.push_back({g, p, iou[g * N + p]});
        if (rows.empty()) continue;
        std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.v > b.v; });
        // np.unique(m[:,1], return_index=True): first occurrence per pred, result ordered by pred index
        std::map<size_t, Row> by_pred;
        for (const auto& r : rows) by_pred.emplace(r.p, r);
        // np.unique(m[:,0], return_index=True) on that: first occurrence per gt in pred-index order
        std::set<size_t> seen_gt;
        for (const auto& kv : by_pred) {
            if (seen_gt.insert(kv.second.g).second) correct[kv.second.p][k] = true;
        }
    }
}

// np.interp on a non-decreasing xp
static double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    const size_t n = xp.size();
    // numpy: x below xp[0] -> fp[0]; otherwise j = LAST index with xp[j] <= x (so a plateau of equal
    // xp values resolves to its last entry, including at x == xp[0]); at or past the end -> fp[n-1].
    if (x < xp[0]) return fp[0];
    const size_t j = static_cast<size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
    if (j + 1 >= n) return fp[n - 1];
    const double dx = xp[j + 1] - xp[j];
    if (dx == 0.0) return fp[j];
    return fp[j] + (fp[j + 1] - fp[j]) * (x - xp[j]) / dx;
}

static double compute_ap(const std::vector<double>& recall, const std::vector<double>& precision) {
    std::vector<double> mrec, mpre;
    mrec.reserve(recall.size() + 2); mpre.reserve(precision.size() + 2);
    mrec.push_back(0.0); mrec.insert(mrec.end(), recall.begin(), recall.end()); mrec.push_back(1.0);
    mpre.push_back(1.0); mpre.insert(mpre.end(), precision.begin(), precision.end()); mpre.push_back(0.0);
    for (size_t i = mpre.size() - 1; i-- > 0;) mpre[i] = std::max(mpre[i], mpre[i + 1]);   // precision envelope
    double area = 0.0, prev_x = 0.0, prev_y = interp(0.0, mrec, mpre);
    for (int i = 1; i <= 100; ++i) {
        const double x = i / 100.0;
        const double y = interp(x, mrec, mpre);
        area += (x - prev_x) * (y + prev_y) / 2.0;
        prev_x = x; prev_y = y;
    }
    return area;
}

std::vector<std::array<bool, 10>> debug_tp(const ImageEval& im) {
    std::vector<std::array<bool, 10>> c; match_image(im, c); return c;
}

MapResult evaluate(const std::vector<ImageEval>& images) {
    MapResult res;
    res.images = static_cast<int>(images.size());
    struct Det { double conf; int cls; std::array<bool, 10> tp; size_t order; };
    std::vector<Det> dets;
    std::map<int, int> n_gt;
    std::vector<std::array<bool, 10>> correct;
    size_t order = 0;
    for (const auto& im : images) {
        for (const auto& g : im.gts) n_gt[g.cls]++;
        if (im.preds.empty()) continue;
        match_image(im, correct);
        for (size_t i = 0; i < im.preds.size(); ++i)
            dets.push_back({im.preds[i].conf, im.preds[i].cls, correct[i], order++});
    }
    // np.argsort(-conf): descending conf; ties broken by dataset order (stable), see plan note
    std::stable_sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.conf > b.conf; });
    std::map<int, int> n_pred;
    for (const auto& d : dets) n_pred[d.cls]++;
    double sum50 = 0, sum_all = 0; int n_cls = 0;
    for (const auto& kv : n_gt) {                        // classes = sorted unique target classes
        ClassAP c; c.cls = kv.first; c.n_gt = kv.second; c.n_pred = n_pred.count(kv.first) ? n_pred[kv.first] : 0;
        ++n_cls;
        if (c.n_pred > 0 && c.n_gt > 0) {
            for (int k = 0; k < 10; ++k) {
                std::vector<double> recall, precision;
                recall.reserve(c.n_pred); precision.reserve(c.n_pred);
                double tpc = 0, fpc = 0;
                for (const auto& d : dets) {
                    if (d.cls != kv.first) continue;
                    if (d.tp[k]) tpc += 1; else fpc += 1;
                    recall.push_back(tpc / (c.n_gt + 1e-16));
                    precision.push_back(tpc / (tpc + fpc));
                }
                c.ap[k] = compute_ap(recall, precision);
            }
        }
        sum50 += c.ap50(); sum_all += c.ap5095();
        res.per_class.push_back(c);
    }
    if (n_cls > 0) { res.map50 = sum50 / n_cls; res.map5095 = sum_all / n_cls; }
    return res;
}

} // namespace yolomaster::metrics
