// Domain-tuned post-processing (C++) — mirrors python/postprocess.py.
// Area-adaptive confidence gate (small VisDrone objects get a lower threshold)
// followed by per-class greedy NMS, matching the Python reference exactly.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace yolo {

struct Detection {
  float x1, y1, x2, y2;  // xyxy in original image pixels (after unscale)
  float score;
  int cls;
};

struct NmsConfig {
  float conf = 0.15f;            // VisDrone-tuned: lower than COCO 0.25
  float small_conf = 0.05f;      // even lower for tiny boxes
  float small_area = 32.f * 32.f;
  float iou = 0.45f;
  int max_det = 300;
  int num_classes = 10;
};

struct Box { float x1, y1, x2, y2; };

inline float iou(const Box& a, const Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  return uni > 0 ? inter / uni : 0.0f;
}

struct Cand {
  Box box;
  float score;
  int cls;
};

// raw points at a (4+nc, N) or (1, 4+nc, N) tensor in row-major layout.
// dims: {d0, d1, d2}. We treat the last axis as N anchors.
inline std::vector<Detection> decode_and_nms(const float* raw,
                                             const std::vector<int64_t>& dims,
                                             const NmsConfig& cfg) {
  // Resolve (channels, N).
  int64_t channels, N;
  if (dims.size() == 3) {
    channels = dims[1];
    N = dims[2];
  } else if (dims.size() == 2) {
    channels = dims[0];
    N = dims[1];
  } else {
    return {};
  }
  const int nc = cfg.num_classes;
  if (channels < 4 + nc) return {};

  std::vector<std::vector<Cand>> per_cls(nc);
  for (int64_t i = 0; i < N; ++i) {
    const float cx = raw[0 * N + i];
    const float cy = raw[1 * N + i];
    const float w = raw[2 * N + i];
    const float h = raw[3 * N + i];
    Box b{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
    const float area = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float thr = (area < cfg.small_area) ? cfg.small_conf : cfg.conf;
    int best_c = 0;
    float best_s = -1.0f;
    for (int c = 0; c < nc; ++c) {
      const float s = raw[(4 + c) * N + i];
      if (s > best_s) {
        best_s = s;
        best_c = c;
      }
    }
    if (best_s >= thr) {
      per_cls[best_c].push_back({b, best_s, best_c});
    }
  }

  std::vector<Detection> out;
  for (int c = 0; c < nc; ++c) {
    auto& v = per_cls[c];
    std::sort(v.begin(), v.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
    std::vector<bool> suppressed(v.size(), false);
    for (size_t i = 0; i < v.size() && int(out.size()) < cfg.max_det; ++i) {
      if (suppressed[i]) continue;
      out.push_back({v[i].box.x1, v[i].box.y1, v[i].box.x2, v[i].box.y2, v[i].score, c});
      for (size_t j = i + 1; j < v.size(); ++j) {
        if (!suppressed[j] && iou(v[i].box, v[j].box) >= cfg.iou) suppressed[j] = true;
      }
    }
  }
  return out;
}

}  // namespace yolo
