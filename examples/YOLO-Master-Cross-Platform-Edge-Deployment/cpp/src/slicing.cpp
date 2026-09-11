// Sliced inference orchestrator - see slicing.hpp for the semantics and slicing_core.hpp
// for the grid math. Parity reference: mac/Sources/YOLOMasterKit/Tiling.swift
// (tiledCandidates) and upstream predictor.py:542-699 (_run_sparse_sahi_single).
#include "slicing.hpp"

namespace yolomaster {

SliceOutput sliced_candidates(Backend& be, const cv::Mat& bgr, const Config& cfg,
                              const SliceConfig& sc, float conf_floor,
                              const std::atomic<bool>* cancel) {
    SliceOutput out;
    const int w = bgr.cols, h = bgr.rows;

    // ---- global pass (honors the current preprocess mode, like any single-pass run) ----
    // Forward at the conf floor so the cached pool covers the whole slider range.
    Config g = cfg;
    g.conf_thresh = std::min(cfg.conf_thresh, conf_floor);
    be.infer(bgr, g);
    out.pre_ms += be.pre_ms;
    out.infer_ms += be.infer_ms;
    out.post_ms += be.post_ms;
    // Snapshot immediately: every subsequent infer() overwrites the backend's cached state.
    std::vector<RawDet> global_cands = be.candidates;
    const LetterboxInfo global_lb = be.cand_lb;
    out.model_is_seg = be.is_seg();
    const bool keep_masks = sc.keep_global_masks && be.is_seg();
    std::vector<float> global_proto;
    int proto_c = 0, proto_h = 0, proto_w = 0;
    if (keep_masks) {
        global_proto = be.proto;
        proto_c = be.proto_c; proto_h = be.proto_h; proto_w = be.proto_w;
    }

    const int tile = clamped_tile_size(sc.tile_size, cfg.imgsz, w, h);
    const int overlap = static_cast<int>(tile * sc.overlap_ratio);
    const std::vector<TileRect> grid = tile_grid(w, h, tile, overlap);
    out.tile_size_used = tile;
    out.tiles_total = static_cast<int>(grid.size());

    // ---- tile selection ----
    std::vector<TileRect> selected;
    if (sc.mode == SliceMode::Sparse) {
        // 1/8-scale objectness mask painted from the global pass (predictor.py:555-568).
        // Painted from floor-NMS'd candidates at FIXED conf 0.05 / IoU 0.5 / Standard NMS so
        // tile selection is independent of the UI sliders and the CW toggle; the 0.15
        // objectness gate does the work.
        const int mH = h / kSliceMaskScale + 1, mW = w / kSliceMaskScale + 1;
        std::vector<float> mask(static_cast<size_t>(mH) * mW, 0.f);
        Config paint = cfg;
        paint.conf_thresh = conf_floor;
        paint.iou_thresh = 0.5f;
        paint.max_det = 300;
        paint.nms_mode = NmsMode::Standard;
        for (const Detection& d : nms_and_cap(global_cands, paint, w, h)) {
            // Upstream: (xyxy / 8).astype(int) - truncation - then clamp to the mask dims.
            const int x1 = std::max(0, static_cast<int>(d.box.x) / kSliceMaskScale);
            const int y1 = std::max(0, static_cast<int>(d.box.y) / kSliceMaskScale);
            const int x2 = std::min(mW, static_cast<int>(d.box.x + d.box.width) / kSliceMaskScale);
            const int y2 = std::min(mH, static_cast<int>(d.box.y + d.box.height) / kSliceMaskScale);
            if (x2 <= x1 || y2 <= y1) continue;      // sub-8px box paints nothing (upstream)
            for (int yy = y1; yy < y2; ++yy)
                for (int xx = x1; xx < x2; ++xx)
                    mask[static_cast<size_t>(yy) * mW + xx] =
                        std::max(mask[static_cast<size_t>(yy) * mW + xx], d.conf);
        }
        for (const TileRect& t : grid) {
            const int mY1 = t.y / kSliceMaskScale, mY2 = std::min((t.y + t.h) / kSliceMaskScale, mH);
            const int mX1 = t.x / kSliceMaskScale, mX2 = std::min((t.x + t.w) / kSliceMaskScale, mW);
            float v = 0.f;
            for (int yy = mY1; yy < mY2; ++yy)
                for (int xx = mX1; xx < mX2; ++xx)
                    v = std::max(v, mask[static_cast<size_t>(yy) * mW + xx]);
            if (v > sc.objectness_threshold) selected.push_back(t);   // strict >
        }
        // Upstream's all-or-nothing fallback (predictor.py:632-654): ONLY when zero tiles are
        // active, run the whole grid. If any tile is active, inactive ones are skipped.
        if (selected.empty() && sc.fallback && !grid.empty()) {
            selected = grid;
            out.used_fallback = true;
        }
    } else {
        selected = grid;                             // dense: every tile unconditionally
    }

    out.capped = static_cast<int>(selected.size()) > sc.max_tiles;
    if (out.capped) selected.resize(sc.max_tiles);   // row-major prefix

    // ---- run tiles sequentially, offset into full-image coords ----
    // Tile coeffs are ALWAYS stripped (meaningless vs the full-image proto). Global-pass
    // coeffs are kept only when the caller asked for masks.
    std::vector<RawDet>& pool = out.candidates;
    pool = std::move(global_cands);
    if (!keep_masks)
        for (RawDet& d : pool) d.mask_coeffs.clear();

    Config tg = g;
    tg.stretch = false;   // identity on a square canvas; documents the pad-to-square contract
    for (const TileRect& t : selected) {
        if (cancel && cancel->load()) { out.cancelled = true; break; }
        // Pre-pad the crop onto a tile x tile gray canvas (top-left, value 114 like the
        // letterbox pad). The backend letterboxes the square down to imgsz with a uniform
        // scale and zero padding - upstream _pad_slice at native scale, no backend changes.
        cv::Mat canvas(tile, tile, CV_8UC3, cv::Scalar(114, 114, 114));
        bgr(cv::Rect(t.x, t.y, t.w, t.h)).copyTo(canvas(cv::Rect(0, 0, t.w, t.h)));
        be.infer(canvas, tg);
        out.pre_ms += be.pre_ms;
        out.infer_ms += be.infer_ms;
        out.post_ms += be.post_ms;
        out.tiles_run += 1;
        for (const RawDet& d : be.candidates) {
            // Clip to real crop content (upstream lets boxes live on the gray padding),
            // then offset into full-image coordinates.
            const float x0 = std::max(0.f, d.box.x), y0 = std::max(0.f, d.box.y);
            const float x1 = std::min(static_cast<float>(t.w), d.box.x + d.box.width);
            const float y1 = std::min(static_cast<float>(t.h), d.box.y + d.box.height);
            if (x1 <= x0 || y1 <= y0) continue;
            RawDet r;
            r.box = cv::Rect2f(x0 + t.x, y0 + t.y, x1 - x0, y1 - y0);
            r.score = d.score;
            r.cls = d.cls;
            pool.push_back(std::move(r));
        }
    }
    std::sort(pool.begin(), pool.end(), [](const RawDet& a, const RawDet& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.box.x != b.box.x) return a.box.x < b.box.x;
        if (a.box.y != b.box.y) return a.box.y < b.box.y;
        if (a.box.width != b.box.width) return a.box.width < b.box.width;
        return a.box.height < b.box.height;
    });

    // ---- postcondition: the backend's cached state now describes the SLICED run ----
    be.candidates = pool;
    be.cand_lb = global_lb;
    be.cand_orig_w = w; be.cand_orig_h = h;
    if (keep_masks) {
        be.proto = std::move(global_proto);
        be.proto_c = proto_c; be.proto_h = proto_h; be.proto_w = proto_w;
    } else {
        be.proto.clear();
        be.proto_c = be.proto_h = be.proto_w = 0;
    }
    be.pre_ms = out.pre_ms;
    be.infer_ms = out.infer_ms;
    be.post_ms = out.post_ms;
    return out;
}

} // namespace yolomaster
