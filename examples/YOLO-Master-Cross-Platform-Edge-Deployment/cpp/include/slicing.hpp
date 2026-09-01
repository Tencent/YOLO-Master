// Sliced inference (Sparse SAHI + dense) over the shared Backend interface - the C++ port of
// the Mac runner's tiled inference (mac/Sources/YOLOMasterKit/Tiling.swift), itself a faithful
// port of upstream YOLO-Master's Sparse SAHI Mode (ultralytics/engine/predictor.py:542-699).
//
// Shape of the port (deviations from upstream are deliberate and documented):
//   * Upstream NMSes each tile at the caller's conf/iou, then merges with one more NMS.
//     Here NO per-tile NMS runs: the merged pool is PRE-NMS candidates (conf floor 0.05) from
//     the global pass + every executed tile, offset into full-image coordinates. The single
//     user-conf/iou NMS at render time subsumes per-tile NMS, keeps the cached pool independent
//     of the UI sliders (the GUI's cache-and-retune pattern), and hands CW-NMS the pre-NMS pool
//     its weighting wants.
//   * Tile detections are clipped to real crop content; upstream lets boxes live on the gray
//     padding.
//   * TILE detections are always boxes-only: their mask coefficients are meaningless against a
//     full-image proto tensor and are stripped. GLOBAL-pass detections keep coeffs (and the
//     backend keeps the global proto) only when keep_global_masks is set.
//   * A max_tiles safety cap bounds the tiles RUN per image (row-major truncation, surfaced in
//     SliceOutput.capped) so a gigapixel input degrades gracefully instead of hanging.
//
// Tile forwards need NO backend changes: each crop is pre-padded onto a tile x tile gray-114
// canvas (top-left), so the backend's own letterbox sees a square and scales uniformly by
// imgsz/tile with zero padding - upstream _pad_slice semantics exactly.
#pragma once
#include "yolomaster.hpp"
#include "slicing_core.hpp"
#include <atomic>

namespace yolomaster {

// Slicing parameters. The UI exposes mode, tile_size and keep_global_masks; the rest stays at
// the upstream defaults (default.yaml:75-79) and is exposed here for the CLI/tests.
struct SliceConfig {
    SliceMode mode = SliceMode::Off;
    // Requested tile edge in source pixels. <= 0 = the model's imgsz (native scale, the
    // upstream default). Clamped PER IMAGE to [imgsz, max(imgsz, min(w,h)/4)]; tiles larger
    // than imgsz are letterboxed down to the model input (upstream slice_size semantics).
    int tile_size = 0;
    // Keep the GLOBAL pass's mask coefficients (+ proto tensor) so segmentation masks still
    // render for global-pass detections in sliced modes.
    bool keep_global_masks = false;
    float overlap_ratio = 0.2f;          // overlap = int(tile * ratio); upstream 0.2
    float objectness_threshold = 0.15f;  // strict > gate on the painted mask
    bool fallback = true;                // sparse only: zero active tiles -> run ALL
    int max_tiles = 256;                 // safety cap on tiles RUN per image
};

// Result of a sliced inference: the merged candidate pool plus run statistics for the UI.
struct SliceOutput {
    std::vector<RawDet> candidates;      // global + tile pool, full-image px, score-desc
    int tiles_total = 0;                 // grid tiles surviving the drop check
    int tiles_run = 0;
    int tile_size_used = 0;              // per-image clamped tile edge actually used
    bool used_fallback = false;          // sparse: zero active tiles -> dense fallback fired
    bool capped = false;                 // max_tiles truncation hit
    bool cancelled = false;              // the cancel token fired mid-run
    bool model_is_seg = false;           // the GLOBAL pass produced a proto (even if dropped)
    // Per-stage sums across the global pass and every executed tile.  Keeping
    // all three stages here prevents a sliced benchmark from accidentally
    // reporting the last tile's pre/post time as the whole image's timing.
    double pre_ms = 0;
    double infer_ms = 0;
    double post_ms = 0;
};

// Global forward + (dense: all tiles | sparse: mask-selected tiles) -> merged pre-NMS pool.
// cfg.imgsz stays constant across every forward (keeps ncnn's fixed input and MNN's session
// shape untouched). sc.mode must not be Off - callers branch to the single-pass path first.
//
// POSTCONDITION (load-bearing): the backend's cached per-call state is rewritten to describe
// the SLICED run, so every existing consumer (GUI recompute_nms/folder snapshot, CLI decode
// path) works unchanged:
//   be.candidates    = the merged pool
//   be.cand_lb       = the GLOBAL pass letterbox; be.cand_orig_w/h = full image dims
//   be.proto/_c/_h/_w = the global proto when keep_global_masks on a seg model, else cleared
//   be.pre_ms/infer_ms/post_ms = the corresponding sums across all forwards
//   ("model-only" inference time is the sum of the individual forwards)
// `cancel` (optional) is polled between tile forwards; on cancel returns cancelled=true with
// the pool accumulated so far.
SliceOutput sliced_candidates(Backend& be, const cv::Mat& bgr, const Config& cfg,
                              const SliceConfig& sc, float conf_floor = 0.05f,
                              const std::atomic<bool>* cancel = nullptr);

} // namespace yolomaster
