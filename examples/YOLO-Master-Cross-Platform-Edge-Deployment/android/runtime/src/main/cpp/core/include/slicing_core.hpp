// Slicing grid math - a faithful port of upstream YOLO-Master's Sparse SAHI grid
// (ultralytics/engine/predictor.py:573-590) shared by the CLI and the Windows GUI.
// Header-only and OpenCV-free on purpose: the grid/clamp logic is the parity-critical
// part and this keeps it testable with a bare compiler.
#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace yolomaster {

// Slicing mode. Dense = global pass + every grid tile ("traditional tiling").
// Sparse = Sparse SAHI: global pass + only tiles where the global pass found evidence
// (objectness mask > threshold), with an all-or-nothing dense fallback when nothing is found.
enum class SliceMode { Off, Dense, Sparse };

struct TileRect { int x = 0, y = 0, w = 0, h = 0; };

// Upstream's mask resolution divisor (predictor.py:556, mask_scale = 8). Fixed, not derived
// from slice size. Also drives the tile drop check.
inline constexpr int kSliceMaskScale = 8;

// The user-facing tile-size bound: no smaller than the model input, no larger than a quarter
// of the image's short side. When the image is too small for that upper bound the floor wins
// and slicing runs at imgsz. requested <= 0 means "auto" (the model's imgsz).
inline int clamped_tile_size(int requested, int imgsz, int width, int height) {
    const int ceiling = std::max(imgsz, std::min(width, height) / 4);
    if (requested <= 0) requested = imgsz;
    return std::min(std::max(requested, imgsz), ceiling);
}

// The upstream slice grid, exactly: stride = slice_size - overlap from 0 while < image dim;
// right/bottom tiles CLIPPED (not shifted back); a tile is dropped when integer division by
// the mask scale collapses it in either axis (y/8 == yMax/8 - so a 7-px sliver at offset 1
// survives, one at offset 0 doesn't).
inline std::vector<TileRect> tile_grid(int width, int height, int slice_size, int overlap) {
    std::vector<TileRect> tiles;
    const int step = slice_size - overlap;
    if (step <= 0 || width <= 0 || height <= 0) return tiles;
    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {
            const int y_max = std::min(y + slice_size, height);
            const int x_max = std::min(x + slice_size, width);
            if (y / kSliceMaskScale == y_max / kSliceMaskScale) continue;
            if (x / kSliceMaskScale == x_max / kSliceMaskScale) continue;
            tiles.push_back({x, y, x_max - x, y_max - y});
        }
    }
    return tiles;
}

// Aggregate slicing statistics over one or many images (a folder run), for the stats card.
struct TileStats {
    int tiles_run = 0;
    int tiles_total = 0;
    int fallbacks = 0;       // images where the sparse all-or-nothing fallback fired
    int capped = 0;          // images where the max-tiles cap truncated the run
    int tile_size_min = 0;   // per-image clamping can vary the tile edge across a folder
    int tile_size_max = 0;
    void add(int run, int total, int size_used, bool fell_back, bool was_capped) {
        tiles_run += run; tiles_total += total;
        fallbacks += fell_back ? 1 : 0; capped += was_capped ? 1 : 0;
        tile_size_min = tile_size_min == 0 ? size_used : std::min(tile_size_min, size_used);
        tile_size_max = std::max(tile_size_max, size_used);
    }
    // "640" or "640-750" for the stats card.
    std::string tile_size_label() const {
        return tile_size_min == tile_size_max ? std::to_string(tile_size_min)
            : std::to_string(tile_size_min) + "-" + std::to_string(tile_size_max);
    }
};

} // namespace yolomaster
