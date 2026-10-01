/* YOLO-Master Edge portable core: C interface for hosts that cannot consume C++ directly
 * (the Swift package imports this header as a plain C module, no C++ interop).
 *
 * Everything here is a thin shim over metrics_core.hpp (in-process mAP), tracker_core.hpp
 * (BoT-SORT / ByteTrack) and bench_stats.hpp (benchmark statistics, hashes). Memory returned
 * through `out` pointers is owned by the core and released with the matching *_free call.
 */
#ifndef YMCORE_H
#define YMCORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- version ---------------------------------------------------------------------------- */
int ym_core_version(void);                 /* 1 */

/* ---- in-process mAP ---------------------------------------------------------------------- */
typedef struct { double x1, y1, x2, y2; int cls; } YmGtBox;
typedef struct { double x1, y1, x2, y2, conf; int cls; } YmPredBox;
typedef struct {
    const YmPredBox* preds; int n_preds;
    const YmGtBox* gts;     int n_gts;
} YmImageEval;
typedef struct { int cls; double ap[10]; int n_gt, n_pred; } YmClassAP;
typedef struct {
    int images;
    double map50, map5095;
    YmClassAP* per_class; int n_classes;   /* ascending class id; freed by ym_map_free */
} YmMapResult;

/* COCO-style mAP 0.50:0.95 over `n` images; identical to scripts/eval_map*.py. */
void ym_map_evaluate(const YmImageEval* images, int n, YmMapResult* out);
void ym_map_free(YmMapResult* r);
/* What "std::cout << float" prints (6 significant digits) read back: the --save-txt rounding. */
double ym_round6(double v);
/* YOLO ("cls cx cy w h" normalized) or VisDrone comma labels for an image of size w x h; a missing
 * or empty file yields 0 boxes and returns 1 (a valid empty ground truth). */
int ym_load_yolo_labels(const char* path, int img_w, int img_h, YmGtBox** out, int* n);
void ym_gt_free(YmGtBox* boxes);
/* ultralytics img2label_paths rule when labels_dir is NULL or "", else "<labels_dir>/<stem>.txt".
 * Returns the length written (excluding NUL); 0 when `cap` is too small. */
size_t ym_label_path_for(const char* image_path, const char* labels_dir, char* buf, size_t cap);

/* ---- benchmark statistics ---------------------------------------------------------------- */
typedef struct { size_t n; double mean, median, p90, p95, p99, min, max; } YmStageStats;
typedef struct { double cold_median_ms, sustained_median_ms, throttle_pct; } YmSustained;
/* Floor-rank percentiles (sorted[min(int(q*n), n-1)]); n == 0 -> all zero. */
void ym_stats_reduce(const double* samples, int n, YmStageStats* out);
/* cold = median of the first `cold_iters`, sustained = median of the slowest quarter. */
void ym_sustained_summary(const double* samples, int n, int cold_iters, YmSustained* out);
void ym_sha256_hex(const uint8_t* data, size_t len, char out[65]);
/* sha256 over the sorted basenames joined by '\n' (what scripts/make_coco_subset.py hashes). */
void ym_image_list_sha256(const char* const* paths, int n, char out[65]);
void ym_timestamp_utc(char out[32]);       /* "YYYY-MM-DDTHH:MM:SSZ" */

/* ---- tracking ---------------------------------------------------------------------------- */
typedef struct { float x, y, width, height; } YmBox;
typedef struct {
    YmBox box; float conf; int class_id;
    const float* mask_coeffs; int n_mask_coeffs;   /* optional, may be NULL / 0 */
} YmTrackInput;
typedef struct { double r00, r01, r10, r11, tx, ty; } YmMotion;   /* previous -> current frame affine */
typedef struct {
    int botsort;                 /* 1: BoT-SORT (xywh Kalman, motion applied); 0: ByteTrack (xyah) */
    float track_high_thresh;     /* 0.25 */
    float track_low_thresh;      /* 0.10 */
    float new_track_thresh;      /* 0.25 */
    float match_thresh;          /* 0.80 */
    int track_buffer;            /* 30 frames, scaled by fps / 30 */
    int fuse_score;              /* 1 */
    double fps;                  /* 30 */
} YmTrackerConfig;
typedef struct {
    int id, class_id; float conf; YmBox box;
    int age, hits, time_since_update;
    int state;                   /* 0 New, 1 Tracked, 2 Lost, 3 Removed */
    int det_index;               /* index into the update() input, -1 = coasting */
} YmTrack;
typedef struct YmTracker YmTracker;

void ym_tracker_default_config(YmTrackerConfig* out);
YmTracker* ym_tracker_new(const YmTrackerConfig* cfg);
/* One frame in order; `motion` may be NULL (no compensation). `out` is valid until the next
 * call on the same tracker or ym_tracker_free. */
void ym_tracker_update(YmTracker* t, const YmTrackInput* dets, int n, const YmMotion* motion,
                       const YmTrack** out, int* n_out);
void ym_tracker_reset(YmTracker* t);
int ym_tracker_frame_count(const YmTracker* t);
void ym_tracker_free(YmTracker* t);

#ifdef __cplusplus
}
#endif
#endif /* YMCORE_H */
