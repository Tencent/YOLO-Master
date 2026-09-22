# STAL experiment workflow

This directory contains the diagnostics, evaluation, and VisDrone export utilities for the A2 small-target adaptive
label-assignment experiments. The formal protocol uses three seeds, a separate Mosaic interaction control, and a clear
distinction between official VisDrone metrics and the project's supplemental COCO-style scale metrics.

The formal results, P2 parameter-sensitivity tables, and evidence-bounded explanation of the unmet P1 target are in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Training controls

The locked fixed-stride behavior remains the default because `stal_candidate_mode=fixed`, `stal_enabled=False`, and
`stal_stats=False`. The legacy `stal_enabled=True` switch remains supported and selects adaptive mode.

| Argument | Meaning | Default |
|---|---|---:|
| `stal_enabled` | Enable relative-area candidate-region relaxation | `False` |
| `stal_stats` | Record assignment counts in `results.csv` | `False` |
| `stal_candidate_mode` | Candidate policy: `pure`, `fixed`, or `adaptive` | `fixed` |
| `stal_small_topk` | Adaptive-mode top-k for small targets; standard TAL remains unchanged | `10` |
| `stal_area_threshold` | Target-to-current-training-image area ratio used by STAL | `0.01` |
| `stal_relaxation` | Maximum total width and height added to the candidate region | `8.0` pixels |
| `stal_warmup_epochs` | Linear ramp duration for the relaxation | `10.0` epochs |
| `stal_zero_positive_rescue` | Rescue an uncovered small GT with its best legal positive-quality candidate | `False` |

`stal_relaxation=8` means adding four pixels on every side because the implementation operates on center-width-height
boxes and increases total width and height by eight pixels.

Run the required three-way candidate-policy comparison with statistics enabled:

```bash
yolo train model=<MODEL> data=<VISDRONE_YAML> stal_candidate_mode=pure stal_stats=True \
  project=<RUNS_DIR> name=pure-tal-seed0 seed=0
yolo train model=<MODEL> data=<VISDRONE_YAML> stal_candidate_mode=fixed stal_stats=True \
  project=<RUNS_DIR> name=fixed-stride-seed0 seed=0
yolo train model=<MODEL> data=<VISDRONE_YAML> stal_candidate_mode=adaptive stal_stats=True \
  stal_area_threshold=0.01 stal_relaxation=8 stal_warmup_epochs=10 \
  project=<RUNS_DIR> name=adaptive-stal-seed0 seed=0
```

`pure` uses the unmodified GT candidate region. `fixed` preserves the repository's existing rule that expands a GT
dimension below the smallest stride to the fixed middle stride. `adaptive` keeps that existing behavior and adds the
relative-area relaxation. For backward compatibility, `stal_enabled=True` also selects `adaptive`; it conflicts with
`stal_candidate_mode=pure`.

For the formal P0/P1 protocol use complete VisDrone train, all 548 validation images, YOLO-Master v0.1-N, `imgsz=800`,
`epochs=120`, and `patience=0`. Subset or short-cycle experiments are screening/smoke evidence only.

The submitted three-seed `adaptive-w0` arm used `stal_candidate_mode=adaptive`, `stal_area_threshold=0.0016`,
`stal_relaxation=8`, `stal_warmup_epochs=0`, and `stal_small_topk=stal_small_topk_min=10`. Candidate-quality gates,
minimum-candidate rescue, NWD, and SimD remained disabled so the comparison isolates candidate-region relaxation.

Run adaptive STAL using the backward-compatible flag if needed:

```bash
yolo train model=<MODEL> data=<VISDRONE_YAML> \
  stal_enabled=True stal_stats=True \
  stal_area_threshold=0.01 stal_relaxation=8 stal_warmup_epochs=10 \
  project=<RUNS_DIR> name=stal-seed0 seed=0
```

The following groups are added to `results.csv` when `stal_stats=True`: `stal`, COCO-style `small`/`medium`/`large`,
and `all`. Every group reports GT count, positive count, positives per GT, zero-positive GT count, and zero-positive
ratio. The original compatibility columns remain, including:

- `assign/stal_gt` and `assign/stal_pos`
- `assign/pos_per_stal_gt`
- `assign/all_gt` and `assign/all_pos`
- `assign/pos_per_gt`

All training groups are defined on the augmented, resized training canvas. This is mechanism evidence and is
deliberately separate from validation AP, which uses GT bbox area in the original validation images.

Score diagnostics now distinguish `assign/stal_*` (relative area below `stal_area_threshold`) from
`assign/small_*` (absolute bbox area below 32 squared pixels), alongside `assign/all_*`. Historical
`small_nonzero_score_*` and `small_target_score_*` columns used the relative STAL gate; do not interpret those old
columns as COCO-style small statistics. Start a new output directory with this schema; incompatible CSV appends fail.
E2E models report one-to-many diagnostics and drain both branches, avoiding duplicate GT counts. Counters reset
at every epoch attempt so OOM replay does not include abandoned batches.
The unconditional nearest-candidate guarantee cannot be combined with a positive geometric IoU capacity floor;
otherwise it could reintroduce a candidate rejected by that floor. Both remain disabled by default.

When `stal_zero_positive_rescue=True`, `results.csv` also records a staged rescue funnel:

- `assign/rescue_attempted`: uncovered STAL-area GTs presented to rescue.
- `assign/rescue_has_legal_candidate`: attempted GTs with at least one candidate in the relaxed region.
- `assign/rescue_has_free_candidate`: attempted GTs with a legal candidate not already assigned to another GT.
- `assign/rescue_has_positive_quality_candidate`: attempted GTs whose best free candidate has a finite positive
  task-alignment score.
- `assign/rescue_proposed`, `assign/rescue_succeeded`, and `assign/rescue_lost_to_conflict`: proposals before the
  second conflict pass, surviving rescues, and proposals removed by that conflict pass.
- `assign/rescue_bootstrap_proposed` and `assign/rescue_bootstrap_succeeded`: zero-quality targets proposed and retained
  through the optional center-prior bootstrap path.

These counters diagnose why rescue does or does not reduce zero-positive targets. They are assignment-mechanism
evidence, not an AP improvement claim.

With `stal_stats=True`, the behavior-neutral assignment funnel also records:

- `assign/stal_no_legal_candidate`: STAL-area GTs with no point in the candidate region.
- `assign/stal_legal_zero_alignment`: candidate-covered STAL-area GTs whose alignment is exactly zero.
- `assign/stal_sub_eps_alignment`: STAL-area GTs with positive alignment no greater than assigner epsilon.
- `assign/stal_above_eps_alignment`: STAL-area GTs with alignment above assigner epsilon.
- `assign/stal_topk_missed_nonzero`: nonzero-alignment STAL-area GTs not selected before conflict resolution.
- `assign/stal_preconflict_positive`: STAL-area GTs selected before conflict resolution.
- `assign/stal_conflict_lost`: selected STAL-area GTs left uncovered by conflict resolution.
- `assign/stal_postconflict_zero`: all STAL-area GTs left uncovered before optional rescue.

The funnel is collected only when statistics are enabled and does not modify candidate masks, ranking, target scores,
or losses.

The bootstrap path is disabled by default. Set `stal_rescue_score_floor` to a value in `(0, 1]` to let an uncovered
small GT with no positive task-alignment score nominate its nearest free legal anchor. The selected anchor receives that
value as a minimum target-score weight, so it contributes bounded classification and localization supervision instead
of a zero-weight label. `stal_rescue_floor_decay_epochs=N` linearly decays this floor to zero by epoch `N`; zero keeps
the configured floor constant. A positive floor requires both `stal_zero_positive_rescue=True` and adaptive mode.

## Size-binned validation AP

First emit Ultralytics prediction JSON from the selected checkpoint:

```bash
yolo val model=<CHECKPOINT> data=<VISDRONE_YAML> save_json=True max_det=500 \
  project=<RUNS_DIR> name=<VAL_NAME>
```

Then build COCO ground truth from the YOLO labels and evaluate the same predictions:

```bash
python scripts/stal/evaluate_scale_ap.py \
  --data <VISDRONE_YAML> \
  --predictions <RUNS_DIR>/<VAL_NAME>/predictions.json \
  --annotations-out <RUNS_DIR>/<VAL_NAME>/visdrone-val-coco.json \
  --metrics-out <RUNS_DIR>/<VAL_NAME>/scale-ap.json
```

The evaluator uses `maxDets=[1,10,500]` and reports `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `AP50s`, `AR500`,
`ARs500`, `ARm500`, and `ARl500`. The bins are the project's COCO-style supplemental definition on original-image
GT bbox area: small `<32²`, medium `[32²,96²)`, and large `>=96²`. They are not official VisDrone area bins.

The formal P1 gate is an absolute increase of at least `0.01` in `APs@[IoU=.50:.95,maxDets=500]`, i.e. 1.0 AP point.
Use original VisDrone annotations and an official-compatible evaluator for claims labelled official VisDrone AP/AP50/
AP75/AR500; a ground truth reconstructed from YOLO labels may not retain ignore/truncation/occlusion metadata.
Install `faster-coco-eval>=1.6.7` in the experiment environment before running it.
Evaluation now rejects missing label files; use an explicitly empty label file for a verified background image.
All returned AP/AR values are fractions, with `-1` for unavailable GT bins. The CLI explicitly marks these as
supplemental metrics rather than official VisDrone scores.

Export the same predictions to the official eight-field VisDrone DET submission layout (including empty result files):

```bash
python scripts/stal/export_visdrone_results.py \
  --predictions <RUNS_DIR>/<VAL_NAME>/predictions.json \
  --images <ORIGINAL_VISDRONE_VAL>/images \
  --output <RUNS_DIR>/<VAL_NAME>/visdrone-official-results
```

Run those TXT files with the official `VisDrone2018-DET-toolkit` and the original annotation directory. The official
toolkit removes detections in ignored regions and reports AP/AP50/AP75 plus AR@1/10/100/500. Keep its console output as
a separate evidence artifact from `scale-ap.json`.
