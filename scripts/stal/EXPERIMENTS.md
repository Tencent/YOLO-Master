# A2 STAL experiment report

This report separates formal accuracy evidence from parameter screening. VisDrone DET does not publish
small/medium/large area metrics. Every `APs`, `APm`, `APl`, and `ARs500` value below is therefore a project
COCO-style supplement computed from original validation-image ground-truth boxes: small `<32^2`, medium
`[32^2, 96^2)`, large `>=96^2`, IoU `.50:.05:.95`, and `maxDets=500`. Training-time STAL gates and assignment
statistics instead use boxes after augmentation and resize on the actual training canvas.

## Acceptance status

| Stage | Requested evidence | Status |
|---|---|---|
| P0 | Full VisDrone baseline, scale metrics, and per-epoch positive-assignment statistics | Completed |
| P1 | Adaptive STAL improves formal three-seed APs over pure TAL by at least 1.0 absolute point | Not met |
| P2 | Scan area/scale controls and warmup, or extend to another dataset/task | Completed through the parameter-sensitivity route; no cross-dataset claim |

P2 completion describes the requested ablation work. It does not override the unmet P1 accuracy threshold.

## Relation to other A2 submissions

At review time, #274 emphasizes candidate expansion, per-tier top-k, a minimum-positive path, and detailed evidence
boundaries; #280 reports a compact three-way table and positive-coverage statistics; #293 keeps an A16
minimum-candidate design and reports its best single-seed attempt. Their training recipes, baselines, assignment rules,
and evaluation handling differ, so their APs values are not merged into the tables below. This submission's distinct
evidence is a matched FP32 three-seed pure/fixed/adaptive comparison plus the parameter and failure analysis in this
document.

## Formal protocol and result

All formal rows use complete VisDrone train/val (`6471/548`), YOLO-Master v0.1-N, `imgsz=800`, 120 epochs,
`patience=0`, batch 6, FP32, MuSGD, optimizer warmup 3, Mosaic 1 with `close_mosaic=10`, and the checkpoint selected
by the repository's normal overall-fitness rule. The submitted adaptive arm uses area threshold `0.0016`, constant
relaxation 8, assignment warmup 0, and small-target top-k 10.

| Assignment | Seed 0 APs (%) | Seed 1 APs (%) | Seed 2 APs (%) | Mean +/- SD (%) |
|---|---:|---:|---:|---:|
| pure TAL | 12.947795 | 13.262371 | 12.880747 | 13.030304 +/- 0.203752 |
| fixed-stride STAL | 13.064431 | 12.961409 | 13.156323 | 13.060721 +/- 0.097510 |
| adaptive-w0 | 13.414071 | 13.319065 | 13.135239 | 13.289458 +/- 0.141754 |

The paired adaptive-minus-pure mean is `+0.259154` absolute point with a 95% Student-t interval of
`[-0.249674, +0.767982]`. Relative to fixed STAL it is `+0.228738` point with interval
`[-0.308802, +0.766278]`. Both intervals cross zero, and the mean gain is below 1.0 point. The Mosaic-off seed-0
pair is also positive but small: pure `11.692718%`, adaptive `12.001212%`, delta `+0.308493` point.
The paired statistics use the unrounded evaluator outputs; the per-seed values displayed in the table are rounded to
six decimal places.

The retained prediction artifacts also support the required secondary scale metrics. The values below were
recomputed from the official-format TXT files, whose box coordinates are rounded to three decimals; this changes APs
by at most `0.0126` absolute point relative to the unrounded JSON values above. They are reported here for the
medium/large and small-recall context, while the unrounded JSON table remains the P1 acceptance source.

| Assignment | Seed | APm (%) | APl (%) | ARs500 (%) |
|---|---:|---:|---:|---:|
| pure TAL | 0 | 32.196537 | 41.687356 | 29.387689 |
| pure TAL | 1 | 32.774068 | 40.917719 | 29.232980 |
| pure TAL | 2 | 32.082815 | 44.245727 | 29.016809 |
| fixed-stride STAL | 0 | 31.888752 | 41.420740 | 29.430121 |
| fixed-stride STAL | 1 | 32.064242 | 44.290912 | 29.419533 |
| fixed-stride STAL | 2 | 32.016438 | 42.538901 | 29.394237 |
| adaptive-w0 | 0 | 31.756169 | 40.036916 | 29.287789 |

The official `VisDrone2018-DET-toolkit` commit `005445782213e20cb91bc50a597db3dd949e749a` produced the
following overall metrics on all 548 original validation annotations. These official metrics do not contain area
bins and are not substituted for the project APs above.

| Assignment | Seed | Official AP (%) | AP50 (%) | AP75 (%) | AR500 (%) |
|---|---:|---:|---:|---:|---:|
| pure TAL | 0 | 22.450013 | 40.160397 | 21.507844 | 39.137572 |
| pure TAL | 1 | 22.602562 | 40.447796 | 21.710440 | 39.197234 |
| pure TAL | 2 | 22.404797 | 40.309586 | 21.363486 | 38.831045 |
| fixed-stride STAL | 0 | 22.636834 | 40.990155 | 21.477616 | 39.205206 |
| fixed-stride STAL | 1 | 22.921176 | 41.358936 | 21.750081 | 39.233329 |
| fixed-stride STAL | 2 | 22.741519 | 41.152091 | 21.607634 | 39.226400 |
| adaptive-w0 | 0 | 22.743838 | 41.140512 | 21.415664 | 39.090666 |

The adaptive-w0 seed-1/2 predictions have project APs results but are absent from the retained official-evaluation
package, so no official three-seed adaptive mean is claimed.

## P2 parameter sensitivity

These screens choose mechanisms and expose tradeoffs; they are not formal P1 results. Each table states its own
budget so that short-run values are not mixed with the formal table.

### Area gate and relaxation

The original relative-area gate was 1%. Screens narrowed the affected training targets to 0.5%, 0.3%, and 0.16%; the
0.16% gate was the only candidate promoted to a full 120-epoch check. At seed 0, moving from the original adaptive
arm to the 0.16% gate with relaxation 8 changed APs from `13.167447%` to `13.337117%`. A mild area-adaptive top-k
variant reached `13.364341%`, only `+0.027224` point beyond the fixed top-k version. This supports narrowing the
intervention, but the final three-seed gain remained small.

A separate 10%-train, 10-epoch screen varied only constant relaxation. Its endpoint is framework mAP50-95, not APs:

| Total width/height relaxation (px) | mAP50-95 | Small positives / GT | Small zero-positive ratio |
|---:|---:|---:|---:|
| 4 | 0.01176 | 3.71981 | 0.153254 |
| 6 | 0.00891 | 3.97614 | 0.159743 |
| 8 | 0.01103 | 4.17120 | 0.156968 |

More expansion increased positive count, but it did not monotonically improve the zero-positive ratio or accuracy.
On fixed real batches, increasing relaxation from 0 to 8 also raised conflict loss among pre-conflict-covered targets
from `10.49%` to `12.31%`, while the fraction with alignment above epsilon stayed near `7.64%`.

Continuous square-root area scaling was also negative on the same 10%-train, 10-epoch protocol:

| Relaxation rule | APs |
|---|---:|
| constant r8 control | 0.009286 |
| sqrt-area r0-to-r8 | 0.007113 |
| sqrt-area r4-to-r8 | 0.006761 |

### Warmup and top-k

The full-data 24-epoch prefix screen retained the 120-epoch learning-rate schedule and evaluated fixed checkpoints.

| Assignment warmup | Small top-k | Epoch-24 APs (%) | Epoch-24 ARs500 (%) |
|---:|---:|---:|---:|
| 0 | 10 | 9.207494 | 23.019632 |
| 5 | 10 | 8.964761 | 23.392646 |
| 10 | 10 | 8.626137 | 22.921022 |
| 10 | 5 | 8.685307 | 23.082611 |

Warmup 0 led at the fixed 24-epoch checkpoint and was therefore promoted. This is a screening decision rather than
evidence that 24-epoch ranking predicts 120-epoch ranking. A retrospective four-arm check found Spearman correlation
between early framework mAP50-95 rank and final APs rank of `-0.8` at epoch 10, `0.0` at epoch 20, and `0.8` at epoch
40; four arms are enough to reject epoch-10 ranking as a dependable gate, not enough to establish a general predictor.

An independent area-adaptive top-k screen used 10% train for 10 epochs:

| Tiny-to-threshold top-k | mAP50-95 | Small positives / GT | Target score / positive |
|---|---:|---:|---:|
| 10-to-10 control | 0.01103 | 4.17120 | 0.153921 |
| 2-to-10 | 0.01091 | 2.26906 | 0.252716 |
| 5-to-10 | 0.01058 | 3.29717 | 0.187562 |

Reducing the nomination budget improved average selected-positive quality but reduced recall and did not improve the
screen endpoint. Uniform larger top-k arms and the final 8-to-10 variant likewise did not show a material advantage.

### Coverage, supervision quality, and auxiliary loss

The final epoch of a matched seed-0 telemetry comparison reports all three training-canvas size bins. The focus row
uses the selected `0.0016` area gate and constant relaxation 8; it predates the later warmup-0 selection, so it is
mechanism evidence rather than a substitute for the adaptive-w0 accuracy table.

| Assignment | Small pos/GT | Small zero (%) | Medium pos/GT | Medium zero (%) | Large pos/GT | Large zero (%) |
|---|---:|---:|---:|---:|---:|---:|
| pure TAL | 3.50122 | 18.8060 | 9.93647 | 0.024975 | 9.98187 | 0.000000 |
| fixed-stride STAL | 4.05073 | 6.92296 | 9.93314 | 0.038597 | 9.98302 | 0.000000 |
| focus t0016-r8 | 5.79637 | 6.26367 | 9.93364 | 0.027245 | 9.97994 | 0.000000 |

A 1618-image, 24-epoch quality-aware coverage screen reduced the training zero-positive ratio from `10.2975%` to
`8.9193%` and raised ARs500 by `0.2018` point, while APs changed from `4.127463%` to `4.108997%`. Likewise, adding at
most one or two adaptive-only candidates sharply reduced zero-positive incidence in an earlier screen but lowered
APs. These paired results directly show that positive coverage alone is not a sufficient promotion criterion.

Changing MoE auxiliary-loss strength was kept separate from STAL geometry. Under auxiliary strength 1, focus-minus-
pure APs was `+0.389322` point at seed 0; under strength 3 it was only `+0.075472` point. The interaction is evidence
that optimization context matters, but auxiliary strength is not claimed as an STAL contribution.

## Why the gain stopped below one point

The following are confirmed observations:

- Candidate expansion activates and materially changes assignment. In the original formal seed-0 comparison it
  raised augmented-small positives per GT from `4.045` (fixed) to `5.904` and reduced zero-positive incidence from
  `6.98%` to `5.95%`.
- The extra assignments are not monotonic evidence of useful supervision. Relaxation increased conflicts, and
  coverage-focused screens improved zero-positive or recall statistics while APs stayed flat or fell.
- Fixed versus original adaptive prediction analysis showed a larger gain at AP50s (`+0.3761` point) but a small loss
  at AP75s (`-0.0489` point) and essentially unchanged ARs500 (`-0.0153` point). At confidence 0.1, unmatched
  small-prediction diagnostics counted 1356 more duplicate/competition boxes and 1108 more localization/mixed-error
  boxes for adaptive. These are geometric diagnostic labels, not an additive AP-loss decomposition.
- Per-class APs moved in both directions: car improved by `+0.577` point, while people, van, and truck changed by
  `-0.242`, `-0.122`, and `-0.251` point in that seed-0 comparison.
- The final three-seed gain varies enough that its confidence interval crosses zero.

The most plausible interpretation is that candidate expansion solves part of the coverage problem but also adds weak
or competing supervision. That can improve coarse-IoU detections while failing to improve stricter localization and
ranking enough for APs averaged over ten IoU thresholds. The training gate also acts on augmented relative area,
whereas APs is grouped by original-image area, so not every evaluated small object receives the same intervention.
These mechanisms are consistent with the observations but have not been isolated as a complete causal decomposition.

The saved predictions are post-processing outputs (`conf>=0.001`, at most 500 detections per image), so they cannot
separate failures originating in the raw head from NMS or output truncation. The report therefore does not attribute a
numerical fraction of the AP gap to localization, classification, duplicates, or missed detections.

## Reproduction and validation boundaries

Use the commands in [README.md](README.md) for training, supplemental scale evaluation, and official VisDrone TXT
export. The submitted code has focused tests for candidate geometry, configuration contracts, pure/fixed/adaptive
behavior, empty and overlapping targets, conflict handling, FP16 area arithmetic, AMP finite loss/gradients,
telemetry, scale evaluation, and VisDrone export validation.

The official VisDrone toolkit was run separately on the original 548-image annotation set for pure and fixed across
all three seeds and adaptive seed 0. Official AP/AP50/AP75/AR values do not contain area bins and are not substituted
for project APs. The full P2 route in this report is parameter sensitivity on VisDrone; no DOTA/AI-TOD, segmentation,
or pose generalization result is claimed.
