> **Archived protocol — not execution authorization.** P2-07 is complete and terminal. The historical sequence, deadlines and commands below are retained as evidence, not instructions to launch runs. Relative code/data paths describe the original evidence checkout, not this compact integration checkout.

Public access: [public derivative of this document](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/DINOV3_P2_NORMALIZED_RESPONSE_FORMAL_PROTOCOL.md). Original document identity remains local at packaging SHA `781a6f148a46d59a89ea40806de8e10eb7b98941`. The public package is a sanitized derivative, not the original Git history; its PROVENANCE.json distinguishes byte-preserved files from path-sanitized records. Local-only references below are not public links. Import edits: this notice and pinned link destinations only; archived figures are linked rather than embedded. Scientific text and numbers are unchanged. The protocol's original implementation is distinct from post-freeze public-API hardening in integration Commit 1 `3a62b6f480e2855c3034f9788f69b804b81a5b56`.

---

# P2-07 | Final normalized Response-Field detection experiment

Frozen before engineering pilot or formal efficacy access, 2026-09-09.
Owner: gao-666 (D2). User authorization: final-route instruction of 2026-09-09.
P2-06 evidence gate: `2a0397102434ddadadc23c8f979319817dbfd738`, Feasibility Go.
P2-04 permanently remains Calibration Failed; P2-05 and P2-06 are not rewritten.

## 1. Stage

Independent P2 efficacy experiment. P2-06 passed signal feasibility only, not detection efficacy.
This is the final Response-Field route; any final outcome ends research experiments.

## 2. Existing facts and reused implementation

Reuse DINOv3-S/BF16, YOLO-Master-N, Detect-input P4 tap (source 19), shared
64-channel projector, existing detector Trainer, frozen eight perturbations,
epoch-major-v1 batch identity, teacher detach, FP32 cosine and exact BN rollback.
Do not reimplement teacher, projector, backbone, Agent, routing or perturbations.
The fixed normalized objective is imported from the frozen P2-06 implementation.

## 3. Unknown

Does normalized response information improve clean detection relative to a second
static snapshot at matched paired-view computation, now that fixed-state signal
feasibility passed? Feasibility at epoch49 does not guarantee training-time matching.

## 4. Hypothesis

Replacing only perturbed-static supervision with fixed normalized-response
supervision improves detection. No tuning of coefficient, epsilon or lambda.

## 5. Competing explanations

Geometry correction may have no task utility; ResponseGap may not be a bottleneck;
Static-2V may suffice; response imitation may harm specialization; three seeds may
leave wide uncertainty. None justifies a new loss or a moved decision threshold.

## 6. Minimal formal experiment

Six fresh runs, order B24, Cnorm24, B25, Cnorm25, B26, Cnorm26; seeds
20260824/20260825/20260826. Each uses 50 epochs, batch4, imgsz256, workers0,
SGD lr0=.01, AMP false, deterministic true, DINOv3-S BF16, P4, align_dim64.
Inherit all remaining arguments from `configs/d2_v3_p1_on.yaml` and this commit's
default configuration. Archive complete resolved arguments before execution.
Dataset: frozen COCO-mini train2048/val512, seed20260901.
Fresh partial-pretrained initialization: `cache/yolo26n.pt`, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Never initialize formal training from pilot or ON64 epoch49 probe checkpoints.

Let `D(a,b)=mean_token(1-cos_FP32(a,b))`, with frozen strict fail-closed cosine.
`r=Zp-Zc`, `t=Tp-Tc`, and
`w=stop_gradient(norm(r)/(sqrt(2)*norm(Zp)+1e-6))` per token.

| Arm | Task | Foundation objective |
| --- | --- | --- |
| B Static-2V | clean only | D(Zc,Tc) + D(Zp,Tp) |
| Cnorm | clean only | D(Zc,Tc) + mean_token(w * (1-cos(r,t))) |

Each sum is multiplied by lambda=.15 and batch size, matching the existing
Foundation wrapper's detection-loss sum convention. No factor1/2, no alpha.
Teacher targets detached; full norm multiplier detached. No loss on perturbed GT.
Two Student and two Teacher image forwards in both arms per training batch.
Perturb the already-augmented clean image using the frozen helper. Source image
names are canonical `images/train2017/<basename>` (the primary source identifier
when mosaic is active); hash actual clean and perturbed tensors, not just names.
BN train semantics on both forwards; snapshot Student+projector buffers after
clean forward, restore exactly after perturbed forward, before backward.

## 7. One engineering pilot, no efficacy

B24/Cnorm24 each use the first32 batches from the ordinary train2048 loader and
the same 50-epoch configuration/warmup. Stop after batch32; val is disabled and
the validation loader/final evaluation must not access validation images.
At the boundary after batch16, serialize FP32 online Student+projector, optimizer,
EMA, scaler, scheduler and RNG, reconstruct a fresh wrapper and optimizer, restore
their state, and continue with the existing live loader iterator. Archive exact
state equality and logical batch16 continuation. This verifies model/optimizer
checkpoint handoff; it is NOT a claim of bitwise fresh-process data-loader resume.
Pending accumulated gradients, if present, must also be restored. The loop's
warmup index, accumulated-step cursor and loader position must remain unchanged.
Pilot weights are discarded as formal initializers.

Record per-batch finite loss, BN train/restoration, actual objective axis, paired
tensor digests, Teacher/Student counts, peak GPU memory, elapsed time and optimizer
steps. During pilot only, compute counterfactual B/Cnorm F-pair gradients from the
same cached graph using autograd.grad; no extra model forward. Report their ratio
descriptively; do not use a new gradient-ratio acceptance threshold or retune.
Gate:32 batches/arm, nonzero finite perturbed-branch/projector gradients, exact
BN restoration and state handoff, matching paired digests and initial state,
no validation reads, no automatic recovery or hyperparameter changes.
Technical failure stops formal dispatch pending diagnosis; do not turn a pilot
failure into an unregistered efficacy-driven retry. Fixes require an archived
technical explanation and commit before rerunning any incomplete pilot.

## 8. Observations

Primary: median of epochs41-50 mAP50-95 (CSV one-based epoch labels), per arm/seed.
Paired delta=Cnorm-B; mean, sample SD (ddof1), 95% Student-t CI (df2).
Secondary: `weights/epoch49.pt` EMA (zero-based index49, CSV epoch50, final
epoch; never substitute best.pt) ResponseGap(Cnorm)-ResponseGap(B), using unchanged P2-03
response128 list, eight conditions, spatial-relation embedding, image-cluster
bootstrap and numeric fail-closed contract. Mechanism Support requires pooled
CI upper<0 and at least2/3 seed CI upper<0; otherwise report uncertainty/harm.
Robustness secondary: epoch49 EMA clean plus all eight frozen corruption
conditions over val512; macro mAP50-95 and clean-minus-macro robustness drop.
Reuse P2-03 evaluation definitions without changing aggregation or perturbations.
No secondary result is used to select checkpoints, coefficients or run settings.

## 9. Falsification

No ResponseGap decrease: mechanism not supported. Gap decrease without clean
improvement: response correction not sufficient for clean task utility. Harm:
possible specialization conflict, not universal teacher failure. Wide CI:
inconclusive. No new trials follow any of these outcomes.

## 10. Frozen primary decision

- Task Support: mean delta>=.003 AND paired95% t CI lower>0.
- No detectable change: abs(mean delta)<.003 AND CI contains0.
- Harm: mean delta<=-.003 AND CI upper<0.
- All other cases: Inconclusive.

AP scale is0-1 (.003=0.3 AP percentage points). No best-epoch substitution,
additional seeds, budget increase or shifted threshold after viewing outcomes.

## 11. Three-layer no-confound audit

Planned configs match except arm/objective and non-treatment output identifiers.
Compare resolved args per seed, optimizer groups/lr schedule, data and initial
asset hashes, Teacher files, augmentation, AMP, epochs, batch and forward counts.
Compare actual paired tensor digest sequences and initial Student/projector state.
Independently assert actual second term=B static_perturbed or Cnorm fixed formula;
the argument audit alone does not establish a correct experimental axis.
OOM/nonfinite means fail-closed: no automatic smaller batch, recovery restart or
unrecorded budget changes. No formal run is silently resumed after interruption;
audit any interruption and preserve original evidence before further action.

## 12. Cost and deadlines

Six runs = 300 epochs, plus 64 total pilot batches; no A or raw Response arm.
Estimate runtime from the non-efficacy pilot. Decide/start before
2026-09-11T00:00:00+08:00; no new design from 2026-09-13T00:00:00+08:00.
Insufficient resources or missed cutoff means stop and report, not shorten runs.

## 13. Claim boundary

At most a scoped DINOv3-S/P4/COCO-mini matched-compute improvement with descriptive
mechanistic support. Not SOTA, universal KD superiority, or proven mediation.
Signal-matched is not performance-optimal. Preserve negative evidence.

## 14. Evidence

`results/p2_normalized_response_formal/`: pilot evidence; protocol/source/config
hashes; complete logs; runtime manifests; resolved args; epoch CSVs; checkpoint
hashes; paired_results.csv/json; mechanism_summary.csv; robustness_summary.csv;
final_result.md. Keep full checkpoints under ignored runs/ and hash them.
Protocol, implementation, pilot, and formal evidence receive separate commits.
No files under experiments/study/ may be changed or staged.

## 15. Next and terminal action

Freeze this protocol -> implement/test -> pilot -> engineering gate -> six runs
-> paired analysis and secondary evaluation -> final report and local PR draft.
Any scientific outcome ends this route. Public push/PR creation requires separate
authorization; current work is locally committed. No A24 or raw Response launch.
