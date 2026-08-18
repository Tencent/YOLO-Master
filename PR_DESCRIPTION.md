# MoT boundary fixes, YAML-reachable shift policy, and drift-immune latency benchmarking

Branch: `feat/mot-boundary-fixes-and-hybrid-configs` (7 commits, based on `main` @ 296a29c)

## What this changes

Five defect fixes in the MoT stack, the boundary-test suite that found them, four
config-only architecture variants, and a correction to how the ablation harness
measures latency.

Nothing here changes the behaviour of an existing YAML or checkpoint. The one
change that touches numerics at all (the Swin shift mask) moves VisDrone mAP50-95
by −0.0089%, roughly 200× smaller than the run-to-run jitter band.

## Defects fixed

Each was reachable from a plain model YAML, and none raised at construction time —
they surfaced as a crash or a silently degraded model much later.

| # | Defect | Symptom before |
|---|---|---|
| 1 | `window_size <= 0` unvalidated | `ZeroDivisionError` deep inside the window expert |
| 2 | `n_points = 0` unvalidated | Deformable attention branch silently zeroed; expert degenerated to its FFN while still reporting as deformable |
| 3 | `shift_size` coerced through `bool` | Any truthy integer became `win//2`; an explicit shift was unreachable |
| 4 | Shifted windows had no attention mask | Cyclic roll brought opposite image edges into one window, so distant tokens attended each other — information leaking across the feature-map border, which canonical Swin masks |
| 5 | `muon_update` only reshaped 4-D grads | `zeropower_via_newtonschulz5` asserts 2-D, so MuSGD died on any model with 3-D params. MoT-N has 12; the baseline has 0 — the optimizer was unusable on the routed architectures |

Two supporting changes: `_sdpa` now converts a boolean mask to `0/-inf` before the
pre-2.0 fallbacks (which *add* the mask), without which fix 4 would only be correct
on PyTorch ≥ 2.0; and expert outputs are cast to the accumulator dtype before
blending, so an expert returning a different precision under AMP cannot silently
upcast or error out of the weighted sum.

## `window_shift` is now reachable from YAML

The per-block shift policy was hardcoded as `bool(i % 2)` inside the `C2fMoT`
constructor, which made a shift-strategy ablation impossible to express in a model
YAML — the one experiment the Swin-style alternation invites.

```yaml
# args: [c2, num_heads, top_k, window_size, n_points, mlp_ratio, temperature,
#        balance_loss_coeff, e, sparse_train, scene_aware_router, scene_hidden_dim,
#        scene_consistency_coeff, sparse_train_warmup_steps, scene_inference_mode,
#        window_shift]
- [-1, 2, C2fMoT, [128, 4, 2, 7, 4, 2.0, 1.0, 0.01, 0.5, False, False, null, 0.0, 0, dynamic, true]]
```

`"alternate"` (the default) is byte-identical to the previous behaviour; `true`/`false`
force one policy onto every block. Anything else raises `ValueError` instead of being
silently coerced. Flags resolve once at build time, so inference stays deterministic
and tracing stays export-stable.

Covered twice — at the constructor and end-to-end through `DetectionModel` — since a
constructor accepting a parameter that `tasks.py` never forwards would leave the
ablation just as unreachable as before.

## Latency measurement was wrong for cross-arm comparison

`benchmark_row` measures each model in one contiguous block, so drift in GPU state
during that block — vGPU contention, clock throttling — is charged entirely to
whichever arm happened to hold the GPU. On our host that flipped conclusions between
sessions: the baseline measured p50 **16.87 ms** in one run and **11.66 ms** in
another, and the `h3` arm came out both faster *and* slower than pure MoT depending
on run order.

`--interleave` keeps every model resident and times one forward per arm per cycle,
rotating visit order so no arm sits at a fixed position. Re-measuring 8 arms over 300
cycles gave a first-half/second-half p50 spread of **≤ 0.11 ms per arm** with stable
ordering, against the ~5 ms inter-session disagreement before.

Rows carry a `sampling` field, and interleaved rows are never shadowed by contiguous
ones when merging into an existing CSV — otherwise a later contiguous run would
silently overwrite the trustworthy numbers.

**Scope limit, stated because it bit us:** interleaving is correct for comparing
*arms*. It cannot measure *within-arm* jitter, since each arm's samples are spread
across the session. That needs per-arm contiguous blocks with block-level
interleaving, which this flag does not do. We had published a tail-latency-stability
claim that neither reading supports; it has been retracted.

## New configs (config-only, per project convention)

| Config | Placement | Params | GFLOPs (actual) | P50 | P99 |
|---|---|:---:|:---:|:---:|:---:|
| *baseline `yolo-master-n`* | *backbone MoE* | *3.450 M* | *8.66* | *12.12* | *19.86* |
| `hybrid-h2` | MoT P4 + MoA P5 | 3.896 M | 12.20 | 25.08 | 39.03 |
| `hybrid-h3` | MoT P4 only (2 blocks) | 3.759 M | 10.61 | 17.74 | 27.82 |
| `hybrid-h4` | MoA P4 + MoT P5 | 3.748 M | 10.69 | 20.79 | 31.79 |
| `mot-backbone` | MoT replaces P4/P5 backbone MoE, dense neck | 3.202 M | 9.78 | 13.15 | 19.83 |

**These are added for reproducibility, not as recommendations.** None beats the MoE
baseline on VisDrone (100 epochs, seed 42). The best, `h2`, reaches +1.11% mAP50-95 on
the final epoch but falls under the +1% threshold on both the best-epoch and
last-20-epoch readings — three readings straddling the line means a tie, not a gain.

The multi-seed re-check has since finished (seeds 42/43/44, both arms, 100 epochs each)
and **the +1.11% did not reproduce**: pooled over n=3 the deltas are −1.58% (final),
−1.48% (best) and −1.42% (last-20), and the per-seed final deltas are +1.11% / −3.34% /
−2.46% — seed 42 was the only positive one. The paired 95% CI is [−0.0128, +0.0073],
which includes zero, so `h2` is statistically **indistinguishable** from the baseline
rather than worse. The underlying problem is that the criterion cannot resolve the
effect: the +1% threshold is 0.00172 mAP against a between-seed sd of 0.00291 (0.59×),
and the baseline's own three seeds span 0.00262 — 1.5× the threshold. Re-running the
baseline under a different seed can manufacture a ">1%" effect on its own.

Two results worth recording for anyone extending this:

- `mot-backbone` is the only routed arm with baseline-parity tail latency
  (P99 19.83 vs 19.86 ms) because it **replaces** rather than stacks routing blocks:
  3 MoE junctions become 1 MoE + 2 MoT, with no net increase. Its mAP is the lowest of
  the eight arms (−2.59%).
- The ES-MoE placement rule does not transfer to MoT. Published MoE results put
  backbone-only placement ahead of neck-only; for MoT the ordering is reversed, and
  removing the backbone/neck cascade did not recover the deficit — so cascading was
  not the cause. Caveat: this arm *replaces* the P4/P5 MoE rather than adding to it
  (0.248 M fewer params than baseline), so the honest reading is "at equal parameter
  budget, swapping backbone MoE for MoT loses", not "MoT fails in the backbone".

`mot-backbone` keeps MoE on P3 deliberately: the LocalConv expert attends over all
H·W tokens, so P3/8 at 640×640 is 6400 tokens and OOMs a 24 GB card (15.5 GiB single
allocation). P4/P5 are 16×/100× cheaper. Putting MoT on P3 needs a windowed or
linear-attention expert first.

## Routing interpretability tooling

`mot_routing_interpret.py`, `run_mot_routing_interpret.py`, and
`mot_routing_figures.py` instrument the routers to answer which expert a token is
dispatched to and whether that shifts with scene content.

The statistics deliberately go beyond per-scene means, because the first pass produced
a finding that did not survive scrutiny: "occlusion suppresses the Deformable expert"
had p=0.002, a consistent sign across arms, and a plausible mechanism — but occluded
images average 74.7 objects against 47.7 for unoccluded ones, and the effect vanished
(p=0.93) once the test was redone within the dense stratum. The tooling therefore
includes stratified permutation tests and BH FDR correction over the full comparison
family, so a density confound cannot pass as a routing result.

## Testing

```
tests/test_mot.py                            68 passed  (was 27)
tests/test_mot_routing_scene_contrasts.py    13 passed  (new)
tests/test_mot_ablation_summary.py            7 passed  (new)
tests/test_mot_routing_diagnostics.py         2 passed
tests/test_default_config_integrity.py        5 passed
tests/test_peft_optimizer_policy.py          12 passed  (3 new, for the muon fix)
                                            ─────────
                                            107 passed
```

`ruff check` and `ruff format --check` are clean on every line this branch adds. The
pre-existing violations in the touched files are left alone rather than swept into a
functional PR.

The three specified boundary cases are all covered. Two of them turned out to be true
negatives — `window_size` larger than the feature map already degraded correctly (the
expert clamps to `min(H, W)`), and `exploration_eps` was already gated on
`self.training`. They are tested rather than assumed, because "already correct" was a
claim about code that had not been run.

## Not included

`MoT_MoA_Ablation_Report.md` and three host-specific helper scripts
(`collect_final_results.py`, `convert_visdrone.py`, `monitor_training.sh`,
`run_mot_interpretability.py`) are left untracked. They hardcode absolute paths for
this machine. The report in particular is a superseded COCO128 run whose mAP values
are ~0.0002 and whose ordering does not reproduce on either real dataset.
