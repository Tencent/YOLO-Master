# Research map — optimization feasibility is not task utility

## Question and setting

Can a frozen Foundation Teacher transfer useful intermediate representation knowledge to a small YOLO detector? If gains are absent, which proposed explanations survive controlled checks?

Main setting: DINOv3-S → YOLO-Master-N; **Detect-input neck/FPN P4, source 19**, alignment dimension 64; COCO-mini. P2-07 uses train2048/val512, three paired seeds and 50 epochs. The detection endpoint is median mAP50-95 over CSV epochs 41–50, never best epoch.

## Research trajectory

| Stage | Question and retained outcome | Boundary |
| --- | --- | --- |
| P0 | Distillation extraction/alignment/gradient chain works; fixed-batch KD can decrease. | Engineering feasibility is not detection efficacy. |
| P1 | Static feature KD: **No-Go** under the frozen paired protocol. | No stable actionable gain here, not universal KD failure. |
| P2-01 | Gradient conflict: **Inconclusive**. | Insufficient evidence for this explanation; not ruled out generally. |
| P2-02 | Alignment 64→128: bottleneck hypothesis **not supported**. | Does not eliminate all capacity explanations. |
| P2-03 | Static alignment improvement did not transfer to measured local response alignment. | No proven causal link from response mismatch to task performance. |
| P2-04 | Raw response objective: **Calibration Failed** within the frozen alpha set. | That route stopped; later work does not rewrite its failure. |
| P2-05 | Local geometry/small response norm explanation supported. | Projector was not the primary amplifier in the tested probe; not a universal attribution. |
| P2-06 | Fixed detached norm compensation: **Feasibility Go**; combined gradient ratio about 15.632→1.151. | Frozen-state signal comparability, not matching throughout training or efficacy. |
| P2-07 | Aggregate response mechanism supported; detection **No detectable change**. | Route terminal. Mechanism support cannot rescue a negative primary result. |

Read the [formal recommendation](DINOV3_P2_FORMAL_RECOMMENDATION.md) for stage-by-stage references, distinguishing public package links from local-only original archives. The sequence is a research decision path, not a proven causal chain.

## Final comparison and result

**B / Static-2V** learns clean static KD plus perturbed static KD. **Cnorm** keeps clean static KD and replaces only the second term with fixed normalized finite-difference response matching. Both retained arms have two Student/two Teacher forwards and clean-view task loss; P2-07 is **B versus Cnorm, not OFF versus ON**.

| Seed | B late10 | Cnorm late10 | Cnorm − B |
| --- | ---: | ---: | ---: |
| 20260824 | 0.052435 | 0.050625 | -0.001810 |
| 20260825 | 0.047660 | 0.050705 | +0.003045 |
| 20260826 | 0.051340 | 0.048680 | -0.002660 |

Mean delta **-0.000475**; paired 95% t CI (df=2) **[-0.008120910, +0.007170910]**. AP scale is 0–1: 0.003 is 0.3 mAP percentage points. The preregistered `abs(mean delta)<0.003 AND CI contains 0` rule gives **No detectable change**. The wide interval allows meaningful positive and negative effects; this is not equivalence.

Secondary pooled ResponseGap delta **-0.007060625**, image-cluster bootstrap CI **[-0.010139964, -0.003883837]**, meets pooled plus at-least-2/3-seed support. Seed 25's interval includes zero. This interval uses 128 training-diagnostic images and fixed models, not the primary training-seed t interval or a causal mediation test.

Sources: [paired results](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_normalized_response_formal/paired_results.json), [mechanism summary](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_normalized_response_formal/mechanism_summary.csv). They remain remote evidence, not local Commit 3 artifacts.

## Interpretation and recommendation

Supported: the engineering path works; static/response closeness can separate; local response-gradient geometry can be corrected; the measured response mechanism can improve without detectable clean-detection utility in this setting.

Not supported: stable clean-mAP improvement, SOTA, general Foundation-distillation ineffectiveness, equivalence, or a causal claim that response mismatch caused P1 No-Go. Do not describe gradient conflict or all capacity/optimization problems as eliminated.

Keep auditable response utilities; **do not enable normalized Response-Field training by default**. Further research requires a new hypothesis/protocol and authorization, not continued alpha tuning.

Matched compute refers to the six retained arms. The interrupted B24 run and pilot consumed extra compute; the completed B24 restarted from epoch zero. See [recovery disclosure](P2_07_RECOVERY_DISCLOSURE.md); `automatic_retries=0` does not mean no manual recovery occurred.

[Return to entry / verification / provenance](README.md) · [Frozen historical protocol](DINOV3_P2_NORMALIZED_RESPONSE_FORMAL_PROTOCOL.md)
