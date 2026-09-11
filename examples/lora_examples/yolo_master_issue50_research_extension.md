# Issue #50 Research Extension: Stability, Strong Baselines, and Audited Evaluation

## Scope

The existing Issue #50 contributions already provide domain configurations, launchers, and LoRA rank sweeps for
Brain Tumor and VisDrone. This extension does not replace those results or claim another rank-sweep contribution.
It asks a different question: under the evaluated protocols, does Stable LoRA compare favorably with conventional
fine-tuning, and which runs are sufficiently complete and stable to support that conclusion?

The extension adds:

- Head-only, Neck + Head, Last Stage + Neck + Head, and Full fine-tuning baselines;
- Stable LoRA and Partial fine-tuning + LoRA comparisons;
- multi-seed evaluation for the best Brain Tumor method and VisDrone Full fine-tuning;
- numerical-stability diagnostics for AMP and grouped Adapter/Router learning rates;
- explicit `requested_batch` and `actual_batch` fields;
- separate queue-execution and formal-validity semantics;
- duplicate-seed, duplicate-directory, OOM-recovery, and failed-implementation auditing; and
- stability-gated, multi-objective Pareto analysis.

Only rows that passed the audited formal-validity gate are included in
`yolo_master_issue50_audited_results.csv`. Failed, `implementation_failed`, and `not_executed` entries remain part
of the research audit but are not exported as formal results.

The audited Phase 2 source commit is `bb20447`. The published personal research-extension commit is
`33e62d8e8548b2690e6b1acb9fad9776cd94923a`. This proposed extension is based on the latest `upstream/main` at
the time of preparation, but the experiments were completed on the historical research branch. They were not
rerun on the current `upstream/main`. The public tables are filtered from the audited Phase 2 result and
seed-summary CSV files without changing the recorded metrics.

## Protocol and validity gate

Brain Tumor used 40 epochs, `imgsz=640`, physical batch 16, and the full training split. VisDrone used 30 epochs,
`imgsz=768`, and `fraction=0.2`. Clean formal VisDrone baselines used physical batch 4. An earlier batch-8 request
that recovered from OOM at batch 4 was retained as a diagnostic run and excluded from formal aggregation.

A formal result required all of the following:

1. process exit code 0;
2. readable `results.csv` and `args.yaml`;
3. both `best.pt` and `last.pt`;
4. an independent log and run manifest;
5. no NaN, Inf, non-finite gradient, or recovery event;
6. no unexplained all-zero metric collapse; and
7. a configuration consistent with the comparison protocol.

Queue execution and formal validity are distinct states. The Phase 2 queue executed 26 jobs, reporting 15
completed and 11 failed executions. Those counts do not equal the number of formal rows because diagnostics,
recovered attempts, duplicate references, and implementation failures are evaluated separately by the validity
gate.

## Stability diagnostics

In the original AMP path, the first localized non-finite gradient was observed at
`model.base_model.model.4.conv.lora_A.default.weight`, while the logged box, classification, and DFL losses at
that step remained finite. This is evidence about the first observed failure in this protocol, not a claim that
all AMP or LoRA stability problems share the same cause.

Disabling AMP removed the observed non-finite-gradient behavior. In a Brain Tumor diagnostic comparison under
FP32, reducing the Adapter learning-rate multiplier from 1.0 to 0.1 changed diagnostic mAP50-95 from 0.02245 to
0.06630. The stable policy used a 1.0x Detection Head rate, approximately 0.5x Router rate, and 0.1x Adapter rate.
These are protocol-specific observations, not universal optimal settings.

The attempted AMP-safe LoRA implementation failed before valid training because its required execution interface
was incomplete. It is classified as `implementation_failed / not_validated` and is not included in the formal
CSV, seed summary, Pareto front, or accuracy claims.

## Audited results

### Multi-seed results

Values are mean plus or minus sample standard deviation. The best value is the best valid single seed.

| Dataset | Method | n | Precision | Recall | mAP50 | mAP50-95 | Best mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Brain Tumor | Last Stage + Neck + Head | 3 | 0.44358 ± 0.02798 | 0.78865 ± 0.02487 | 0.54427 ± 0.02979 | **0.39172 ± 0.02316** | **0.40944** |
| VisDrone | Full fine-tuning, actual batch 4 | 3 | 0.38211 ± 0.00387 | 0.29718 ± 0.00420 | 0.27550 ± 0.00294 | **0.15342 ± 0.00221** | **0.15527** |

### Representative method comparison

The table uses valid seed-0 runs so that methods with only one seed are not presented as multi-seed estimates.

| Dataset | Method | mAP50-95 | Trainable parameters | Peak GPU memory (GiB) | Train time (s) |
|---|---|---:|---:|---:|---:|
| Brain Tumor | Head-only | 0.36997 | 347,718 | 6.21 | 306.462 |
| Brain Tumor | Neck + Head | 0.34770 | 901,574 | 6.47 | 278.315 |
| Brain Tumor | Last Stage + Neck + Head | **0.40021** | 2,114,633 | 6.51 | 385.617 |
| Brain Tumor | Full fine-tuning | 0.37121 | 2,662,546 | 9.44 | 537.624 |
| Brain Tumor | Stable LoRA | 0.06395 | 409,174 | 7.46 | 325.142 |
| Brain Tumor | Partial fine-tuning + LoRA | 0.35319 | 965,574 | 7.47 | 486.725 |
| VisDrone | Neck + Head | 0.14359 | 903,134 | 22.50 | 1705.060 |
| VisDrone | Last Stage + Neck + Head | 0.14988 | 2,116,193 | 22.50 | 1745.970 |
| VisDrone | Full fine-tuning | **0.15527** | 2,664,106 | 23.00 | 1912.460 |

Brain Tumor favored training the detection head and later backbone layers. Head-only retained 0.36997 mAP50-95
with 347,718 trainable parameters, while Last Stage + Neck + Head produced the best absolute result. VisDrone
favored broader adaptation, with Full fine-tuning producing the best valid result.

Stable LoRA denotes numerical stability, not highest accuracy. It did not provide an absolute accuracy advantage
on Brain Tumor. On VisDrone, the strict gate rejected the available LoRA runs, so the evidence does not support an
accuracy-advantage claim and should not be read as a complete formal head-to-head result.

## Audit corrections

The final audit corrected several result-management hazards:

- trainable parameters were recovered from initialization evidence rather than a validation-time "0 gradients"
  state;
- formal Stable LoRA runs were distinguished from diagnostics;
- a batch-8 VisDrone request that actually ran at batch 4 after OOM was not aggregated with clean batch-4 runs;
- repeated Head-only queue references were marked `not_executed` rather than described as independent OOM runs;
- each aggregation group contains at most one valid run per seed; and
- missing evidence remains `unknown` or `evidence_missing` rather than being replaced by zero.

Pareto membership is computed only after formal-validity gating. The audit maximizes mAP50-95 while minimizing
trainable parameters, peak GPU memory, and training time. A Pareto point is a trade-off, not automatically the
recommended model. The validator shipped with this extension recomputes this front from the exported formal rows
and independently verifies seed uniqueness and sample statistics.

Semantic completion was also stricter than process completion: final artifacts were considered complete only
after the queue ended, CSV files parsed, reports and figures were generated, an archive and SHA256 existed, and no
fatal integrity error remained.

## Version and interpretation boundary

- The experiments belong to the research branch used at execution time.
- The experiment commits included upstream fixes #124 and #125, but did not include the later #170 and #177.
- The LoRA configurations requested `lora_backend=auto`, enabled RS-LoRA, and resolved to the PEFT backend rather
  than fallback in the formal rank runs.
- Although #170 and #177 primarily repair fallback RS-LoRA behavior, these results must not be interpreted as the
  final performance of RS-LoRA on the current repaired `upstream/main`.
- VisDrone LoRA used `fraction=0.2`.
- Several methods have only one valid seed.
- AMP-safe LoRA was not validated successfully.

## Reproduce the audit

Run the dependency-free checker from the repository root:

```bash
python examples/lora_examples/validate_yolo_master_issue50_research_extension.py
```

It validates schemas, formal status, batch fields, unique seeds, multi-seed means and sample standard deviations,
and the stability-gated four-objective non-dominated front. It does not start training.
