# Foundation response diagnostics: compact research companion

Can intermediate Foundation features help a small detector, and why might learnable feature matching fail to improve detection? This DINOv3-S → YOLO-Master-N study motivates the response utilities in this PR. It does **not** establish a detection improvement or recommend a new default training strategy.

## Start here

1. [Research map](RESEARCH_MAP.md): question, trajectory and final result in one short entry.
2. [Formal recommendation](DINOV3_P2_FORMAL_RECOMMENDATION.md): bounded conclusions and evidence index.
3. [Frozen P2-07 protocol](DINOV3_P2_NORMALIZED_RESPONSE_FORMAL_PROTOCOL.md): historical rules, not permission to execute them.
4. [B24 recovery disclosure](P2_07_RECOVERY_DISCLOSURE.md): manual interruption, epoch-zero restart and interpretation limits.
5. [Archived final result](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_normalized_response_formal/final_result.md): primary and secondary outcomes.

## What this integration contains

[Response primitives](../../ultralytics/nn/foundation/response.py) provide deterministic paired perturbations, strict FP32 cosine/response losses and BN buffer preservation. They reuse upstream Foundation APIs. No experimental normalized trainer, Teacher assets, data, full logs, queue or probe scripts are included. No default Foundation/Response training behavior is enabled.

Integration Commit 1 is `3a62b6f480e2855c3034f9788f69b804b81a5b56`, based on upstream `3bd0a602be88336523eb9fdebfacdd2e92265756`. It includes **post-freeze public-API hardening** for FP32 overflow, BN name collisions and optional eval-mode admission. Those changes did not regenerate any scientific results and are not retrospectively attributed to the historical training implementation.

## Evidence availability

The original scientific freeze remains local and unchanged:

- Scientific anchor: `e3e11d0cfebcd0893f5cd5bae46cf7f48ba8429a`.
- Packaging anchor: `781a6f148a46d59a89ea40806de8e10eb7b98941`.

These original commits are not published because historical logs contain privacy-sensitive host-local paths.

A selected sanitized derivative is publicly available at [`fedc360e73cdbfeaed930b17365458f632b1abc8`](https://github.com/gao-666/YOLO-Master/commit/fedc360e73cdbfeaed930b17365458f632b1abc8). The public package does not inherit the original research Git history. Protected scientific result/protocol files were copied byte-for-byte; other included records were sanitized only according to the documented privacy rules. See [PROVENANCE.json](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/PROVENANCE.json) for source/public hashes and sanitization metadata.

This is selected evidence, not the complete raw archive. Per-batch JSONL, earlier-stage archives, scripts and figures not included in the package remain local-only references in the historical documents below. The package's README/audit wording describes its pre-publication build stage; anonymous fixed-SHA access was subsequently verified on 2026-09-12. This integration commit records that later publication state without changing the evidence commit.

The compact integration checkout now includes the final four result tables and [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json). Larger logs, epoch traces, recovery records and other supporting evidence remain in the pinned public sanitized evidence package.

## Safe local verification

From the repository root, with project/test dependencies installed:

```text
python -m pytest tests/test_foundation_response.py --override-ini addopts= -q
```

These are CPU synthetic contracts, not a formal experiment; no Teacher weights or dataset download is required. If user settings are inaccessible, point `YOLO_CONFIG_DIR` to a writable temporary directory. Validated on Python 3.11.15 / torch 2.11.0+cu128 using CPU: **27 response tests**, or **68** including upstream projector/protocol/loss contracts. This is not the historical evidence suite's 59-test count or full upstream CI.

The new response implementation and synthetic test are Ruff-clean. Formatting passes for all three Commit 1 files. Unfiltered Ruff reports `I001` and `RUF022` in the initializer; the frozen upstream base reports the same ordering findings. The patch intentionally preserves existing import/export order. Actual upstream CI acceptance remains to be checked; inherited findings can still make CI fail.

## Terminal boundary

P2-07: **No detectable change**, not equivalence. Secondary aggregate Mechanism Support does not override the detection endpoint or establish causal mediation. The route is terminal; no additional loss/alpha/seed search or training is authorized. Full historical reproduction belongs to the evidence checkout and requires separately authorized, licensed assets and frozen environment restoration—not running commands from this compact checkout.
