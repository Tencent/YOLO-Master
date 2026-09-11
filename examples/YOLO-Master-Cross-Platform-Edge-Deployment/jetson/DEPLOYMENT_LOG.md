# Jetson deployment log template

Use this file as the human-readable companion to an evidence manifest. It is a
template, not a record of a run performed in this repository. Replace every
`TBD` field from the target device and archive the unedited command output.

## Platform

| Field | Value |
| --- | --- |
| Device and memory | TBD |
| JetPack / Ubuntu | TBD |
| CUDA / cuDNN / TensorRT | TBD |
| Power mode and clocks | TBD |
| CPU/GPU temperature during benchmark | TBD |
| Runner commit | TBD |
| Compiler and CMake version | TBD |

## Model and engine

| Field | Value |
| --- | --- |
| Checkpoint path and SHA256 | TBD |
| ONNX path and SHA256 | TBD |
| Engine path and SHA256 | TBD |
| Input shape and output shape | TBD |
| Precision recipe | TBD (FP32 / FP16 / calibrated INT8) |
| Calibration list digest and count | TBD / not applicable |

## Benchmark protocol

| Parameter | Value |
| --- | --- |
| Ordered image-list digest | TBD |
| Image count | TBD |
| Input size | TBD |
| Confidence / IoU / max detections | TBD |
| Multi-label decoding | TBD |
| Warm-up runs | TBD |
| Timed runs | TBD |
| Threads / execution provider | TBD |

## Results

Report device-side compute and end-to-end timing separately. Do not fill a
metric cell until the corresponding prediction directory and reference JSON
have been verified by `evidence_manifest.py verify`.

| Engine | Compute mean (ms) | End-to-end P50 (ms) | P95 (ms) | P99 (ms) | FPS | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Calibrated INT8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

An uncalibrated `trtexec --int8` run may be retained as a parser or throughput
diagnostic, but it must be labelled diagnostic and must not be used as an
accuracy result.

## Build observations

Record only observations reproduced on this device and software stack. For
each failure include the exact command, exit status, relevant log excerpt and
the change that resolved it.

| Symptom | Reproduction command | Resolution | Evidence path |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |

## Required attachments

* raw `trtexec` and runner logs;
* model/engine and image-list SHA256 records;
* per-image predictions and the PyTorch/reference metric JSON;
* the completed evidence manifest and its verification output;
* the exact build and benchmark commands.

Without these attachments, this log supports only a procedural description and
not a cross-platform deployment claim.
