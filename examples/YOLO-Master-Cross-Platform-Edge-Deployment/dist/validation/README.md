# v1.2.0 Linux validation receipts

- `a100-bundle-acceptance-1.2.0.log` / `.tar.gz`: the six prebuilt bundles on a plain A100 80GB pod
  (driver 570.172, Ubuntu 22.04, no cuDNN / TensorRT installed): 65 checks, 0 failures
  (`test_bundles_a100.sh` is the script). The tarball holds every bench JSON, the sustained JSON,
  the tracking txt dumps, the COCO label export, the two TensorRT engines compiled on the A100
  (sm80), one annotated frame and one tracked clip.
- `l40s-v120-cuda-receipts.tar.gz`: the L40S pod where the bundles were built: bootstrap / build /
  packaging logs, `validate_gpu.sh` log (CUDA preprocessing parity, TRT and ORT-CUDA parity, bench,
  server suites), the post-validation timings and accuracy, the v0.1-N TensorRT fp16 engine (sm89),
  and the pod notes (`README-v120-cuda.md`).

Bundles and checksums: `../SHA256SUMS-linux-1.2.0.txt`. Built from branch `dist/v1.2.0-linux`
(= main at the Linux phase + the packaging fixes), `scripts/package_linux.sh <variant> 1.2.0`.
