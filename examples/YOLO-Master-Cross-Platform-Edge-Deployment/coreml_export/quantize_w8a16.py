#!/usr/bin/env python3
"""W8A16 for an existing .mlpackage: int8 per-channel linear-symmetric weights, fp16 activations.

    python coreml_export/quantize_w8a16.py yolov12x.mlpackage yolov12x-w8a16.mlpackage

The recipe scripts/export_coreml_p03.py uses for the Project03 packages, applied to any ML program
(runs on Linux: weight-only, no calibration data). The dequantize happens on the compute unit at
load / run time (`constexpr_affine_dequantize`), so the package halves on disk and in memory; the
arithmetic stays fp16, which is why the accuracy pass usually moves by less than 0.001 mAP50-95.
"""
import sys

import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights


def main(src: str, dst: str) -> None:
    m = ct.models.MLModel(src, skip_model_load=True)
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                                   granularity="per_channel"))
    q = linear_quantize_weights(m, config=cfg)
    for k, v in m.user_defined_metadata.items():
        q.user_defined_metadata[k] = v
    q.user_defined_metadata["precision"] = "w8a16"
    q.user_defined_metadata["quantization"] = "int8 weights per-channel linear_symmetric, fp16 activations"
    q.save(dst)
    print(f"OK  {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    main(sys.argv[1], sys.argv[2])
