# Export Capability Matrix

> Generated from `ultralytics/cfg/export-capability-matrix.yaml`. Do not edit manually.

## Formats

| Format | Supported | Default strategy | Known limitation |
|---|---:|---|---|
| `pytorch` | yes | `dynamic` | - |
| `torchscript` | yes | `dense_fallback` | trace uses shape-specific dense fallback |
| `onnx` | yes | `dense_fallback` | data-dependent routing is replaced by dense fallback |
| `openvino` | yes | `dense_fallback` | inherits ONNX dense fallback limitations |
| `engine` | yes | `dense_fallback` | TensorRT verification is backend and hardware specific |
| `coreml` | yes | `dense_fallback` | dynamic sparse dispatch is not preserved |
| `saved_model` | yes | `dense_fallback` | dynamic sparse dispatch is not preserved |
| `pb` | yes | `dense_fallback` | dynamic sparse dispatch is not preserved |
| `edgetpu` | yes | `dense_fallback` | Edge TPU support requires downstream operator validation |
| `paddle` | yes | `dense_fallback` | component roundtrip is unverified |
| `mnn` | yes | `dense_fallback` | component roundtrip is unverified |
| `ncnn` | yes | `dense_fallback` | component roundtrip is unverified |
| `imx` | yes | `dense_fallback` | hardware-specific validation is required |
| `rknn` | yes | `dense_fallback` | hardware-specific validation is required |
| `executorch` | yes | `dense_fallback` | component roundtrip is unverified |
| `axelera` | no | `refuse` | routed module export has not been validated for Axelera |
| `deepx` | no | `refuse` | routed module export has not been validated for DEEPX |
| `qnn` | no | `refuse` | routed module export has not been validated for Qualcomm QNN |
| `litert` | yes | `dense_fallback` | dynamic sparse dispatch is not preserved |
| `hailo` | no | `refuse` | routed module export has not been validated for Hailo |

## Routed Modules

| Module family | Supported | Dense fallback | Requires merge | Known limitation |
|---|---:|---:|---:|---|
| `MoA` | yes | yes | no | - |
| `MoE` | yes | yes | no | - |
| `MoLoRA` | yes | yes | no | dynamic router cannot be represented as an exact static merge |
| `MoT` | yes | yes | no | Deformable expert uses grid_sample and requires backend operator support |

## Effective Policies

The effective policy intersects each format default with the module policy. Runtime preflight may refuse a declared dense fallback when a concrete module does not advertise a safe implementation.

| Module family | Format | Effective strategy | Dense fallback | Requires merge | Known limitation |
|---|---|---|---:|---:|---|
| `MoA` | `pytorch` | `dynamic` | yes | no | - |
| `MoA` | `torchscript` | `dense_fallback` | yes | no | trace uses shape-specific dense fallback |
| `MoA` | `onnx` | `dense_fallback` | yes | no | data-dependent routing is replaced by dense fallback |
| `MoA` | `openvino` | `dense_fallback` | yes | no | inherits ONNX dense fallback limitations |
| `MoA` | `engine` | `dense_fallback` | yes | no | TensorRT verification is backend and hardware specific |
| `MoA` | `coreml` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoA` | `saved_model` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoA` | `pb` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoA` | `edgetpu` | `dense_fallback` | yes | no | Edge TPU support requires downstream operator validation |
| `MoA` | `paddle` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoA` | `mnn` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoA` | `ncnn` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoA` | `imx` | `dense_fallback` | yes | no | hardware-specific validation is required |
| `MoA` | `rknn` | `dense_fallback` | yes | no | hardware-specific validation is required |
| `MoA` | `executorch` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoA` | `axelera` | `refuse` | no | no | routed module export has not been validated for Axelera |
| `MoA` | `deepx` | `refuse` | no | no | routed module export has not been validated for DEEPX |
| `MoA` | `qnn` | `refuse` | no | no | routed module export has not been validated for Qualcomm QNN |
| `MoA` | `litert` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoA` | `hailo` | `refuse` | no | no | routed module export has not been validated for Hailo |
| `MoE` | `pytorch` | `dynamic` | yes | no | - |
| `MoE` | `torchscript` | `dense_fallback` | yes | no | trace uses shape-specific dense fallback |
| `MoE` | `onnx` | `dense_fallback` | yes | no | data-dependent routing is replaced by dense fallback |
| `MoE` | `openvino` | `dense_fallback` | yes | no | inherits ONNX dense fallback limitations |
| `MoE` | `engine` | `dense_fallback` | yes | no | TensorRT verification is backend and hardware specific |
| `MoE` | `coreml` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoE` | `saved_model` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoE` | `pb` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoE` | `edgetpu` | `dense_fallback` | yes | no | Edge TPU support requires downstream operator validation |
| `MoE` | `paddle` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoE` | `mnn` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoE` | `ncnn` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoE` | `imx` | `dense_fallback` | yes | no | hardware-specific validation is required |
| `MoE` | `rknn` | `dense_fallback` | yes | no | hardware-specific validation is required |
| `MoE` | `executorch` | `dense_fallback` | yes | no | component roundtrip is unverified |
| `MoE` | `axelera` | `refuse` | no | no | routed module export has not been validated for Axelera |
| `MoE` | `deepx` | `refuse` | no | no | routed module export has not been validated for DEEPX |
| `MoE` | `qnn` | `refuse` | no | no | routed module export has not been validated for Qualcomm QNN |
| `MoE` | `litert` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved |
| `MoE` | `hailo` | `refuse` | no | no | routed module export has not been validated for Hailo |
| `MoLoRA` | `pytorch` | `dynamic` | yes | no | dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `torchscript` | `dense_fallback` | yes | no | trace uses shape-specific dense fallback; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `onnx` | `dense_fallback` | yes | no | data-dependent routing is replaced by dense fallback; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `openvino` | `dense_fallback` | yes | no | inherits ONNX dense fallback limitations; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `engine` | `dense_fallback` | yes | no | TensorRT verification is backend and hardware specific; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `coreml` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `saved_model` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `pb` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `edgetpu` | `dense_fallback` | yes | no | Edge TPU support requires downstream operator validation; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `paddle` | `dense_fallback` | yes | no | component roundtrip is unverified; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `mnn` | `dense_fallback` | yes | no | component roundtrip is unverified; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `ncnn` | `dense_fallback` | yes | no | component roundtrip is unverified; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `imx` | `dense_fallback` | yes | no | hardware-specific validation is required; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `rknn` | `dense_fallback` | yes | no | hardware-specific validation is required; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `executorch` | `dense_fallback` | yes | no | component roundtrip is unverified; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `axelera` | `refuse` | no | no | routed module export has not been validated for Axelera; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `deepx` | `refuse` | no | no | routed module export has not been validated for DEEPX; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `qnn` | `refuse` | no | no | routed module export has not been validated for Qualcomm QNN; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `litert` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; dynamic router cannot be represented as an exact static merge |
| `MoLoRA` | `hailo` | `refuse` | no | no | routed module export has not been validated for Hailo; dynamic router cannot be represented as an exact static merge |
| `MoT` | `pytorch` | `dynamic` | yes | no | Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `torchscript` | `dense_fallback` | yes | no | trace uses shape-specific dense fallback; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `onnx` | `dense_fallback` | yes | no | data-dependent routing is replaced by dense fallback; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `openvino` | `dense_fallback` | yes | no | inherits ONNX dense fallback limitations; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `engine` | `dense_fallback` | yes | no | TensorRT verification is backend and hardware specific; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `coreml` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `saved_model` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `pb` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `edgetpu` | `dense_fallback` | yes | no | Edge TPU support requires downstream operator validation; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `paddle` | `dense_fallback` | yes | no | component roundtrip is unverified; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `mnn` | `dense_fallback` | yes | no | component roundtrip is unverified; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `ncnn` | `dense_fallback` | yes | no | component roundtrip is unverified; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `imx` | `dense_fallback` | yes | no | hardware-specific validation is required; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `rknn` | `dense_fallback` | yes | no | hardware-specific validation is required; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `executorch` | `dense_fallback` | yes | no | component roundtrip is unverified; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `axelera` | `refuse` | no | no | routed module export has not been validated for Axelera; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `deepx` | `refuse` | no | no | routed module export has not been validated for DEEPX; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `qnn` | `refuse` | no | no | routed module export has not been validated for Qualcomm QNN; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `litert` | `dense_fallback` | yes | no | dynamic sparse dispatch is not preserved; Deformable expert uses grid_sample and requires backend operator support |
| `MoT` | `hailo` | `refuse` | no | no | routed module export has not been validated for Hailo; Deformable expert uses grid_sample and requires backend operator support |

## Evidence Boundary

A `supported` entry means preflight has a declared execution strategy. It does not imply full-model or hardware-specific numerical verification. Consult `model-registry.yaml` for executable evidence status.
