# A3 真动态专家导出/运行时 PoC（2026-09-01）

## 结论

暂停继续扩展 masked-dense INT8 路径，改为研发真正的条件专家执行。第一阶段块级 PoC 已通过：

- 路由器、每个专家和后处理分别导出为 ONNX。
- 宿主运行时先执行路由器，只为被选中的样本/专家对调用专家会话。
- 未选专家不会创建 ONNX Runtime 会话，也不会执行前向。
- 清单明确设置 `masked_dense_allowed=false`，不允许把乘零掩码的全专家执行声明为动态稀疏。
- 每次运行产生专家调用审计；若空间路由的专家并集覆盖全部专家，会明确标记无缩减，并可用 `require_reduction=True` 拒绝该次调用。

本阶段已经从模块级 PoC 推进到“完整 YOLO 外图 + ORT 路由 + checkpoint PyTorch 专家”的混合精度验证路径，
但还不是完整 YOLO 网络的可部署动态运行时。

当前三项硬限制：

1. 路由块仍是分离导出；完整 YOLO 可混合执行，但还没有导出为可部署的统一动态 DAG。
2. 已增加真实 MoT checkpoint + 单张 VisDrone 的块级证据，但尚未完成 548 张全量端到端证据。
3. 当前为 ORT FP32 主机调度，不是 TensorRT，也没有端到端 GPU 加速数据。

## 导出与执行结构

```text
input
  -> router.onnx
  -> host Top-K dispatcher
       -> expert_0.onnx（仅在被选中时执行）
       -> expert_1.onnx（仅在被选中时执行）
       -> ...
  -> weighted accumulation
  -> postprocess.onnx
  -> output
```

动态包包含 `dynamic_bundle.json`，记录执行语义、路由粒度、专家数、Top-K、输入约束以及每个 ONNX 文件的大小和 SHA256。

## 验证结果

验证命令：

```powershell
$env:YOLO_CONFIG_DIR = 'D:\YOLO_Master\YOLO_Master\.tmp_dynamic_runtime_validation'
& 'D:\Anaconda3\envs\yolo_env\python.exe' scripts\validate_dynamic_expert_runtime.py `
  --output reports\a3_dynamic_expert_runtime_poc_20260901.json
```

结果文件：`reports/a3_dynamic_expert_runtime_poc_20260901.json`

| 检查项 | 结果 |
|---|---:|
| 纯 NumPy 调度器未选专家调用 | 0 次 |
| dense 路由冒充 Top-K | 已拒绝 |
| 空间路由全专家并集冒充缩减 | 已拒绝 |
| ES-MoE eager sparse 与 split-ORT 最大绝对误差 | `5.9604645e-08` |
| ES-MoE 实际加载专家 | `[0] / 3` |
| ES-MoE 样本/专家对缩减率 | `66.67%` |
| MoT 样本级路由 eager sparse 与 split-ORT 最大绝对误差 | `2.3841858e-07` |
| MoT 实际加载专家 | `[0] / 3` |
| MoT 样本/专家对缩减率 | `66.67%` |
| ORT-router + PyTorch 专家输出误差 | `0` |
| ORT-router 混合适配器加载 ONNX 专家 | `0` |

验证状态：`PASS`。

## 真实 MoT checkpoint + VisDrone 单图结果

真实 checkpoint：`best_for_quantization.pt`，共捕获并导出 6 个 MoT 路由块。输入图片为
`0000364_01765_d_0000782.jpg`，原始尺寸 `540x960`，SHA256：
`451961d5e3fd86f3778f2c59204de3f20f5601d53071f9c4da94fb02390e4c9a`。

模型级清单：
`D:/YOLO_Master/P1_dynamic_runtime_20260901/mot_v10_visdrone_image_blocks/dynamic_model_blocks.json`

总体状态为 `VALIDATED_WITH_ROUTE_DRIFT`：所有专家与后处理子图在部署路由下通过，最大绝对误差不超过
`1.4305115e-06`；但 `model.14.m.0` 的 1600 个空间路由位置中有 1 个位置在 ORT 与 eager 间选择不同，
位置漂移率为 `0.0625%`。该位置处于近似并列边界，因此不能把它写成严格路由一致。

| 块 | Top-K | 部署路径最大误差 | 路由位置漂移 | 本次加载专家 | 样本/专家对缩减 |
|---|---:|---:|---:|---:|---:|
| `model.14.m.0` | 2/3 | `1.43e-06` | 1/1600 | 0,1,2 | 0% |
| `model.14.m.1` | 2/3 | `9.54e-07` | 0/1600 | 1,2 | 33.33% |
| `model.20.m.0` | 2/3 | `1.43e-06` | 0/1600 | 0,1,2 | 0% |
| `model.20.m.1` | 2/3 | `9.54e-07` | 0/1600 | 1,2 | 33.33% |
| `model.23.m.0` | 1/3 | `9.54e-07` | 0/400 | 1 | 66.67% |
| `model.23.m.1` | 1/3 | `9.54e-07` | 0/400 | 0 | 66.67% |

路由器 ONNX 输出 dense 概率，但 dense 概率不触发专家执行。主机根据
`probability - expert_id * 1e-6` 排序并形成稀疏 Top-K，随后只调用专家并集。该规则只稳定近似并列的选择，
原始概率仍用于混合权重。若使用 `--require-exact-route`，上述单图会因 1 个漂移位置而严格失败。

## 完整 YOLO 混合执行结果

已将真实 checkpoint 中的 6 个 MoTBlock 替换为 split-ORT 动态适配器，并在同一张 VisDrone 图片上完成完整
YOLO 前向。共享干线和检测头仍是 PyTorch，因此这是完整模型混合执行，不是完整模型导出。

结果文件：
`D:/YOLO_Master/P1_dynamic_runtime_20260901/full_yolo_hybrid_visdrone_image.json`

- 输出形状：`[1, 14, 8400]`。
- 6/6 动态块均实际执行并产生专家调用审计。
- 两个 Top-2 空间路由块的专家并集覆盖 3/3，未获得专家计算缩减。
- 两个 Top-2 块调用 2/3 专家，样本/专家对缩减 `33.33%`。
- 两个 Top-1 块调用 1/3 专家，样本/专家对缩减 `66.67%`。
- 完整输出相对 eager 最大绝对差 `0.4237366`，由已记录的近似并列路由漂移向后传播，因此不通过精度门禁。
- 单次未预热 CPU 诊断时间约 eager `923 ms`、混合桥接 `15718 ms`；桥接包含多次 NumPy 拷贝，明确不能作为性能结果。

该结果应标记为 `EXECUTION_PASS_WITH_ROUTE_DRIFT`，`accuracy_gate_passed=false`。只有 548 张 VisDrone mAP
验证能够判断这类极少量路由漂移对最终精度是否可接受。

## 面向云端全量 mAP 的 GPU 验证适配器

已新增 `ORTRouterTorchExpertAdapter`。它与上一节的全 split-ORT 桥接严格区分：

- `router.onnx` 在 ORT CPU 上产生 dense 概率，宿主形成稀疏 Top-K。
- ONNX 专家会话加载数保持为 0。
- 原 checkpoint 的 MoT 专家、投影、归一化和残差留在 PyTorch 设备上，按稀疏路由实际调用。
- 可选地同时运行 eager router，累计记录每个路由位置的选择漂移。
- 该路径用于完整 YOLO mAP 与真实专家调用验证；CPU/GPU 同步开销使其不能作为部署延迟结果。

真实 checkpoint + VisDrone 单图完整前向已经通过，结果文件：
`D:/YOLO_Master/P1_dynamic_runtime_20260901/full_yolo_router_torch_visdrone_image.json`。

- 输出形状 `[1, 14, 8400]`，6/6 动态块均执行。
- 6 个块的 ONNX 专家加载数均为 0，checkpoint PyTorch 专家调用数均大于 0。
- 专家调用/缩减统计与全 split-ORT 路径一致。
- 已知的 `model.14.m.0` 漂移仍为 1/1600；完整输出差值仍为 `0.4237366`，未隐藏或修正该差异。
- 单次未预热 CPU 诊断为 eager `343.65 ms`、混合 `4047.92 ms`，明确不作为加速数据。

全量门禁入口为 `scripts/validate_dynamic_mot_full_val.py`。它在完全相同的固定 640x640（`rect=False`）
预处理下依次评估 eager checkpoint 和动态混合版本，并记录：548 张 mAP、精度百分点差、逐块专家真实调用、
样本/专家对缩减、路由漂移以及 ONNX 专家加载数。默认精度门槛为绝对差 `<= 0.5` 个百分点。

已使用 1 张真实 VisDrone 图片对该完整验证入口做集成冒烟：eager 与动态版本的 mAP50-95 均为
`0.05094136`，差值为 `0`；逐块适配器均执行且 ONNX 专家加载数为 0。该结果只证明验证器接线正确，
明确不作为 548 张精度证据。验证脚本会在 `on_val_start` 清零框架 warmup 调用，最终专家统计只覆盖真实
validation 批次。

## 真实性边界

1. 当前 split-ORT 运行时是真实条件调用，不是静态 masked-dense 图。
2. 当前 MoT 训练配置默认使用空间路由。虽然每个位置只选 Top-K，但一个样本内所有位置的专家并集可能覆盖全部专家；此时现有专家结构仍需对该样本执行全部专家，不能声称获得专家计算缩减。
3. 若要求 MoT 每个样本都严格只执行 K 个专家，需训练/微调 `use_spatial_router=False` 的样本级路由版本。
4. 若必须原样保留空间级动态路由语义并获得真实算力缩减，则下一阶段需要 token/window 级 gather-dispatch-scatter，以及 CUDA/TensorRT/MNN 自定义算子；块级整图专家调用无法满足这一点。
5. 当前空间尺寸固定为导出样本尺寸，batch 维动态；尚未覆盖完整 YOLO 图的前后分段和多层路由调度。

## 下一阶段

1. 在真实 ES-MoE/MoT checkpoint 上捕获每个路由块的输入规格并导出 split bundle。
2. 将完整 YOLO 图切成共享干线段、路由块和检测头段，构建统一 DAG 调度器。
3. 对真实 VisDrone validation 输入记录每层专家选择、实际调用、延迟和显存。
4. ES-MoE 先按样本级动态执行落地；MoT 同时评估样本级路由微调和空间级自定义 CUDA dispatcher 两条路线。
5. 在动态 FP32 运行时稳定并证明真实加速前，不继续扩展 INT8，也不发布动态 INT8 结论。

## 云端触发条件

以下工作开始前需要云端 GPU/TensorRT 环境，届时应明确通知并提供一键命令：

1. 完整 YOLO 动态 DAG 在 548 张 VisDrone val 上的 mAP 与路由漂移统计。
2. TensorRT 自定义 Top-K/dispatch/gather-scatter 插件编译和正确性验证。
3. FP32 动态运行时端到端 GPU 延迟、吞吐、显存和专家实际调用统计。

其中第 1 项的本地代码现已就绪，**现在需要云端 GPU 运行验证**。第 2、3 项仍需后续 TensorRT/CUDA
自定义调度实现，不能用当前混合桥接的耗时替代。

云端一键流程（在已有 `/mnt/workspace/YOLO-Master`、VisDrone 和 MoT checkpoint 的环境执行）：

```bash
cd /mnt/workspace/YOLO-Master
export YOLO_CONFIG_DIR=/tmp/yolo_dynamic_settings
mkdir -p "$YOLO_CONFIG_DIR"

CKPT=/mnt/workspace/YOLO-Master/examples/artifacts/mot_v10_visdrone_50e/best_for_quantization.pt
BUNDLES=/mnt/workspace/YOLO-Master/examples/artifacts/mot_v10_dynamic_blocks
IMAGE=/mnt/workspace/visdrone/images/val/0000364_01765_d_0000782.jpg
DATA=/mnt/workspace/YOLO-Master/examples/configs/visdrone_local_for_mot.yaml
RESULT=/mnt/workspace/YOLO-Master/examples/results/mot_v10_dynamic_full_val/DYNAMIC_MOT_FULL_VAL.json

python scripts/export_dynamic_blocks_from_checkpoint.py \
  "$CKPT" "$BUNDLES" \
  --input-image "$IMAGE" --imgsz 640 --batch 1 --device cpu \
  --family mot --validate --overwrite

python scripts/validate_dynamic_mot_full_val.py \
  "$CKPT" "$BUNDLES/dynamic_model_blocks.json" "$DATA" "$RESULT" \
  --device 0 --imgsz 640 --batch 8 --workers 4 \
  --map-tolerance-pct-points 0.5
```

不要在本次已知存在近似并列漂移的模型上默认加入 `--require-exact-route`；该选项是严格研究门禁，会在任何
一个位置漂移时失败。默认流程仍会完整记录漂移，并以最终 548 张 mAP 门禁决定精度是否可接受。
