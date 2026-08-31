# A3 P1 最终交付：ES-MoE / MoT INT8 PTQ 等价量化实验

> 日期：2026-08-31
>
> 状态：**P1 有条件完成（实验闭环完成，部署加速目标未达成）**
>
> 数据：VisDrone train 6,471 张、val 548 张；输入 640×640
>
> 原始结构化证据：[`a3_p1_int8_ptq_evidence_20260831.json`](a3_p1_int8_ptq_evidence_20260831.json)

## 1. 作业版结论

P1 要求“至少完成 ES-MoE 与 MoT 两族 INT8 PTQ 或等价量化实验，记录精度、时延、模型体积和失败算子”。本次已对两族模型完成真实训练/权重选择、FP32 基线、FP32 ONNX 一致性、VisDrone train 静态标定、INT8 ONNX 全量 548 张验证，并保存失败路径。

结论必须同时保留以下两面：

1. **精度与体积目标完成。** 两族 INT8 相对各自 FP32 ONNX 的 mAP50-95 差值均小于 0.5 个百分点，模型体积下降 51%–69%。
2. **部署加速目标未完成。** 当前 CPU 上 ES-MoE INT8 延迟增加 6.46%，MoT INT8 延迟增加 31.17%。本次只能交付“有效等价量化实验”，不能宣称 INT8 加速成功。
3. **当前导出不是真正条件稀疏执行。** FP32/INT8 ONNX 保留输入相关的 Top-K 掩码与重归一化语义，但导出图仍计算全部专家，再用掩码组合结果；它不是“只执行被选专家”的动态部署。
4. **后续暂停继续扩大 INT8。** 若验收必须保留真正动态专家执行，应先研发自定义动态导出/运行时，再在该运行时上重新做 INT8；不再把 masked-dense 当作动态稀疏加速。

## 2. 统一评估口径

- 训练质量使用 Ultralytics 标准验证器报告。
- 导出与量化差值使用同一套 VisDrone 预处理、`conf=0.001`、相同 NMS 和同一 548 张 val 图。
- FP32 ONNX 与 INT8 ONNX 的时延均固定在 ONNX Runtime `CPUExecutionProvider` 下比较。
- 量化校准使用 VisDrone train 固定随机种子 `42` 抽取 300 张图，不使用 COCO8。
- 标准 PyTorch mAP 与自定义部署评估器 mAP 只分别用于“训练质量”和“部署后端差值”，不交叉相减。

## 3. ES-MoE 实验结果

训练权重标准验证：mAP50=`0.078959`，mAP50-95=`0.040964`。

| 后端 | mAP50 | mAP50-95 | 相对 FP32 差值（百分点） | CPU 延迟 | 大小 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| FP32 ONNX | 0.058628 | 0.033058 | 0 | 63.242 ms | 11.033 MB | 部署基线 |
| INT8 QDQ S8S8 | 0.056872 | 0.031632 | -0.1426 | 128.851 ms | 3.620 MB | 精度过门，但明显更慢 |
| **INT8 QOperator U8S8** | **0.058503** | **0.032613** | **-0.0445** | **67.329 ms** | **3.374 MB** | **最终选择** |

最终 ES-MoE 产物相对 FP32：

- mAP50-95 下降 `0.0445` 个百分点；
- 体积下降约 `69.42%`；
- CPU 延迟增加约 `6.46%`，没有加速。

## 4. MoT 实验结果

MoT v0.10-N 在 VisDrone 完成 50 轮真实训练。控制台标准验证（四舍五入显示）：mAP50≈`0.0678`，mAP50-95≈`0.0337`；模型 4,013,099 参数，约 10.9 GFLOPs。

FP32 导出使用 `export_masked=True`，ONNX 图中保留输入相关 Top-K 掩码。全量一致性验证：自定义 PyTorch mAP50-95=`0.030617`，FP32 ONNX=`0.030589`，差值仅 `-0.0028` 个百分点。

| 后端 | mAP50 | mAP50-95 | 相对 FP32 差值（百分点） | CPU 延迟 | 大小 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| FP32 ONNX | 0.055107 | 0.030589 | 0 | 143.421 ms | 17.493 MB | 部署基线 |
| **INT8 QDQ S8S8** | **0.056099** | **0.030532** | **-0.0057** | **188.120 ms** | **8.510 MB** | **最终选择** |

最终 MoT 产物相对 FP32：

- mAP50-95 下降 `0.0057` 个百分点；
- 体积下降 `51.35%`；
- CPU 延迟增加 `31.17%`，没有加速；
- 属于 **Conv INT8 + TopK/Softmax/MatMul/注意力/路由控制 FP32** 的混合精度 PTQ，不是全模型纯 INT8。

## 5. 失败算子与兼容性证据

| 模型族 / 路径 | 真实失败 | 处理结论 |
|---|---|---|
| ES-MoE 原生 MNN PTQ（KL/EMA） | `std::length_error: cannot create std::vector larger than max_size()`，返回码 `-6` | 记录为 MNN 量化工具与 TopK/mask 图兼容性失败 |
| ES-MoE ORT INT8 → MNN | `ONNX::QLinearConv` 不受 MNNConverter 支持 | ORT INT8 只能作为等价量化实验，不能冒充 MNN INT8 |
| MoT QOperator U8S8 | ORT quantizer 处理非标准 bias 时触发 `AttributeError: 'NoneType' object has no attribute 'data_type'` | 保存失败；改用可运行的 QDQ S8S8 |
| MoT QDQ S8S8 | 成功；QDQ 图含 431 个 `QuantizeLinear`、804 个 `DequantizeLinear`（云端控制台记录） | 进入 548 张全量验证并通过 |

警告 `Please consider to run pre-processing before quantization` 和 `Expected bias ... to be an initializer` 被如实保留；它们没有阻止 QDQ 产物生成、ONNX checker 和 ORT 运行验证。

## 6. 动态路由的声明边界

当前 masked Top-K 导出解决的是**数值语义对齐**：

```text
输入 → Router → TopK（随输入变化）→ 掩码/归一化
                            ↓
             所有专家均计算 → 按掩码组合
```

因此可以声称：

- 路由决策仍然是输入相关的；
- 导出图与 eager Top-K 权重语义一致；
- PyTorch / ONNX 全量 mAP 差值已验证。

不能声称：

- ONNX/INT8 只执行被选中的 K 个专家；
- 当前 INT8 已获得真实稀疏计算收益；
- 当前 CPU 时延优于 FP32。

仓库已有的可行性分析见 [`mot_export_parity_feasibility_20260823.md`](mot_export_parity_feasibility_20260823.md)。

## 7. P1 验收映射

| 验收项 | ES-MoE | MoT | 状态 |
|---|---|---|---|
| 真实 FP32 基线 | 有 | 有 | 完成 |
| FP32 ONNX 一致性 | 通过 | 通过（-0.0028 个百分点） | 完成 |
| INT8 PTQ / 等价量化 | ORT QOperator U8S8 | ORT QDQ S8S8 混合精度 | 完成 |
| 548 张精度 | 有 | 有 | 完成 |
| 同 CPU 延迟 | 有 | 有 | 完成，但均无加速 |
| 模型体积 | 有 | 有 | 完成 |
| 失败算子/工具 | MNN length_error、QLinearConv | QOperator bias data_type | 完成 |
| 真正条件动态专家执行 | 无 | 无 | **未完成，转后续研发** |

## 8. GitHub 证据与本地二进制归档

本次 GitHub 提交包含：

- 本报告与结构化 evidence JSON；
- 云端原始指标、门禁、导出清单、校准清单和失败清单，见
  [`evidence/a3_p1_20260831/`](evidence/a3_p1_20260831/)；
- AP 评估排序修复、NumPy 2.x 积分兼容、输出布局识别和零基线拒绝逻辑；
- ES-MoE masked-dense 补丁的 Top-K 权重重归一化与 `_sparse_forward` 保留修复。

云端产物已于 2026-08-31 下载并完成逐文件校验：

```text
ZIP:       D:\YOLO_Master\P1_cloud_artifacts_20260831\A3_P1_cloud_artifacts_20260831.zip
解压目录: D:\YOLO_Master\P1_cloud_artifacts_20260831\bundle
文件数量: 37
文件总量: 80,585,611 bytes
ZIP SHA256: FE36DF96F7F1A0E187BF5081DE04A6A4A36830DDDC7FE3FC60AB0E9B8379A6CC
逐文件校验: 37/37 通过，missing=0，hash mismatch=0
```

核心二进制哈希如下；完整 37 项清单见
[`bundle_manifest.json`](evidence/a3_p1_20260831/bundle_manifest.json)。

| 产物 | 大小 | SHA256 |
|---|---:|---|
| ES-MoE FP32 ONNX | 11.033 MB | `c5b25ae666c352eb050aa29b9897f8265bbb17a371e65a85aa4acc1127469397` |
| ES-MoE INT8 QOperator U8S8 ONNX | 3.374 MB | `c2308774af0121154921ea211c89bc26b2f495c2fbb81d0aeae3aba64f1721f9` |
| MoT 量化源权重 | 8.322 MB | `5f6ff684f74c773de5cdf0a2e4a51773317829bbeccc48c5c13f0ed3a2f3d417` |
| MoT FP32 ONNX | 17.493 MB | `ef8344b88801bd7216dd9bfa75f7b65d0d114671694bb2041372bc9a46a6a023` |
| MoT INT8 QDQ S8S8 ONNX | 8.510 MB | `463ce6aeb9b573cd46d5cfdc9646d7fd96d75aba5768d433c408e78f8cebbb73` |

ONNX/PT 二进制保存在 Git 仓库外的本地归档目录，防止污染 Git 历史；GitHub 提交保留可审查的小型原始证据。若必须在线分发模型，应将 ZIP 或模型上传为 GitHub Release 附件，而不是直接提交到普通 Git 历史。

## 9. 后续方向：暂停 INT8，研发真正动态导出

若下一阶段必须保留“仅执行被选专家”的真实动态计算，建议冻结当前 P1 产物，不继续在 masked-dense 图上堆叠 QAT/更多 INT8 组合，改做以下 PoC：

1. **建立动态执行契约**：输出不只要求数值一致，还必须证明每个 token/sample 实际只调用 K 个专家，而不是 E 个专家。
2. **实现自定义运行时节点**：优先评估 TensorRT `IPluginV3` 或 ORT Custom Op，将 Router TopK、token gather、专家 dispatch、scatter/reduce 封装成可执行节点。
3. **先做 FP32/FP16 动态 PoC**：在自定义运行时验证输出、实际 expert calls、kernel 时间和动态 shape，再恢复 INT8。
4. **再做混合 INT8**：Router 与 TopK 保 FP16/FP32，专家 Conv/MatMul 按敏感度量化；记录 FP32/INT8 Top-K 一致率。
5. **验收门槛**：mAP 差值 <0.5 个百分点、实际 expert calls 接近 K/E、端到端时延优于 masked-dense，并提交失败算子和可复现引擎构建日志。

在这些门槛完成前，当前结果应被描述为“动态路由决策 + masked-dense 执行的等价量化实验”，而不是“动态稀疏 INT8 部署完成”。
