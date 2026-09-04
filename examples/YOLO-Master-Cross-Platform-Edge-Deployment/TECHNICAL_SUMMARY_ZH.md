# Issue #51 边缘部署与一致性验证技术总结

## 摘要

本文档给出 YOLO-Master EsMoE-N 在 VisDrone 或 SKU-110K 垂直场景中的
边缘部署验证协议、实现边界与证据要求。文档的目标是使一次实验能够被
第三方在另一台 Linux 或 Windows 机器上复核，而不是仅报告一个不可追溯
的 mAP 或 FPS 数字。

本分支已经提供 ONNX Runtime、NCNN 与 MNN 的导出辅助程序、C++ 推理运行
时、统一的 letterbox 与后处理、mAP 评估器、量化辅助程序以及 CMake 构建
入口。当前仓库不包含 EsMoE-N 权重、VisDrone 数据集、548 张验证图像的
逐图预测文件或目标机器的完整原始日志。因此，本分支目前的结论是
“验证基础设施可复现”，不能据此声称已经完成 Issue #51 所要求的完整
训练、跨后端精度验收或 ARM64 实机部署。

## 面向评审的成果摘要

本提交将 Issue #51 的部署要求落实为一套可执行、可审计的验收流程。成果
与证据边界如下，便于在 Issue 评论或 Pull Request 描述中直接引用：

| 成果项 | 可核对的实现或产物 | 当前证据等级 |
|---|---|---|
| 统一 C++ 推理入口 | ONNX Runtime、NCNN、MNN 后端适配器；同一预处理、解码和 NMS | L0（代码与契约测试） |
| 导出与转换检查 | ONNX checker/简化、NCNN param/bin 配对和 sidecar 名称、MNN 转换提示 | L0（结构检查） |
| 精度验收协议 | 固定有序图像清单、`eval_map.py`、百分点门禁、逐图 prediction diff | L0（工具可运行） |
| INT8 校准协议 | 至少 300 张训练图像、内容 hash 去重、默认敏感节点保留 FP32 | L0（工具可运行） |
| Linux 运行验证 | Ubuntu 22.04 x86_64 上 YOLOv5s ONNX 单图 smoke：6 个检测框 | L1（功能 smoke） |
| EsMoE-N 完整结果 | VisDrone/SKU-110K、至少 500 张验证图、mAP 与双平台日志 | 待目标机器补充 |

这里的 L0/L1/L2--L4 与第 3 节定义一致。表中“工具可运行”只表示接口、
输入校验和错误门禁已经实现；只有同时归档模型、图像清单、预测、日志及
SHA256 后，才可将相应项目升级为正式验收结果。

该实现覆盖 ONNX Runtime、NCNN 与 MNN 的 C++17 推理路径，并将导出检查、
预处理/后处理、精度评估、INT8 校准和证据归档纳入同一协议。现有契约测试
与 Ubuntu 22.04 x86_64 的单图 smoke 证明了基础链路可运行；完整 EsMoE-N
精度、量化和目标设备性能仍须在真实 checkpoint、数据集及硬件上按本协议
复现后报告。

## 1. 问题定义与验收边界

Issue #51 的验收对象是垂直场景下的模型部署闭环，至少应覆盖以下内容：

| 类别 | 最低要求 | 本分支状态 |
|---|---|---|
| 数据与模型 | VisDrone 或 SKU-110K 微调 checkpoint | 需由目标机器补充 |
| 导出 | ONNX 加简化、opset/checker 校验；NCNN 或 MNN 转换 | 脚本已提供 |
| 预处理 | 与训练一致的 RGB、NCHW、宽高比保持 letterbox | 运行时已提供 |
| 后处理 | 类别感知 NMS、可复现阈值、最大检测数 | 运行时已提供 |
| 精度 | 同一有序验证集不少于 500 张，报告 mAP50 与 mAP50-95 | 评估器已提供，实测数据待补 |
| INT8（可选） | 训练图像校准不少于 300 张，验证集严格隔离 | 量化脚本已提供，精度门禁待补实测 |
| 性能 | 固定线程、预热、重复次数，报告均值/P50/P95/P99/FPS | C++ CSV 入口已提供 |
| 平台 | 至少两个目标平台完成 CMake 构建与运行 | 当前仅有 Linux x86_64 smoke 记录 |

“已提供”表示代码路径和输入校验存在；“实测数据待补”表示没有把
第三方报告中的数字写成当前仓库的实验结果。

### 1.1 训练与数据来源记录

实验结果的可复核性依赖训练来源，而不仅是导出文件。正式提交应记录
基础模型或代码提交号、微调数据集版本及划分、类别映射、训练 epoch、
随机种子、最佳 checkpoint 路径和 SHA256，以及导出所用的 Ultralytics、
PyTorch、ONNX Runtime 和转换工具版本。若这些信息缺失，mAP 数字即使能够
复现，也无法确认对应的是同一模型和同一数据划分。本分支没有替代这些
信息的 checkpoint，必须由目标机器补齐。

建议将训练元数据按下表归档，并与 `best.pt` 放在同一证据包中：

| 项目 | 记录内容 |
|---|---|
| 代码与基础模型 | Git commit、模型配置或基础权重版本 |
| 数据 | 数据集发布版本、train/val 划分、类别映射、图像数量 |
| 优化设置 | epoch、batch size、优化器、学习率策略、随机种子、deterministic 设置 |
| 软件环境 | Python、PyTorch、Ultralytics、导出器和运行时版本 |
| 主机记录 | `scripts/collect_environment.py` 生成主机、编译器、SDK 和可选 GPU 信息；缺失项显式标记，不以默认值代替 |
| 产物 | checkpoint 路径、文件大小、SHA256、导出命令 |

表中每一项均应对应可读取的配置或原始日志；缺失项应标记为
“未记录”，不得以默认值补齐。

## 2. 可复核实验协议

### 2.1 固定输入集合

验证集必须先生成一个有序的图像清单，并在所有后端复用。建议使用
VisDrone 的 548 张验证图像；若使用 SKU-110K，应记录实际数量和划分来源。
清单不得包含重复 stem，否则 C++ 的 TXT 输出会发生覆盖风险。

对于标准 VisDrone 验证划分，建议先生成并冻结 548 行的 UTF-8 路径清单，
再将该清单的 SHA256 写入 manifest。所有后端、PyTorch 基线和性能测试均应
按同一行序读取；目录遍历顺序不能作为隐含协议。若实际划分不是 548 张，
应在结果中报告真实数量，并说明筛选规则。

C++ runner 的 `--source` 可直接接收该清单（`.txt`）。清单中的相对路径以
清单所在目录为基准解析，空行和 `#` 注释行忽略；缺失文件、不支持的后缀及
重复文件 stem 会在推理前报错。这样可确保逐图 TXT 文件与 manifest 一一对应，
而不是依赖不同文件系统的目录枚举顺序。

```bash
python scripts/evidence_manifest.py create \
  --dataset visdrone --split val \
  --images artifacts/visdrone-val.list \
  --image-root /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val \
  --predictions artifacts/onnx_txt \
  --checkpoint runs/esmoe_n/weights/best.pt \
  --training-metadata artifacts/training-provenance.json \
  --model onnx=artifacts/esmoe_n.onnx \
  --model mnn=artifacts/esmoe_n.mnn \
  --report metrics=artifacts/onnx_map.json \
  --command "./yolomaster_edge --profile visdrone --source artifacts/visdrone-val.list" \
  --acceptance \
  --output artifacts/onnx-evidence.json
```

该命令对每张图记录相对路径、字节数和 SHA256，并记录清单摘要、运行平台、
Git 提交号和后处理参数。模型和数据不应直接提交到 Git；建议将它们与
预测、日志和清单一起放入 GitHub Release，并在技术总结中引用 Release 的
SHA256。

两个 mAP 评估器分别记录仅包含有序相对路径的
`image_manifest_sha256`，以及对每一行 `相对路径 + 文件 SHA256` 计算的
`image_list_sha256`。后者与 evidence manifest 使用同一算法。参考结果门禁
同时比较这两个字段；即使文件名与顺序未变，只要图像内容被替换，比较也会
失败，从而避免数据版本漂移被误判为跨后端一致。若清单位于数据集目录之外，
必须使用 `--image-root` 指定归一化根目录；根目录之外的条目会被拒绝。

将证据包从 Ubuntu 传回 Windows 或 Release 目录后，可用 `verify` 重新核对
文件内容：

```bash
python scripts/evidence_manifest.py verify artifacts/onnx-evidence.json \
  --acceptance \
  --images-root /data/VisDrone/images/val \
  --labels-root /data/VisDrone/labels/val \
  --predictions-root artifacts/onnx_txt \
  --models-root artifacts \
  --checkpoint-root runs/esmoe_n/weights \
  --calibration-root /data/VisDrone/images/train
```

### 2.2 预处理与后处理

VisDrone 的默认 profile 为 `imgsz=640`、`conf=0.001`、`iou=0.70`、
`max_det=300`、`multi_label=true`。输入采用居中 letterbox，填充值为
114，颜色顺序为 RGB，张量布局为 NCHW，数值归一化为 `float32/255`。
运行时的通用默认值仍保留给普通检测场景；验收命令应显式写出
`--profile visdrone`，并在日志中保留最终生效的参数。

EsMoE 还必须记录路由语义：静态 ONNX/NCNN 导出采用
`dense_fallback`，PyTorch 基线须使用相同路径；只有在后端确实保留
top-k dispatch 并完成独立核验时，才可标记为 `native_sparse`。不同语义的
结果不得直接计算精度差值。

小目标场景可在单独的 NMS sweep 中比较 `conf`、`small_conf`、面积阈值、
IoU 与 `max_det`。sweep 的结果不能替代固定协议下的主验收结果；每一次
比较都必须使用同一图像清单和同一类别映射。

### 2.3 精度指标及单位

评估器同时输出两种差异，避免把单位混用：

* `delta_mAP50-95_pp = (candidate - reference) * 100`，单位为百分点；
* `delta_mAP50-95_pct = (candidate - reference) / reference * 100`，单位为相对百分比。

Issue #51 的验收命令推荐使用 `--max-abs-delta-pp 0.5`（FP32）和
`--max-abs-delta-pp 1.0`（INT8）。旧参数 `--max-abs-delta-pct` 仅用于
明确的相对百分比比较，不能与百分点门禁同时使用。

正式评估必须使用 `visdrone2yolo` 转换后的 YOLO 标签。原生 VisDrone
`x,y,w,h,score,category,...` 行仅用于诊断，不能作为忽略区域语义已经
定义的正式验收标签。

```bash
python scripts/eval_map.py \
  --preds artifacts/onnx_txt \
  --images artifacts/visdrone-val.list \
  --image-root /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val \
  --label-format yolo --classes visdrone \
  --routing-semantics dense_fallback \
  --imgsz 640 --conf 0.001 --iou 0.70 --max-det 300 --multi-label \
  --min-images 500 \
  --reference-json artifacts/pytorch_map.json \
  --max-abs-delta-pp 0.5 \
  --json artifacts/onnx_map.json
```

### 2.4 性能测量

各后端必须在相同 CPU、输入尺寸、线程数和图像顺序下测量。建议先预加载
图像，进行至少 10 次预热，然后进行 100 次以上重复，并将预处理、推理、
后处理和端到端耗时分别记录。报告至少包含均值、P50、P95、P99 与 FPS，
同时给出 CPU 型号、运行时版本、编译器、线程数和精度模式。运行器可通过
`--benchmark-json` 输出包含协议、主机架构、编译器、CPU、逻辑 CPU 数、
构建日期和汇总统计的机器可读 sidecar；`--csv` 保留逐图计时。虚拟机结果
只能标记为虚拟机结果，不应外推为 ARM 或 Jetson 实机性能。

主机与工具链信息可在运行前由无第三方依赖的
`scripts/collect_environment.py` 采集，并按 `environment.schema.json` 校验。
该记录应与 benchmark sidecar 和 evidence manifest 一并归档；它描述运行
条件，不构成独立的精度或性能结论。

发布时建议使用统一结果表，并为每个单元格保留证据引用：

| 后端 | 模型/清单摘要 | 图像数 | mAP50-95 | 相对参考差值（百分点） | 端到端 P50/P95/P99 | FPS | 主机与运行时 |
|---|---|---:|---:|---:|---|---:|---|
| PyTorch 基线 | manifest / JSON | N（VisDrone 为标准 548；SKU-110K 按实际划分） | JSON | -- | CSV | CSV | manifest |
| ONNX Runtime | manifest / JSON | N（VisDrone 为标准 548；SKU-110K 按实际划分） | JSON | JSON | CSV | CSV | manifest |
| NCNN 或 MNN | manifest / JSON | N（VisDrone 为标准 548；SKU-110K 按实际划分） | JSON | JSON | CSV | CSV | manifest |

该表是报告结构模板，不是本分支的实验结果；只有在模型摘要、图像清单
摘要、逐图预测和原始日志均可核验时，才可填入数值。

## 3. 证据分级

为避免把 smoke test 误写成完整验收，采用以下分级：

| 级别 | 内容 | 可支持的结论 |
|---|---|---|
| L0 | Python 契约测试、CMake 诊断构建、参数错误处理 | 接口与静态约束成立 |
| L1 | 真实 ONNX/图片单图或小子集运行 | 模型加载、预处理、解码链路可运行 |
| L2 | >=500 张固定验证集、参考 JSON、逐图预测和 SHA256 | 可审计的 FP32 精度验收 |
| L3 | L2 加 >=300 张独立校准集、INT8 mAP 门禁 | 可审计的 INT8 验收 |
| L4 | 两个平台原生构建、同图逐框对齐和原始 benchmark 日志 | 跨平台部署结论 |

截至本分支的用户提供记录，针对 Issue #51 的契约测试已通过；Ubuntu 22.04
x86_64 上完成过 YOLOv5s ONNX 单图 smoke（6 个检测框，端到端约
970.873 ms）。该记录属于 L1，模型不是 EsMoE-N，不能替代 L2--L4。测试
数量随门禁增补而变化，因此以提交时的测试日志为准，不在技术结论中固定计数。

## 4. 导出与运行时注意事项

1. EsMoE 的稀疏路由可能包含导出不友好的动态控制流。NCNN 导出脚本使用
   dense routing，并在转换后执行实际的 param/bin 加载 smoke。
2. NCNN 的输入和输出 blob 名称不是 ABI 固定值。导出器将实际名称同时写入
   `<param-stem>.metadata.yaml` 与兼容用的 `metadata.yaml`；C++ 运行时优先
   使用同名 sidecar，并在加载前校验其名称确实存在于 `.param` 图中。没有
   sidecar 时仅推断唯一端点；多输入或多终端图必须显式提供元数据，显式声明
   的 prototype 缺失则直接失败，旧的 `in0/out0/out1` 只保留为兼容回退。
3. ONNX Runtime 输出在进入共享 decoder 前必须满足 FP32、rank-3、正维度
   的检测张量约束；同时兼容 `[1,features,anchors]` 和
   `[1,anchors,features]` 两种常见布局。
4. Windows 模型路径按 UTF-8 转换为 UTF-16，避免中文目录在 ORT 加载时
   被截断或替换。
5. INT8 脚本只负责生成量化模型和校准清单；只有把生成的预测交给
   `eval_map.py` 并通过百分点门禁，才能在报告中写“INT8 验收通过”。

## 5. 发布结构与审核材料

正式发布按“结论、方法、证据、限制”四部分组织。每一个数值都必须能够
回溯到模型摘要、图像清单摘要、逐图预测和原始日志；缺少任一项时，该字段
应标记为“待复现”，不能用估计值或其他运行的结果填充。

| 评审字段 | 正式结果应提供 | 本分支当前状态 | 升级条件 |
|---|---|---|---|
| 训练与数据 | checkpoint、epoch、数据版本、划分和类别映射 | 工具支持记录，文件尚未提供 | 归档 provenance JSON 与 checkpoint SHA256 |
| 导出产物 | ONNX checker/opset、NCNN/MNN 文件及哈希 | 导出与结构检查已实现 | 对真实 EsMoE-N 运行并保存 `export_summary.json` |
| 精度一致性 | 至少 500 张固定验证图、PyTorch 基线、逐图预测 | 评估器和百分点门禁可运行 | 生成三后端 TXT、JSON 指标和清单哈希 |
| 垂类后处理 | 输入尺寸、letterbox、NMS/小目标阈值 | `visdrone`/`sku110k` profile 已固定 | 用同一协议完成 NMS sweep，并归档配置 |
| 性能 | 固定线程、预热/重复次数、P50/P95/P99/FPS | C++ CSV 与摘要字段已实现 | 在同一主机上完成至少两个后端的原始日志 |
| 平台与发布 | 两个平台的构建/运行记录、Release 产物 | Ubuntu x86_64 L1 smoke；无第二平台实测 | 补 Windows/ARM64 原生日志及模型 Release |

该矩阵是提交前的状态记录，不是对外的实验结果。只有“升级条件”全部满足
后，才应在 Discussion 中填写具体 mAP、延迟或吞吐数字。

本分支的改进重点是把这些要求固化为可执行门禁：

* `scripts/evidence_manifest.py` 生成可审计的输入与产物清单；
* `scripts/eval_map.py` 明确百分点/相对百分比的差异单位；
* `--profile visdrone` 固定垂直场景的默认后处理；
* CMake 的 `REQUIRE_ORT/REQUIRE_NCNN/REQUIRE_MNN` 防止发布出静默缺后端的
  部分二进制；
* NCNN sidecar、ORT 形状检查和 Windows UTF-8 路径处理降低跨平台隐性差异。

## 6. 完成正式验收前的待办项

1. 获得并记录 EsMoE-N checkpoint、VisDrone 数据版本及 SHA256。
2. 生成固定的 548 张验证清单和至少 300 张训练校准清单，确认两者按内容
   hash 不相交。
3. 在 PyTorch、ONNX 和 NCNN 或 MNN 上生成逐图 TXT、参考 JSON、raw tensor
   parity 与完整日志。
4. 在 Linux x86_64 与 Windows x64（或真实 ARM64 设备）完成 CMake 构建，
   保存编译命令、二进制信息和同图逐框对齐报告。
5. 仅在百分点门禁通过后更新结果表，并将模型、预测和清单发布到 Release。

完成以上步骤后，`TECHNICAL_SUMMARY_ZH.md` 可以作为 Discussion 技术总结
的主体；在此之前应将结果表标记为“待目标机器复现”，并保留对应的原始日志
与文件摘要。
