# Issue #54 Phase 1 实验基础设施实现报告

> 状态：Phase 1 Review、修复、环境诊断和 random-init synthetic integration 已完成；提交状态以 Git 记录为准，未 push 或创建 PR
>
> 分支：`issue54-mot-routing-stability`
>
> Phase 0 基线提交：`5c0db33af899b039f94bfdd6453857ff9795542c`

## 1. 范围与停止条件

本阶段仅实现实验身份登记、MoT 路由证据导出、跨独立 seed 分析、合成输入测试和协议文档。
没有下载 VisDrone，没有启动正式训练或 GPU 长实验，也没有修改 MoT、MoE、MoA 的核心数学行为。

Phase 1 在以下条件满足后停止：

1. Phase 0 文档单独提交且不 push；
2. Phase 1 工具通过静态检查、Python 编译、CPU 单元测试和 synthetic CLI smoke；
3. OpenMP 门禁和真实官方 MoT YAML 的 random-init synthetic integration 通过；
4. 正式数据、训练协议、AMP/FP32 和 batch 等参数继续保持未冻结。

## 2. 新增实现

### 2.1 实验登记

`scripts/issue54/build_experiment_registry.py` 从显式 manifest 构建 registry，不从目录名称推断任何参数。
它拒绝以下会夸大独立重复数的情况：

- 重复 `experiment_id`；
- 同一模型、数据版本、split 和 seed 被重复登记；
- 相同 checkpoint 哈希被包装成不同 seed；
- `diagnostic`、`failed` 或 `not_executed` 被计入正式训练数。

证据强度标签固定为：少于 3 seeds 不足以开展跨 seed 推断，3--4 seeds 仅为探索性，
至少 5 个预注册独立 seeds 才达到更强结论的最低门槛。

### 2.2 MoT 路由导出

`scripts/issue54/export_mot_routing.py` 在不修改核心模块的前提下，对现有 MoT router 注册临时 forward hook。
hook 读取当前 upstream 已公开的 `(weights, indices, logits)` 输出，并使用 router 中保存的 temperature
重建完整专家概率。导出器会先确认第三项不是已归一化概率，再用 sparse weights 和 top-k indices 反向核对 logits
语义；token 级概率必须有限、非负且按专家轴归一化。导出完成或发生异常后都会移除 hook，并恢复所有 module 的
train/eval 状态。

真实导出模式要求显式提供：

- 已验证的 `status=passed` 实验 manifest；
- 脱敏的图像 manifest；
- checkpoint、config 和数据根目录；
- checkpoint、config、图像及数据清单的 SHA-256；
- 实际推理 batch 和重复推理次数。

synthetic 模式只构建一个小型 CPU `MoTBlock`，用于验证 hook、概率、重复推理和输出契约；它产生
`status=diagnostic` 证据，不能进入正式统计。

### 2.3 跨 seed 分析

`scripts/issue54/analyze_cross_seed_routing.py` 以
`model_variant + dataset + dataset_version + split + image_id + layer_name + layer_index`
为协议和样本身份，并按 `expert_names` 的语义而非数组位置对齐专家。
所有 per-image、per-layer、global、utilization 和 scene 汇总都保留完整协议维度，不跨 dataset/version/split 混合。

输出包含：

- 所有可对齐的 seed-pair 原始比较；
- dominant expert 和 token top-1 agreement；
- 完整概率的 pairwise Jensen--Shannon divergence；
- raw/normalized route entropy；
- per-image、per-layer 和全局描述汇总；
- 每个 seed 的分层专家利用率及 between-seed 样本方差；
- 场景组效应方向在 seed 间的重现率；
- 同一 checkpoint 重复推理的确定性审计；
- 缺失 image、缺失 layer 和无法对齐 shape 的显式记录。

NaN、Inf、负概率、概率和错误、专家集合不一致、manifest 身份冲突和重复路由记录会被拒绝。

## 3. 复用的 upstream 能力

本实现复用了以下当前 `upstream/main` 能力：

- `MoTBlock.router(..., return_logits=True)` 的完整 logits 输出契约；
- `scripts/diagnose_mot_routing.py` 中的三类 MoT 专家语义名称；
- `ultralytics.data.augment.LetterBox` 的检测输入预处理；
- `ultralytics.utils.checks.check_is_path_safe` 的根目录边界检查；
- `agent.runtime.cli.contract.manifest_checksum` 和 `redact_sensitive` 的校验与脱敏；
- 现有 MoT 模块的 train/eval 状态和 router temperature checkpoint 状态。

没有另建新的 MoT router，也没有改变 forward、top-k、temperature、expert 或 loss 行为。

## 4. 与 PR #190 的边界

PR #190 是未合并工作，其单 checkpoint perturbation 分析、单 checkpoint JSD/entropy、P5-only、
utility router、adaptive K 和相应解释逻辑不属于当前 `upstream/main`，本项目没有 cherry-pick 或复制这些实现。

本阶段新增价值严格限定为：

- 多个独立训练 checkpoint 的身份审计；
- 相同图像、层和专家语义的跨 seed 对齐；
- 保存全部 seed-pair 原始比较；
- seed 级证据强度与伪重复防护；
- 跨 seed 专家利用率方差、场景结论重现率和 checkpoint 重复推理确定性。

如果 PR #190 后续合并，应在兼容性审计后复用其稳定的底层单分布指标原语，并删除本项目中相应的薄层数学实现；
跨 seed registry、对齐、有效性管理和汇总仍是本项目独立部分。

## 5. 统计有效性边界

- 独立训练 seed/checkpoint 是最高层实验单位；
- image、layer、token、inference repeat 和 seed-pair 都不会增加正式 seed 数；
- 图像级汇总只描述给定 checkpoint 内的样本；
- 所有 pairwise seed 比较均保存，不以单一均值替代原始 seed-pair；
- 3 seeds 只报告探索性分布、效应方向和原始数据；
- 本阶段不产生 p 值，也不使用图像数制造看似精确的总体结论。

## 6. 阶段式实验矩阵

MVP 为 MoT 3 seeds、MoE 1 seed、MoA 1 seed，共 5 次正式训练。推荐扩展为
MoT 5 seeds、MoE 3 seeds、MoA 1 seed，共 9 次正式训练。

扩展只能在 MVP 审计完成后决定：

1. MoT 3 seeds 的 mAP、路由利用率或场景结论明显不同，才扩展 MoT 到 5 seeds；
2. 需要判断差异是否为 MoT 特有现象，才扩展 MoE 到 3 seeds；
3. 3 seeds 不得包装成充分验证的总体统计结论；
4. MoA 是否增加多 seed 由官方合规结果和剩余预算另行决定。

## 7. 尚未解决的问题

- 尚未用真实 YOLO-Master checkpoint 验证路由值和显存开销；本阶段只验证了官方 YAML 的随机初始化模型；
- 尚未冻结 image manifest、场景标签和 VisDrone version 标识；
- 尚未将检测 mAP、loss、目标尺度和遮挡元数据接入联合相关性分析；
- 尚未确定正式训练的 epochs、optimizer、requested/actual/effective batch、AMP 或 FP32；
- 尚未验证 Windows/AutoDL 两个环境产生的导出是否完全一致；
- PR #190 后续若合并，需要执行一次 API 去重与迁移审计。

## 8. 进入真实数据 smoke 前的门槛

1. 用户批准数据来源、版本和保存位置；
2. 固化数据 manifest、样本 ID、sequence 和 scene/occlusion 元数据定义；
3. 选择一个非正式 smoke seed，验证 train/val、checkpoint 恢复和路由导出；
4. 在同一 checkpoint 上运行两次 FP32 推理和两次 AMP 推理；
5. 比较 mAP、loss、路由 top-1、概率差、显存和时间；
6. smoke 审计通过后才冻结 epochs、batch、precision 和正式 seed 列表。

## 9. AMP 与 FP32 建议

当前不锁死精度模式。建议以 FP32 的重复推理结果作为小样本参考基线，再验证 AMP：

- 若 AMP 的训练稳定性、验证指标和路由统计都通过预设容差，正式训练优先采用 AMP，以降低 4090 时间和显存成本；
- 每个正式 checkpoint 保留固定子集 FP32 路由导出，用于检查 AMP 是否改变结论方向；
- 若 AMP 出现非有限值、确定性失败或路由差异超过冻结阈值，则回退 FP32，或把精度模式作为显式实验因素。

阈值必须在真实数据 smoke 后预先冻结，不能在看到正式结果后调整。

## 10. Phase 1 Review 发现与修复

### 10.1 Blocker

- 空 routing JSONL 原先可生成成功 analysis 并返回 0。现要求至少一个 `passed` 或 `diagnostic` routing record；
  空输入 CLI 返回非零且不创建结果。

### 10.2 Major

- JSON/JSONL 原先会静默覆盖。现在默认拒绝已有结果，只有显式 `--overwrite` 才允许替换，并提前检查多输出路径。
- hook 原先只凭三元组位置假设第三项是 logits。现在检查 logits、temperature、dense probabilities、sparse weights
  和 top-k indices 的一致性，并拒绝已归一化概率出现在 logits 位置。
- hook 注册循环原先位于 `try/finally` 外。现在部分注册失败、forward 异常和成功路径都会清理已注册 hook。
- 汇总原先可能丢失 dataset/version/split。现在所有协议级汇总保留完整维度，`analyzed_seed_counts` 不再被最后一组覆盖。
- 缺少 repeat 0 原先仍可能以 repeat 1 为基准通过。现在明确标记为 `invalid_missing_base_repeat`。
- 真实导出默认时间戳原先使用墙钟时间。现在默认继承 manifest timestamp，保证同输入重导出可复现。

### 10.3 Minor

- synthetic provenance 的 Git SHA 已改为 Phase 0 基线 SHA，随机初始化身份使用真实内存 `state_dict` 指纹，
  不再伪装成磁盘 checkpoint。
- CLI 成功日志只显示输出文件名；已知输入/文件系统失败使用无私人绝对路径的单行错误。
- 导出端提前拒绝空 image entries 和重复 `image_id`。
- 新增覆盖保护、空输入、协议隔离、hook 成功/失败清理、参数/梯度/状态保持、中文路径和真实 YAML 集成测试。

## 11. OpenMP 环境诊断

最初命中的解释器是 Conda base，Python 3.13.9、Torch 2.11.0+cpu、NumPy 2.3.5。
该环境同时包含：

- `<conda-base>/Library/bin/libiomp5md.dll`，由 MKL/Conda NumPy 路径使用；
- `<conda-base>/Lib/site-packages/torch/lib/libiomp5md.dll`，由 pip Torch 携带。

在 base 中，单独 `import torch` 和 `torch -> numpy` 均以 OMP Error #15、退出码 3 失败；`numpy -> torch`
偶然成功，说明问题与 DLL 加载顺序有关，但导入顺序不能作为可靠修复。

仓库已有独立 Conda 环境 `yolo-master`。本阶段安全切换到该环境，没有安装、卸载、
删除或重命名任何包/DLL，也没有设置 `KMP_DUPLICATE_LIB_OK`。该环境为：

- Python 3.11.15；
- Torch 2.6.0+cu118；
- TorchVision 0.21.0+cu118；
- NumPy 2.4.4，使用 OpenBLAS；
- CUDA build 11.8，`torch.cuda.is_available() == True`；
- NVIDIA GeForce GPU，4 GiB 显存。

该环境内只有 Torch 携带的一份 `libiomp5md.dll`，没有 MKL DLL。单独导入和两种导入顺序均通过；门禁命令
`python -c "import torch; import numpy; print(torch.__version__); print(torch.cuda.is_available())"` 返回 0。
Conda base 本身仍保留原冲突，本项目不再使用它执行 Phase 1 测试。

## 12. 真实 MoT YAML random-init synthetic integration

使用官方 `ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml`，固定 seed 54，随机初始化，
CPU FP32、`batch=1`、`64x64` synthetic tensor、`eval` 和 `torch.no_grad()`。没有下载或加载 checkpoint。

捕获 6 个真实 MoT 层：

1. `model.14.m.0`
2. `model.14.m.1`
3. `model.20.m.0`
4. `model.20.m.1`
5. `model.23.m.0`
6. `model.23.m.1`

每层专家名称均为 `LocalConvTransformer`、`WindowTransformer`、`DeformableTransformer`。所有概率有限、
非负且和为 1；相同输入重复两次的概率和 token top-1 完全一致。成功路径结束后 hook 数恢复到原值，所有 module
的 train/eval flag 恢复，完整 `state_dict` 未变化，参数没有梯度。

该结果的 manifest 和 routing records 均标记为 `diagnostic`。它只证明工具能接入真实官方模型结构，
不能作为正式实验或效果证据。

## 13. Phase 1 验证记录

- Phase 1 文件 Ruff lint：通过；
- Phase 1 文件 Ruff format check：通过；
- 全仓 `ruff check ultralytics/ tests/ scripts/ agent/`：基线存在 450 个非 Issue #54 错误；
- 全仓 `ruff format --check ultralytics/ tests/ scripts/ agent/`：基线存在 172 个非 Issue #54 待格式化文件；
- Python compile：通过；
- Markdown structural lint（markdown-it 解析、H1/heading、fence、tab 和尾随空白检查）：通过；
- `git diff --check`：通过；
- Issue #54 测试（含真实 YAML integration 和 synthetic CLI E2E）：`46 passed`；
- `tests/test_mot.py`：`27 passed`，4 条既有 head-adjustment warning；
- 三 synthetic seed CLI：导出、registry 和跨 seed analysis 全链路通过；
- 中文输出目录、输入不存在非零退出、空 routing 非零退出和显式覆盖保护：通过；
- 正式证据计数：0，synthetic 结果正确标记为 `diagnostic_not_formal_evidence`。

仍未验证真实 checkpoint、VisDrone、mAP、训练恢复、AMP/FP32 结论或训练稳定性；这些边界不能由本阶段
random-init synthetic diagnostic 外推。
