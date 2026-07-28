# Issue #54 实验与研发链路

本文只记录真实发生的问题、修改和实验。正式指标在 30 epoch 训练、统一测速及路由审计完成后
从生成文件回填；未完成部分不得提前写结论。

## 1. 从任务要求到研究问题

Issue #54 要求完成 MoE、MoT、MoA 消融，解释不同场景的路由行为，并探索混合架构。仓库已具备
三类模块、基础诊断工具和若干边界测试，继续重复“能否训练”价值有限。

检查现有实现后发现一个影响结论可信度的缺口：若分别训练不同数据集的模型，再比较路由，观察到的
差异同时包含“输入域变化”和“checkpoint 参数变化”，无法单独归因于场景。

因此本实验聚焦两个问题：

1. 固定同一个 VisDrone 训练所得 MoT checkpoint，只改变输入域或场景，路由是否仍有稳定差异？
2. 完整 MoT 的 6 个 Transformer block 成本较高，只在低分辨率 P5 neck 放置 1 个 MoTBlock，
   能否取得更好的精度/延迟折中？

## 2. 实验控制设计

四个模型统一使用 VisDrone、640 输入、30 epoch、batch 16、seed 42：

| key | 结构 | 作用 |
|---|---|---|
| `v10` | EsMoE backbone + 普通 neck | MoE 基线 |
| `v10_mot` | EsMoE backbone + 6 个 MoTBlock | MoT 实验组 |
| `v10_moa` | EsMoE backbone + MoA neck | MoA 对照组 |
| `v10_mot_p5` | EsMoE backbone + P5 的 1 个 MoTBlock | 新混合架构 |

控制措施：

- 四模型分配到四张相同的 RTX 5090，同时训练；
- benchmark 在训练结束后于同一张 GPU 串行执行；
- benchmark 加载各自 `best.pt`，记录 checkpoint SHA-256；
- 跨域路由分析在一个进程中只加载一次 `v10_mot/best.pt`；
- 各域等量、固定 seed 抽样，统计单位为图像而不是 token；
- brain-tumor 只做域外路由审计，不报告跨类别体系的医疗 mAP。

## 3. 第一次实现：先打通最小闭环

新增一键编排脚本，串联数据准备、模型构建、并行训练、统一测速和路由审计；新增 P5-MoT YAML；
新增跨域 hook，提取各 MoT 层三类专家的 top-1 share、dense probability、熵、有效专家数和 margin。

为科研图像补充：

- 16-bit/float TIFF 读取；
- 有限像素 0.5/99.5 分位数归一化；
- 灰度复制为三通道；
- 保持宽高比的 letterbox。

先用 COCO128、160 输入、1 epoch 做 smoke test，不直接消耗完整实验预算。

## 4. Smoke test 暴露的两个问题

### 4.1 四进程首次启动发生共享资源竞争

**现象**：四个训练进程首次并发进行 AMP 能力检查时，共享的参考权重 `yolo26n.pt` 尚未准备好，
产生下载/读取竞争。

**原因**：每个子进程都假定公共依赖已经存在，但首次运行不满足该前提。

**处理**：在父进程中增加串行 preflight，先完成一次参考权重检查，再启动四个训练子进程。

**再实验**：四个模型能够并发进入训练，不再争用首次下载。

### 4.2 P5-MoT 在 autocast 推理时报 dtype 不一致

**现象**：P5-MoT smoke 流程在稀疏专家分派处触发 FP32/FP16 赋值不匹配。

**原因**：expert 输出受 autocast 影响，而稀疏路径使用高级索引回写 FP32 累加器；高级索引赋值
不会像普通非原地加法一样自动提升 dtype。

**处理**：在回写前把 `expert_out * weight` 显式转换为累加器 dtype，并新增 CPU bfloat16
autocast 回归测试。

**再实验**：四模型 smoke 训练、测速和跨域路由审计全部跑通。

## 5. 全尺寸校准又发现的问题

### 5.1 自动 AMP 恢复掩盖了真实失败

**现象**：VisDrone、640、batch 16 的 1 epoch 校准中，四个模型的首次 AMP pass 均触发非有限
梯度保护；训练器恢复健康 checkpoint、关闭 AMP 后重跑。最终 `results.csv` 是有限值，看起来像
一次正常训练。

**风险**：只看最终 CSV 会漏报首轮失败，同时一个“1 epoch”请求实际执行了额外计算，破坏训练
耗时比较。

**处理**：

- 恢复时输出明确 warning；
- 写入 `recovery_events.jsonl`，记录 epoch、原因、AMP 状态、checkpoint 和时间；
- 稳定性汇总增加恢复次数、原因和 AMP fallback 字段；
- 正式公平对比统一改为显式 `--no-amp`。

**再实验**：正式实验首轮四组均正常完成，未产生恢复事件。

### 5.2 初版 benchmark 测错了模型身份

**现象**：校准汇总最初按 YAML 重新构建未训练模型，类别数为 80；VisDrone checkpoint 实际为
10 类。这会让 Params、FLOPs、延迟和被报告 checkpoint 不一致。

**原因**：训练与 benchmark 是两条独立路径，后者没有显式接收训练权重。

**处理**：增加 `--trained-weights`，从每组 `best.pt` 加载精确模型，同时写入 `nc`、权重路径和
SHA-256；实际 FLOPs 也在该模型对象上测量。

**再实验**：校准汇总正确识别 `nc=10`，参数量与训练 checkpoint 一致。

### 5.3 “确定性”仍有硬件算子边界

正式日志提示 CUDA memory-efficient attention 和部分 pooling backward 在当前 PyTorch 中没有
确定性实现。当前做法是固定 seed、数据、机器和软件版本，并完整保留 warning；因此本实验属于
协议级可复现，不声称跨硬件 bitwise 一致。

### 5.4 checkpoint 与指标行的语义不应混用

完整训练中曾出现某轮指标回落，暴露出初版汇总把末轮 `results.csv` 指标与 `best.pt` 身份放在同一行，
却没有明确两者关系。正式四组的最佳值最终恰好都出现在第 30 轮，但代码仍增加
`final_*`、`best_observed_*` 和对应 epoch 字段，避免其他实验在最佳轮不是末轮时误报。

## 6. 路由解释如何避免伪结论

初版想直接把所有 token 当样本做显著性检验，但同一图像内 token 强相关，会夸大样本量。
最终改为先按“图像 × 层 × 专家”聚合，再进行：

- bootstrap 95% CI；
- 双侧 permutation test；
- Hedges' g；
- Benjamini-Hochberg FDR 校正。

只有 `q <= 0.05` 且差值 CI 不跨 0，才描述为稳定差异。`irregular_occluded` 由目标密度、尺度和
长宽比变异构成，只是遮挡/不规则场景代理；DeformableTransformer 激活上升也不能直接证明检测
更准确。

### 6.1 场景集合重叠使独立检验失效

首次场景审计后检查样本指纹发现：

- `dense` 与 `irregular_occluded` 重叠 104/128；
- `large_objects` 与 `sparse` 重叠 76/128；
- `dense`/`sparse` 和 `small_objects`/`large_objects` 两组主比较互斥。

共享图像不满足独立样本 permutation test 的前提。修复后新增 `sample_overlap.csv`；共享样本的组
只保留描述统计，`comparison_valid=false`，不再生成可引用的 CI、p-value 或 FDR 显著性。两条互斥
主比较保持有效。

### 6.2 深层 Top-1 路由接近均匀

分层热力图显示最深的两个 `model.23` 路由概率几乎固定为 1/3，Deformable top-1 为 0。参数没有
冻结；代码检查发现该层使用 `top_k=1`，单个 top-k 值重归一化后恒为 1，主任务梯度主要依赖 2%
`exploration_eps`。同一随机输入的任务损失探针中，router 梯度范数约为：

| 配置 | router 梯度范数和 |
|---|---:|
| top-k=1, exploration=0 | 8.9e-9 |
| top-k=1, exploration=0.02 | 1.53e-4 |
| top-k=2, exploration=0.02 | 9.90e-3 |

该结果解释了深层路由弱分化，但尚未证明 straight-through 或更高 exploration 能提升检测精度。
因此本次把它作为后续消融假设，不在没有重训数据时提交行为改变。

## 7. 正式实验结果

> 状态：运行中。结果将从 `runs/mot_cross_domain/` 自动汇总后回填，不手工预估。

待回填内容：

- 四模型 mAP50-95、mAP50、Params、实际 FLOPs、P50/P95/P99 latency；
- loss/NaN/恢复事件；
- 同 checkpoint 的 VisDrone、COCO128、brain-tumor 路由统计；
- VisDrone 密集/稀疏、小/大目标、遮挡代理场景统计；
- 至少 3 条带数据、置信区间和限制条件的场景建议；
- P5-MoT 是否达到 `mAP50-95 +1%` 或 `latency -10%` 的 Issue 判据。

## 8. 证据与复现入口

| 证据 | 路径 |
|---|---|
| 固定提交、命令、设备与数据清单 | `runs/mot_cross_domain/experiment_protocol.json` |
| 每模型原始训练日志 | `runs/mot_cross_domain/logs/train_*.log` |
| loss 与检测指标 | `runs/mot_cross_domain/training/*/results.csv` |
| checkpoint 身份与统一汇总 | `runs/mot_cross_domain/training/summary.csv` |
| 路由明细、统计检验与样本哈希 | `runs/mot_cross_domain/routing/` |
| 实现提交 | `58cb439407e2f5f7a6e1c4b6a3a9382499713e88` |

代码回归测试为 80 passed，覆盖既有 MoT 边界、dtype 修复、TIFF、统计方法、P5 配置和恢复事件。
