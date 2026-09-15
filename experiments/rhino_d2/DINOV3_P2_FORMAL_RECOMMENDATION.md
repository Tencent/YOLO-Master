> **Archived research document.** Statements about copying records, running tests and delivery dates below describe the original packaging stage, not actions performed by this integration commit. Referenced evidence stays in the archive; it is not bundled here.

Public access: [public derivative of this document](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/DINOV3_P2_FORMAL_RECOMMENDATION.md). Original document identity remains local at packaging SHA `781a6f148a46d59a89ea40806de8e10eb7b98941`. The public package is a sanitized derivative, not the original Git history; its PROVENANCE.json distinguishes byte-preserved files from path-sanitized records. Local-only references below are not public links. Import edits: this notice and pinned link destinations only; archived figures are linked rather than embedded. Scientific text and numbers are unchanged. The protocol's original implementation is distinct from post-freeze public-API hardening in integration Commit 1 `3a62b6f480e2855c3034f9788f69b804b81a5b56`.

---

# D2 / DINOv3-S：P2 正式结论与终止建议

日期：2026-09-10。最终科学证据锚点：`e3e11d0cfebcd0893f5cd5bae46cf7f48ba8429a`。
状态：**terminal；new_experiments_allowed=false**。本文件是证据归纳，不是新的实验协议。

## 一、决策摘要

当前 DINOv3-S → YOLO-Master-N、P4、COCO-mini、冻结预算下，固定归一化
Response-Field 在独立 train-only probe 上通过信号可行性门，在最终评估上满足
预注册 Mechanism Support；但相对 matched-compute Static-2V 的 clean detection
主指标为 **No detectable change**。不将本路线作为已验证的性能增益方案推广。

此处matched-compute仅指最终保留六臂的受控计算结构；人工中断段及pilot额外计算另行披露。

优化可行性与任务效用并不等价：本次几何修正及响应对齐改善，没有转化为可检测的稳定检测收益。
这不是“证明方法无效”，也不是“两方法等价”；三 seed 的区间仍容纳有意义的正负收益。

## 二、一条主线，八个保留原判定的阶段

| 阶段 | 冻结结论 | 支持什么 / 不支持什么 |
| --- | --- | --- |
| P1（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/DINOV3_P1_GO_NO_GO.md`） | No-Go；mean ON−OFF=-0.002170 | 静态 KD 可学习，但未检出稳定检测收益；不等于 Foundation 蒸馏普遍无效。 |
| P2-01（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_gradient_conflict/DINOv3_P2_GRADIENT_CONFLICT_RESULT.md`） | Gradient conflict Inconclusive | 局部梯度证据不足；不能写成“排除了梯度冲突”。 |
| P2-02（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_align_dim/DINOV3_P2_ALIGN_DIM_RESULT.md`） | Dim bottleneck No support | 64→128 的单变量解释未获支持；不排除所有容量/维度问题。 |
| P2-03（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_response_gap/formal/DINOV3_P2_RESPONSE_GAP_RESULT.md`） | H1/H2 Support；H3 Inconclusive | Static alignment 改善未迁移到 response alignment；response gap 与任务退化的因果关系未成立。 |
| P2-04（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/DINOV3_P2_RESPONSE_FIELD_CALIBRATION_RESULT.md`） | Calibration Failed | 原冻结 alpha 候选无合格者；永久保留失败，不用后续成功改写。 |
| P2-05（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/DINOV3_P2_RESPONSE_GRADIENT_GEOMETRY_RESULT.md`） | Geometry H1 Support | 去双支 aligned-Z 放大约16.51×，norm贡献约12.44×，投影相对传递约0.92；是局部几何诊断，不是性能归因证明。 |
| P2-06（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/DINOV3_P2_RESPONSE_NORMALIZATION_RESULT.md`） | Feasibility Go | 新 train64b 上组合 r_grad 从约15.632到1.151，normalized r_loss≈0.627；只证明冻结 checkpoint/数据上的信号可比，不保证全训练轨迹匹配。 |
| [P2-07](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_normalized_response_formal/final_result.md) | Mechanism Support + No detectable change | 响应机制指标改善，但 clean detection 未检出稳定收益；本路线终止。 |

P2-05 的第二项梯度放大与 P2-06 的组合 Foundation 预算比不是同一统计量，不能直接作数值等式比较。
本项目沿用的实际 tap 是 Detect 输入的 **neck/FPN P4（第19层）**，不是第6层 backbone P4。

## 三、三个明确贡献

### 1. 工程贡献：可接手、可审计的集成

在仓库已有 Foundation 框架上完成 D2 实验链路与验证：教师资产/预处理锁定、教师冻结、
Student hook、共享 projector、真实检测损失与独立 KD、确定性 paired perturbation、
双 forward 的 BN train semantics/post-clean 精确恢复、非有限数值停止、运行参数及证据哈希审计。
这些是集成、补全和实证交付，**不声称从零发明整套 teacher、hook 或 projector 框架**。

主要入口：paired primitives（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:ultralytics/nn/foundation/response.py`）、
原有 Foundation wrapper（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:ultralytics/nn/foundation_distill_model.py`）、
D2 扩展（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/scripts/normalized_response_trainer.py`）、队列与审计（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/scripts/complete_normalized_response_formal.py`）。

### 2. 方法学贡献：可证伪而非追逐单次涨点

预注册单一主指标及阈值；三 paired seeds；相同视图、初始化、Teacher/Student forward 数和训练预算；
配置、resolved runtime、实际 loss 轴三层审计；先机制 probe 再训练；失败不扩 alpha 候选；
区分机制次指标与检测主指标。Matched-compute 指保留六组实验的受控计算结构，不是声称墙钟绝对相等。

### 3. 科学发现：局部几何、响应机制与任务效用可分离

静态接近不保证局部响应接近；raw response cosine 的小 response norm 对梯度放大有局部支持；
固定 norm compensation 在冻结 probe 上修复信号尺度，并在预定义 response 指标上获得改善。
然而最终检测主指标没有稳定改善。这削弱“只因该已测局部几何失配所以未涨点”的解释，
不证明所有优化问题均已排除，也不证明 response mismatch 是 P1 No-Go 的原因。

## 四、P2-07 主指标：必须同时展示三个 seed

每臂取 CSV 第41–50轮 mAP50-95 中位数；Δ=Cnorm−B，AP标度0–1。

| seed | B Static-2V | Cnorm | Δ | Δ（mAP百分点） |
| --- | ---: | ---: | ---: | ---: |
| 20260824 | 0.052435 | 0.050625 | -0.001810 | -0.1810 |
| 20260825 | 0.047660 | 0.050705 | +0.003045 | +0.3045 |
| 20260826 | 0.051340 | 0.048680 | -0.002660 | -0.2660 |

mean Δ=-0.000475；sample SD=0.003077893；paired 95% t CI（df=2）=
[-0.008120910,+0.007170910]，即[-0.812091,+0.717091] mAP百分点。
满足 `abs(mean Δ)<0.003` 且 CI 含0，故判 **No detectable change**。
seed25 单独超过+0.3百分点，不能取代三 seed 判定。未进行等价性检验。

三 seed 与均值区间（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_terminal_freeze/figures/paired_delta.png`）

## 五、机制及鲁棒性次指标

ResponseGap(Cnorm)−ResponseGap(B) 的 pooled 均值=-0.007060625，image-cluster
95% CI=[-0.010139964,-0.003883837]；seed24/26 区间上界<0，seed25区间含0。
满足冻结的 pooled + 至少2/3 seed 规则，不应写成“三个 seed 都改善”。
该区间条件于冻结模型和128张训练诊断图，**不是主指标的训练-seed t CI，也不是因果中介证明**。

8-corruption macro mAP 的 Cnorm−B，按seed依次约+0.000590、+0.000737、-0.005186；
只作描述，不支持另包装“鲁棒性稳定涨点”。完整数值见
[robustness_summary.csv](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_normalized_response_formal/robustness_summary.csv)。
这里 clean secondary 使用 final epoch EMA 与统一 square 预处理，不能替换主指标 late-window 口径。

## 六、Evidence supports / does NOT support

**Evidence supports**

- Static KD 梯度链工作，且在冻结 probe 中改善 clean static relation。
- Static/response matching 可脱钩；被测 raw response cosine 有显著局部放大。
- 固定归一化在独立 train-only probe 上恢复可比信号，最终机制指标达到预注册支持门。
- 当前检测结果没有达到稳定、可行动的改善标准；单 seed 结论会误导。

**Evidence does NOT support**

- 所有梯度冲突、容量、维度或优化问题已被排除。
- 全训练过程始终 signal-matched，或 response alignment 是检测瓶颈的因果证明。
- normalized Response-Field 提高 clean detection、普遍优于静态蒸馏、达到 SOTA。
- DINOv3/response distillation 普遍无效，或者三 seed 证明两方法等价。

## 七、最终冻结与 B24 恢复披露

对锚点的五份最终文件、三个 pair audit 和六臂 epoch/逐批证据已独立离线核验：
**三组 PASS，六臂各50 epoch / 25600 batches**；重新计算 mean/SD/t CI 一致。
封存审计（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_terminal_freeze/audit.json`） 保留文件与最终 checkpoint 哈希。
未修改原始结果、loss、seed、arm、threshold；未启动任何新训练。

补充收尾测试从6,144条response原始记录独立重算四组image-cluster bootstrap区间，
与归档一致；另检查文档链接、恢复副本哈希及绘图数据源。它不替代全仓库CI或全量训练复跑。
本次相关回归合计 59 passed（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_terminal_freeze/regression.xml`），独立文档读者复核通过；
上述数值核验与文档审阅是不同层次的检查，不互相替代。

B24 的首次执行被手动中断，隔离目录留有33个完整 CSV epoch。随后一次重启被旧证据目录保护拒绝，
该尝试未开始训练；归档后从 epoch0 新初始化运行完整B24，而非接续旧 checkpoint。
中断段不拼入正式50轮，不进入 paired delta；其额外计算成本不纳入受控六臂预算，需单独披露。
`automatic_retries=0` 只表示无自动重试，**不表示没有人工恢复**。

[恢复说明](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/P2_07_RECOVERY_DISCLOSURE.md) 收录原始记录与完整路径/hash清单。
中断原因依据当时 recovery note 与用户说明；文件本身不是独立的操作授权/动机证明。
因此不声称从日志证明了“从未看过中间指标”。这种治理局限不应被掩盖。

## 八、答辩主图与结项安排

研究机制链（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_terminal_freeze/figures/mechanism_chain.png`）

图中箭头表示研究检查顺序，不表示已证明的因果链。答辩先讲三个贡献，再用三seed图解释为何不宣称涨点。

| 日期 | 只做交付包装；不是自动定时任务 |
| --- | --- |
| 9/10–9/11 | 统一建议书、数据表与两张图、DINOv3最终PR草稿；本文件及封存审计已交付。 |
| 9/12 | 全量复现审计、链接/命令、README与测试清单；只核验既有证据，不重跑训练或改科学定义。 |
| 9/13 | 答辩叙事、风险、limitations；预留评审追问。 |
| 9/14 | 只修文档、排版和展示问题；事实纠错须留变更记录，不重判阈值。 |

**正式建议：当前协议下不继续扩大 Response-Field 路线。未来若重启，应另立协议并取得授权，优先重新审视数据规模、任务或 teacher-stage，而不是继续调 alpha。**
