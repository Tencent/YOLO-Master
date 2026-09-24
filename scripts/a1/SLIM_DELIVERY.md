# A1 MoE × End-to-End：P0/P1/P2 实验与机制诊断（结项）

作者：Ayanami-Makima。

## 本 PR 的边界

这是原 PR [#281](https://github.com/Tencent/YOLO-Master/pull/281) 的**精简候选稿**。
迁移基线为 Tencent/YOLO-Master `3b3ac8b0efc74f0bb7ef0a3c0f9666d2676b85c5`。
本次仅保留 8 个可独立审阅的文件，生产代码的既有文件仅修改 `head.py` 的 6 行可选梯度桥。
不修改 `default.yaml`、`tasks.py`、`loss.py`、全局路由器、优化器或 CI。

本稿不是把 900 多个文件拼成一个大文件，也不是完整历史实验复现包：
配置、训练器、权重、逐轮日志及批量审计结果不进入本次 diff，仍通过固定提交链接追溯。
历史实验使用官方基线 `acce839c7e895d6b179de7f7093fa879e237cc7b` 和原实验分支；
下列历史 AP/延迟**不是在本次迁移基线上重新训练测得**。

## 保留的核心贡献

1. **P1 预训练保持机制**：提取独立 `ResidualFactorAdapter`，实现
   `base(x) + gain * factor(base(x))`。gain 零初始化；base 参数与 BN 统计冻结，
   但不 detach 输入，保留向前层传梯度的能力。可显式传入原生 Dense 或 MoE factor。
2. **P2 梯度机制干预**：仅显式设置普通 Detect 头的
   `p2_o2o_gradient_alpha` 才在训练期建立缩放梯度通路。
   默认值 0 保持 detach；推理/export 忽略该开关；非零干预拒绝 Detect 子类。
   桥接前向数值不变，改变的是共享输入梯度，不是检测损失定义或预测算法。
3. **诊断可靠性**：hook 仅由实验入口显式安装，上下文退出后释放；
   覆盖正常结束、forward 异常、检查失败、写文件失败及从未执行 forward 的清理。
   不删除调用者已有 hook；检查所有尺度的梯度标志，拒绝部分失效。
4. **可维护的回归证据**：独立 CPU 单测覆盖初始等价、冻结、序列化、
   梯度缩放、NMS/E2E 默认前向及 hook 生命周期，而不是仅提交实验结论表。

## 与历史完整实现的区别

- 精简适配器通过 Python 显式构建，不注册新 YAML 类型。
  **不直接加载历史 `C3k2ResidualFactor` checkpoint**；回放旧实验需使用下方固定实验提交。
- 不携带旧实验特定的 gain 优化器策略、探索噪声、dense_mlp 专家实现、断点恢复调度。
  本次 Dense/MoE 单测验证当前原生 factor 的零初始化机制，不声称复现 r28 路由与预算。
- 不将候选 assigner/loss 调参或 one-to-one 轻量 MoE 自动加入默认模型。
- 冻结策略若被外部训练器统一重置，须在该步骤之后调用 `freeze_base_parameters()`；
  本工具不拦截或修改全局训练器。

## P0 / P1 / P2 历史成果摘要

| 阶段 | 实现与证据 | 结论 |
| --- | --- | --- |
| P0 | 同一官方 yolo26n.pt 的 NMS/E2E 路径复评；COCO val5000；真实抑制核插桩 | mAP50-95 为 0.402 / 0.395；固定 16 图 NMS 调用 16 / 0 次 |
| P1 | Dense/MoE × NMS/E2E；3 seeds、12 个正式单元；冻结/路径/导出/性能审计 | A/B/C/D 均值为 0.40200 / 0.39488 / 0.40222 / 0.39465，当前共享 Backbone MoE 无稳定精度收益 |
| P2 | 梯度桥、匹配/阈值/几何诊断、后层效率筛选及 seg pilot | 验证 detach 阻断 one-to-one 对共享特征的直接梯度；恢复梯度不等于获得稳定 AP 提升 |

P1 预算：COCO train20000/val5000，15 epochs，batch=4，imgsz=640；
seeds=260829/260830/260831；SGD，lr0=1e-4，lrf=0.2，momentum=0.9，
weight_decay=5e-4；AMP=False；训练层 4/6/8/23，base 与 BN 冻结；
mosaic/mixup/copy_paste=0。gain 参数组与路由等完整细则以固定证据报告为准。

同设备 batch=1 的 P1 closure GPU 全流程延迟 A/B/C/D：
12.496 / 12.049 / 25.880 / 26.111 ms。
独立 forward profiling 为 5.686 / 6.154 / 13.180 / 14.078 ms；
两套计时范围不同，不混用。Router 与分发/聚合约占被分析 MoE 子模块耗时的 80%；
不据此声称已证明大量 CPU/GPU 数据往返。

这些结果形成了可复核的负结果和下一步优化依据，不宣称 MoE 必然无效，
也不将 detach 解释为精度差距的唯一原因或 router 总梯度为零。

## 固定证据入口

- [P0 基线报告](https://github.com/Ayanami-Makima/YOLO-Master/blob/48ac6071f8d047881935bd68667ebb8883d0f433/smoke/a1/P0_PRETRAINED_REPORT.md)
- [P1 r28 三种子与部署审计](https://github.com/Ayanami-Makima/YOLO-Master/blob/48ac6071f8d047881935bd68667ebb8883d0f433/smoke/a1/P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md)
- [P2 阶段交付](https://github.com/Ayanami-Makima/YOLO-Master/blob/48ac6071f8d047881935bd68667ebb8883d0f433/smoke/a1/A1_P2_STAGE_DELIVERY_REPORT.md)
- [最终 Discussion](https://github.com/Tencent/YOLO-Master/discussions/294)

## 使用和验证

在仓库根目录运行，CPU 即可，不需要数据下载或 checkpoint：

```bash
python -m scripts.a1.diagnose_o2o_gradients --alpha 0
python -m scripts.a1.diagnose_o2o_gradients --alpha 0.1
python -m pytest tests/test_a1_residual_factor.py tests/test_a1_gradient_bridge.py tests/test_a1_diagnostic_hooks.py -q
python -m pytest tests/test_master_model_configs.py -q
python -m pytest tests/test_yolo26_task_matrix.py -q -k train_inference_and_graph_contract
```

probe 使用合成特征的输出能量作为标量，仅测梯度连通性，不代替真实检测 loss、
匹配审计、COCO 验证或训练收益结论。alpha=0 时 one-to-one 输入梯度为 None；
alpha=0.1 时非零，且相对 alpha=1 按比例缩放。

2026-09-24 本地验证环境：Windows、Python 3.13.12、PyTorch 2.11.0+cpu。
新增 26 项测试通过，Master 配置 8 项通过，四任务基础矩阵 4 项通过。
export=True 前向测试不是本轮 ONNX 后端复测；未运行全平台远端 CI。

已知基线问题：当前上游 default.yaml 含两个 stal_min_candidates（3 / 4），
已有唯一键测试失败（该测试文件共 5 项通过、1 项失败）。
本 PR 不修改该配置，也不把继承的失败标为通过；提交前需与维护者协调。
新增 Python 文件 Ruff 通过，7 个 Python 文件格式/编译检查通过；
head.py 整文件 Ruff 的 24 项问题在未改基线上同样存在，无本次新增项。
