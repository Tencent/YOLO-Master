# A3 真动态 DAG 与自定义 dispatch 接口（2026-09-12）

## 状态结论

本轮完成的是**可执行接口基线**，不是 TensorRT 交付：

- 确定性 Top-K 合同已统一到 PyTorch eager、NumPy host 和新导出的 bundle。
- 动态 DAG 已能显式表达普通节点、路由张量依赖和条件专家分支。
- `ConditionalDispatchBackend` 要求后端声明条件执行能力并返回实际专家调用审计。
- CPU NumPy 参考后端已验证只调用选中专家；未选专家 callable 的调用次数为 0。
- masked-dense 被合同禁止，不存在自动回退。

仍未完成：完整 YOLO 图计划生成、CUDA/TensorRT backend、设备内 Top-K、零拷贝张量交换和端到端
P50/P95 性能数据。动态 INT8 在这些 FP32 基础设施闭环前继续冻结。

## 确定性 Top-K 合同

新策略名为 `max_deadband_then_lowest_expert_id`。对每个 Top-K 次序：

1. 找到剩余专家概率最大值；
2. 把与最大值差不超过 `route_tie_tolerance` 的专家放入同一 deadband；
3. 在 deadband 内选择最小专家 ID；
4. 只用该规则决定索引，混合权重仍取原始概率并重新归一化。

旧 bundle 的 `probability_minus_expert_id_times_tolerance` 保留兼容回放。策略名和容差写入 manifest，运行时
不根据环境猜测策略。

这一改动针对本次 173 个 `1e-7` 量级边界翻转，但不能仅凭本地构造输入宣称真实数据漂移已经归零。必须
用同一 checkpoint 和 548 张 VisDrone 重新运行完整路由门禁。

## 动态 DAG 合同

`DynamicDAG` 是拓扑有序的版本化计划，包含：

- `DAGCallableNode`：执行注册表中的普通算子/子图；
- `DAGDispatchNode`：读取输入和 sparse routing weights，只调用所选专家；
- 显式命名输入、输出和中间值；
- `masked_dense_allowed=false` 的硬约束。

`DynamicDAGExecutor` 在启动时检查：依赖必须先产生、输出不能重复、专家数量与 Top-K 合法、runner 必须
存在、backend 必须声明 `conditional_execution=true` 和 `emits_execution_audit=true`。每次运行保存每个
dispatch 节点的实际专家 ID、样本/专家对数量和相对 dense 的缩减率。

## 自定义 backend 接口

`ConditionalDispatchBackend.dispatch(...)` 接收输入、稀疏路由权重、专家 runners 以及路由粒度，返回输出和
`DynamicDispatchAudit`。当前 `NumpyConditionalDispatchBackend` 是 CPU 参考实现，支持 `sample` 与
`spatial_union`。

后续 CUDA/TensorRT backend 必须满足同一合同：

- 未选专家 kernel 不得启动；
- 不得用全专家计算后乘 mask 冒充动态执行；
- 必须记录实际专家集合和样本/专家对；
- 空间路由应进一步支持 token/window 分组，避免 batch union 覆盖全专家；
- 正确性通过后才测 warmup、重复次数、P50/P95、吞吐和显存。

## 本地验证

- Ruff：相关 Python 文件通过。
- `py_compile`：相关实现和测试文件通过。
- NumPy/PyTorch 近似并列 Top-K 合同断言通过。
- 动态 DAG 条件调用断言通过：3 个专家中只调用 ID 0、2，ID 1 调用次数为 0。
- ES-MoE 与 MoT 构造模块的 split ONNX/ORT 对 eager 数值回归通过，manifest 记录新策略。
- 系统默认 pytest 环境因 NumPy 2.3.4 与其 OpenCV 二进制不兼容而无法收集；项目 `yolo_env` 未安装
  pytest，因此本轮用相同环境直接运行等价断言。该环境限制不伪装成 pytest 全套通过。

## 下一次云端验证触发点

代码合并前需要一次 GPU 云端任务：重新导出真实 MoT checkpoint 的 6 个动态块，并对 VisDrone val 548 张
运行 eager 与混合动态路径。门禁至少包括：mAP50-95 差值不超过 `0.5` 个百分点、路由漂移统计、6/6 块执行、
未选专家调用缩减。此任务仍是“GPU PyTorch 专家 + CPU ORT 路由”；只有进入 CUDA/TensorRT backend 阶段
才开始采集部署性能。可直接运行的 Notebook 单元见
[`a3_deterministic_topk_cloud_revalidation_20260912.md`](a3_deterministic_topk_cloud_revalidation_20260912.md)。
