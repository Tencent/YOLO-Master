# CLAUDE.md

本项目是 YOLO-Master —— 基于 Ultralytics 构建、深度集成 MoE 的 YOLO 实时目标检测框架。

## 首先阅读

完整的 agent 任务路由、目录职责、高风险区域和验证命令映射见 [`AGENTS.md`](AGENTS.md)。本文件仅补充 Claude 专属注意事项。

## 关键约定

- **代码风格**：Ruff（line-length=120、Google docstring），变更后务必运行 `ruff check` 和 `ruff format --check`
- **测试选择**：按变更区域选择测试，不要每次跑全量。参见 AGENTS.md 的"任务路由"表
- **Doctest**：`pytest --doctest-modules` 默认启用，新增公共函数需附带可运行 doctest
- **配置驱动**：模型架构通过 YAML 定义（`ultralytics/cfg/models/`），不硬编码架构参数
- **MoE 路由安全**：修改 `ultralytics/nn/modules/moe/routers.py` 前必须理解 4-D NCHW 约束和通道校验逻辑
- **PEFT 零修改**：LoRA 微调纯配置驱动，不改架构代码；通过 V-PEFT 编译器注入

## 变更前必做

1. 确认变更区域，在 AGENTS.md "任务路由"表找到对应验证命令
2. 先跑 `ruff check <变更路径>` 确认无 lint 违规
3. 跑对应区域的聚焦测试（非全量）
4. 如修改了 `ultralytics/cfg/`，额外跑 `tests/test_default_config_integrity.py`

## 变更后必做

1. `ruff check` + `ruff format --check` 覆盖所有变更文件
2. 聚焦测试通过
3. 如涉及公共 API，跑 `pytest --doctest-modules ultralytics/<变更模块>/`
4. 如涉及 Agent Skill，跑 `python agent/scripts/validate_yolo_master_skill.py --suite quick`

## 避免

- 不要在 `ultralytics/cfg/default.yaml` 中添加未文档化的参数
- 不要绕过 `ultralytics/nn/modules/moe/` 的路由边界检查
- 不要直接修改 `ultralytics/utils/callbacks/`（coverage omit 区域，变更需手动验证）
- 不要在全量测试中使用 `--slow` 标记（仅供显式调用）
- 不要在 `agent/SKILL.md` 之外硬编码 YOLO CLI 命令字符串

## YOLO 操作

需要执行 YOLO 训练/验证/推理/导出等操作时，阅读 [`agent/SKILL.md`](agent/SKILL.md) 获取 CLI 命令、分发器用法和 MPS/CUDA 设备选择规则。
