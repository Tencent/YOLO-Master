# AGENTS.md

YOLO-Master 是首个深度集成 Mixture-of-Experts (MoE) 的 YOLO 实时目标检测框架，基于 Ultralytics 构建，新增 ES-MoE、MoA、MoT、MoLoRA、Sparse SAHI 和 Agent Skill 系统。

## 快速启动

```bash
pip install -e .                # 安装本地包（开发模式）
yolo version                    # 验证 CLI 可用
```

## 目录结构与职责

| 目录 | 职责 |
|------|------|
| `ultralytics/nn/` | 神经网络核心：MoE/MoA/MoT 模块、路由层、检测头、PEFT |
| `ultralytics/nn/modules/moe/` | MoE 核心模块族（routers、pruning、expert） |
| `ultralytics/nn/peft/` | V-PEFT 编译器、MoLoRA 参数高效微调 |
| `ultralytics/engine/` | 训练/验证/预测/导出/调参引擎 |
| `ultralytics/cfg/` | 模型 YAML、数据集配置、默认参数 |
| `ultralytics/data/` | 数据加载与增强管线 |
| `ultralytics/trackers/` | 多目标跟踪器（BoT-SORT、ByteTrack） |
| `ultralytics/utils/` | 通用工具、回调、导出、Lora 子包 |
| `agent/` | Agent Skill 运行时、脚本、多模态推理管线 |
| `tests/` | 端到端与回归测试套件 |
| `scripts/` | 消融、复现、基准与诊断脚本 |
| `benchmarks/` | 基准测试套件与调度器 |
| `docs/` | MkDocs 文档站点源文件 |
| `wiki/` | 多语言 Wiki 文档仓库 |
| `.github/workflows/` | CI 流水线（CI、Wiki 部署、Model Zoo） |
| `docker/` | 多平台 Dockerfile |
| `examples/` | 社区示例与跨平台部署参考 |

## 任务路由与验证

按变更区域选择对应测试。Lint 在提交前跑一次即可。

```bash
ruff check ultralytics/ tests/ scripts/ agent/
ruff format --check ultralytics/ tests/ scripts/ agent/
codespell
```

| 变更区域 | 验证命令 |
|----------|----------|
| `ultralytics/nn/modules/moe/routers.py` | `pytest tests/test_moe_router_boundaries.py -v` |
| `ultralytics/nn/modules/moe/pruning.py` | `pytest tests/test_moe_dynamic_schedule.py -v` |
| `ultralytics/nn/peft/molora/` | `pytest tests/test_molora_routing_aware_merge.py tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py tests/test_vpeft.py -v` |
| `ultralytics/engine/` | `pytest tests/test_engine.py -v` |
| `ultralytics/cfg/default.yaml` | `pytest tests/test_default_config_integrity.py -v` |
| `ultralytics/cfg/models/` | `pytest tests/test_master_model_configs.py -v` |
| `agent/scripts/run_yolo_master_skill.py` | `python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only` |
| `ultralytics/nn/modules/`（MoE/MoA/MoT 架构） | 对应 `tests/test_moe_*.py` + `tests/test_master_model_configs.py` |
| `ultralytics/nn/peft/`（LoRA） | `tests/test_vpeft.py` + `tests/test_molora_*.py` |
| `ultralytics/data/` | 相关 `tests/test_*.py` + `pytest --doctest-modules ultralytics/` |
| `ultralytics/trackers/` | 相关跟踪测试 |
| `docs/`、`wiki/` | `mkdocs build --strict` |
| `.github/workflows/` | 检查 YAML 语法 + 路径触发器 |
| `scripts/` | 相关消融/诊断脚本 dry-run |

CI P0/P1 回归门：

```bash
pytest tests/test_default_config_integrity.py tests/test_master_model_configs.py --tb=long
pytest tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py \
       tests/test_molora_merge_semantics.py tests/test_adapter_backend_contract.py --tb=long
```

## 代码规范

- **Python 版本**：>= 3.8
- **Linter / Formatter**：Ruff（line-length=120，Google docstring convention）为主
- **Import 排序**：isort（line_length=120，multi_line_output=0）
- **拼写**：codespell（忽略词列表见 `pyproject.toml`）
- **测试**：pytest，`--doctest-modules` 启用，`--slow` 标记慢速测试
- **覆盖率**：coverage，source=`ultralytics/`，omit=`ultralytics/utils/callbacks/*`

## YOLO 操作

CLI 基本操作（train/val/predict/export）见 [`agent/SKILL.md`](agent/SKILL.md) 前 30 行。多模态推理、批评估、Pipeline 等高级用法见 SKILL.md 内对应章节或 `agent/references/` 下的专题文档。
