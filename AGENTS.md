# AGENTS.md

YOLO-Master 是首个深度集成 Mixture-of-Experts (MoE) 的 YOLO 实时目标检测框架，基于 Ultralytics 构建，新增 ES-MoE、MoA、MoT、MoLoRA、Sparse SAHI 和 Agent Skill 系统。

## 快速启动

```bash
pip install -e .                # 安装本地包（开发模式）
yolo version                    # 验证 CLI 可用
yolo checks                     # 环境健康检查
```

## 目录结构与职责

| 目录 | 职责 | 何时关注 |
|------|------|----------|
| `ultralytics/nn/` | 神经网络核心：MoE/MoA/MoT 模块、路由层、检测头、PEFT | 修改模型架构、路由逻辑、专家模块 |
| `ultralytics/nn/modules/moe/` | MoE 核心模块族（routers、pruning、expert） | MoE 路由、专家剪枝、动态调度 |
| `ultralytics/nn/peft/` | V-PEFT 编译器、MoLoRA 参数高效微调 | LoRA 配置、适配器合并、PEFT 比较 |
| `ultralytics/engine/` | 训练/验证/预测/导出/调参引擎 | 修改训练流程、验证逻辑、导出格式 |
| `ultralytics/cfg/` | 模型 YAML、数据集配置、默认参数 | 修改模型配置、添加数据集、调整默认值 |
| `ultralytics/data/` | 数据加载与增强管线 | 修改数据预处理、增强策略 |
| `ultralytics/trackers/` | 多目标跟踪器（BoT-SORT、ByteTrack） | 跟踪算法、MOT 集成 |
| `ultralytics/utils/` | 通用工具、回调、导出、Lora 子包 | 工具函数、回调钩子、导出工具 |
| `agent/` | Agent Skill 运行时、脚本、多模态推理管线 | Agent Skill 开发、VLM 融合、CLI 分发器 |
| `tests/` | 端到端与回归测试套件（99 个测试文件） | 添加测试、验证变更 |
| `scripts/` | 消融、复现、基准与诊断脚本 | 实验复现、消融分析、MoE 诊断 |
| `benchmarks/` | 基准测试套件与调度器 | 性能基准、MoLoRA/MoT 调度基准 |
| `docs/` | MkDocs 文档站点源文件 | 文档更新、API 参考 |
| `wiki/` | 多语言 Wiki 文档仓库 | Wiki 内容、MoE/PEFT 文档 |
| `.github/workflows/` | CI 流水线（CI、Wiki 部署、Model Zoo） | CI 配置、部署流程 |
| `docker/` | 多平台 Dockerfile | 容器化部署 |
| `examples/` | 社区示例与跨平台部署参考 | 部署参考、推理示例 |

## 高风险区域

以下区域变更需特别谨慎，务必运行对应验证：

- **`ultralytics/nn/modules/moe/routers.py`** — MoE 路由核心，影响所有专家选择逻辑。验证：`tests/test_moe_router_boundaries.py`
- **`ultralytics/nn/peft/molora/`** — MoLoRA 层与模型，影响参数高效微调。验证：`tests/test_molora_routing_aware_merge.py`、`tests/test_vpeft.py`
- **`ultralytics/nn/modules/moe/pruning.py`** — MoE 剪枝与动态调度。验证：`tests/test_moe_dynamic_schedule.py`
- **`ultralytics/engine/`** — 训练/验证/预测引擎，影响所有任务流程。验证：`tests/test_engine.py`
- **`ultralytics/cfg/default.yaml`** — 默认配置，影响全局行为。验证：`tests/test_default_config_integrity.py`
- **`ultralytics/cfg/models/`** — 模型 YAML 定义。验证：`tests/test_master_model_configs.py`
- **`agent/scripts/run_yolo_master_skill.py`** — Agent Skill 分发器入口。验证：`python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only`

## 验证命令

### Lint（变更后必跑）

```bash
ruff check ultralytics/ tests/ scripts/ agent/          # 代码检查
ruff format --check ultralytics/ tests/ scripts/ agent/ # 格式检查
codespell                                               # 拼写检查
```

### 测试（按变更区域选择）

```bash
# 全量测试（CI 等价）
pytest tests/ -n auto --dist=loadfile --cov=ultralytics/ --cov-report=xml

# MoE 路由与边界
pytest tests/test_moe_router_boundaries.py -v

# MoE 动态调度
pytest tests/test_moe_dynamic_schedule.py -v

# MoLoRA 合并与路由
pytest tests/test_molora_routing_aware_merge.py tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py -v

# V-PEFT 编译器
pytest tests/test_vpeft.py -v

# 配置完整性
pytest tests/test_default_config_integrity.py tests/test_master_model_configs.py -v

# 引擎
pytest tests/test_engine.py -v

# 导出
pytest tests/test_exports.py --export-env base -v

# 慢速测试（默认跳过，需显式启用）
pytest tests/ --slow -v

# Doctest
pytest --doctest-modules ultralytics/
```

### Agent Skill 验证

```bash
# 快速套件（默认）：fast-smoke + dry-run + contract
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only

# 仅契约检查
python agent/scripts/validate_yolo_master_skill.py --suite contract --pretty --summary-only
```

### CI Mixture 回归门

```bash
# P0/P1 回归门（CI 中执行）
pytest tests/test_default_config_integrity.py tests/test_master_model_configs.py --tb=long
pytest tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py \
       tests/test_molora_merge_semantics.py tests/test_adapter_backend_contract.py --tb=long
```

## 任务路由

| 任务类型 | 入口 | 验证 |
|----------|------|------|
| 修改模型架构（MoE/MoA/MoT） | `ultralytics/nn/modules/` | 对应 `tests/test_moe_*.py` + `tests/test_master_model_configs.py` |
| 修改 PEFT/LoRA | `ultralytics/nn/peft/` | `tests/test_vpeft.py` + `tests/test_molora_*.py` |
| 修改训练/验证引擎 | `ultralytics/engine/` | `tests/test_engine.py` + `pytest --doctest-modules ultralytics/` |
| 修改配置 | `ultralytics/cfg/` | `tests/test_default_config_integrity.py` |
| 修改 Agent Skill | `agent/` | `python agent/scripts/validate_yolo_master_skill.py --suite quick` |
| 修改数据加载 | `ultralytics/data/` | 相关 `tests/test_*.py` + doctest |
| 修改跟踪器 | `ultralytics/trackers/` | 相关跟踪测试 |
| 文档更新 | `docs/`、`wiki/` | `mkdocs build --strict`（如安装了 mkdocs） |
| CI 配置变更 | `.github/workflows/` | 检查 YAML 语法 + 路径触发器 |
| 脚本/实验 | `scripts/` | 相关消融/诊断脚本 dry-run |

## 代码规范

- **Python 版本**：>= 3.8
- **Linter**：Ruff（line-length=120，Google docstring convention）
- **格式化**：Ruff format + yapf（pep8 base，column_limit=120）
- **Import 排序**：isort（line_length=120，multi_line_output=0）
- **Docstring**：Google 风格，类型用括号包裹，如 `(bool)`
- **拼写**：codespell（忽略词列表见 `pyproject.toml`）
- **测试**：pytest，`--doctest-modules` 启用，`--slow` 标记慢速测试
- **覆盖率**：coverage，source=`ultralytics/`，omit=`ultralytics/utils/callbacks/*`

## YOLO 操作

对于 YOLO 训练、验证、推理、导出、跟踪、基准测试等操作，参考 [`agent/SKILL.md`](agent/SKILL.md) 获取完整的 Agent Skill 工作流、CLI 命令和分发器用法。

## 环境注意事项

- **Apple Silicon**：训练/验证默认使用 MPS，失败时自动回退 CPU
- **多 GPU**：`device="0,1,2,3"` 指定多卡训练
- **依赖安装**：`pip install -e ".[dev]"` 安装开发依赖（pytest、coverage 等）
- **导出依赖**：`pip install -e ".[export]"` 安装导出依赖（onnx、onnxslim 等）
