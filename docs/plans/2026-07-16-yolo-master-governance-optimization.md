# YOLO-Master 治理与稳定性优化执行方案

> **For Codex:** 按阶段执行本计划；每个阶段完成后运行对应验证命令，并在通过门禁后再进入下一阶段。

**Goal:** 在保持现有 YOLO-Master 研究能力和模型配置兼容性的前提下，修复已确认的 P0/P1 风险，统一 MoE/MoA/MoT/MoLoRA/PEFT 运行时契约，建立可持续的测试、配置、导出和实验治理体系。

**Architecture:** 采用“稳定性优先、兼容层过渡、分阶段收敛”的方案。第一阶段只修复可复现阻断问题和配置漂移；第二阶段建立统一的 routing auxiliary-loss 与 adapter backend 协议；第三阶段再推进真正稀疏 dispatch、V-PEFT 主链路接入和实验/部署优化。旧模块先保留兼容包装，不进行一次性大规模删除。

**Tech Stack:** Python 3.11、PyTorch、pytest、Ruff、PyYAML、PEFT、ONNX/TorchScript optional export backends、现有 Ultralytics trainer/validator/parse_model 体系。

---

## 1. 执行原则

### 1.1 优先级

| 优先级 | 目标 | 进入条件 | 退出条件 |
|---|---|---|---|
| P0 | 修复阻断和数据/模型错误 | 已有可复现失败 | 失败用例通过，且有回归测试 |
| P1 | 统一核心运行时和 IO | P0 全部通过 | 新旧路径结果一致，API 有契约测试 |
| P2 | 性能、稀疏计算和架构收敛 | P1 稳定、基线可重复 | benchmark 达到目标，导出能力有明确声明 |
| P3 | 研究扩展和 V-PEFT 在线化 | P2 的接口稳定 | 有独立实验开关、回滚路径和消融结果 |

### 1.2 默认约束

- 不删除历史 MoE class；先标记为 `stable`、`experimental` 或 `legacy`。
- 不改变默认 `lora_r=0`、`molora_num_experts=0`、`mot_sparse_train=False` 的行为。
- 不把“Top-K 路由”直接宣传为“稀疏计算”；所有 benchmark 同时报告 activated experts 和实际 expert execution。
- 不在没有 parity test 的情况下修改 merge、export 或 checkpoint 格式。
- 每个行为变更必须先写回归测试，再修改实现。
- 每个阶段使用独立 commit；发现失败时只回滚当前阶段 commit。

### 1.3 总体路线

```mermaid
flowchart LR
    A[Phase 0 基线冻结] --> B[Phase 1 P0 修复]
    B --> C[Phase 2 配置与模型治理]
    C --> D[Phase 3 统一运行时契约]
    D --> E[Phase 4 Adapter IO 与 MoLoRA 收敛]
    E --> F[Phase 5 导出与部署矩阵]
    F --> G[Phase 6 真稀疏 dispatch]
    G --> H[Phase 7 V-PEFT 在线接入]
```

推荐执行顺序：Phase 0 → 1 → 2 → 3 → 4 → 5；Phase 6 和 Phase 7 在稳定分支上单独排期，不阻塞前五阶段交付。

## 2. 当前基线与已知问题

### 2.1 当前验证基线

本计划基于 2026-07-16 的代码检查和测试结果：

- MoA/MoT/MoE 核心测试：`146 passed`。
- MoLoRA/PEFT 测试：`156 passed, 6 skipped, 1 failed`。
- Planner/LOVO 相关测试：`106 passed, 10 skipped`。
- 重点模块 `compileall`：通过。
- 可构建配置：
  - `ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml`
  - `ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml`
  - `ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml`
- 当前不可构建：`ultralytics/cfg/models/26/yolo26-master-n.yaml`。

### 2.2 P0/P1 风险清单

| ID | 风险 | 位置 | 优先级 | 目标阶段 |
|---|---|---|---|---|
| R-01 | MoLoRA `.half()` 后 Conv2d dtype mismatch | `ultralytics/nn/peft/molora/layer.py:91` | P0 | Phase 1 |
| R-02 | YOLO26-Master 的 SPPF YAML 参数不匹配 | `ultralytics/cfg/models/26/yolo26-master-n.yaml:25` | P0 | Phase 1 |
| R-03 | MoLoRA adapter 未接入 trainer/model save API | `ultralytics/engine/trainer.py:1605`、`ultralytics/engine/model.py:423` | P1 | Phase 4 |
| R-04 | MoLoRA merge 用均匀专家平均，不等价于动态路由 | `ultralytics/nn/peft/molora/layer.py:519` | P1 | Phase 4 |
| R-05 | registry、`last_aux_loss`、wrapper collector 三套 aux loss 语义 | `ultralytics/utils/loss.py`、MoLoRA/MoA/MoT modules | P1 | Phase 3 |
| R-06 | MoT/MoLoRA 默认不是完整稀疏计算 | `mot.py:859`、`molora/layer.py:475` | P2 | Phase 6 |
| R-07 | `default.yaml` 有重复 key，MoLoRA CLI 字段不完整 | `ultralytics/cfg/default.yaml` | P1 | Phase 2 |
| R-08 | V-PEFT 未进入 `apply_lora()` 主链路 | `ultralytics/vpeft/`、`ultralytics/utils/lora/api.py` | P3 | Phase 7 |
| R-09 | 多代 MoE class 平面注册，稳定性和导出边界不清 | `ultralytics/nn/tasks.py:1736` | P2 | Phase 2/5 |
| R-10 | 导出能力依赖模块和 backend，缺少统一 capability matrix | `tests/test_mixture_export.py`、`engine/exporter.py` | P1 | Phase 5 |

## 3. Phase 0：基线冻结与执行准备

**目标：** 确保后续所有修复都有可比较的基准，并避免把用户已有工作误判为本轮改动。

### Task 0.1：创建治理分支和基线记录

**Files:**

- Create: `docs/plans/2026-07-16-yolo-master-governance-optimization.md`（本文件）
- Create: `docs/governance/baseline-20260716.md`
- Create: `docs/governance/model-registry.yaml`

**Steps:**

1. 从 `main` 创建工作分支：

   ```bash
   git switch -c codex/yolo-master-governance
   ```

2. 记录当前 commit、Python、PyTorch、CUDA/MPS、PEFT、ONNX 版本。
3. 记录以下测试命令和输出摘要：

   ```bash
   python -m pytest -q tests/test_moa.py tests/test_mot.py tests/test_moe.py
   python -m pytest -q tests/test_molora.py tests/test_peft_adapters.py
   python -m pytest -q tests/test_planner.py tests/test_planner_enhancement.py tests/test_planner_integration.py
   python -m compileall -q ultralytics/nn/modules ultralytics/nn/peft ultralytics/utils/lora ultralytics/vpeft
   ```

4. 基线文件只记录事实，不写推测性结论。

**验收：** 基线文件包含 commit、环境、测试结果和已知失败项；后续每个阶段都能与其对比。

### Task 0.2：定义配置和模块分层

**Files:**

- Modify: `ultralytics/nn/modules/moe/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `docs/governance/model-registry.yaml`

**Steps:**

1. 将已有 MoE class 分为 `stable`、`experimental`、`legacy` 三个集合。
2. 只新增元数据和导出，不改变 YAML 解析行为。
3. 在 registry 中记录每个主模型配置的 task、模块、状态、导出状态和最后验证 commit。

**验收：** 现有稳定配置构建结果不变；CI 能检查 stable/experimental 集合无重叠。

## 4. Phase 1：P0 修复

**目标：** 修复当前已复现的阻断问题，建立最小回归保护。

### Task 1.1：修复 MoLoRA dtype 处理

**Files:**

- Modify: `ultralytics/nn/peft/molora/layer.py:91-106`
- Test: `tests/test_peft_adapters.py:619-728`
- Optional create: `tests/test_molora_dtype.py`

**Implementation decision:** 低秩路径必须保证输入与 adapter 权重 dtype 一致；如果启用 float32 accumulation，必须同时将 adapter modules 临时/持久转换为 float32，并将结果 cast 回 base output dtype。不得只转换输入。

**Steps:**

1. 先新增以下参数化测试：

   - float32 CPU。
   - float16 CPU（如果当前 PyTorch backend 不支持某一算子，测试应明确 skip 条件）。
   - float16 CUDA（有 CUDA 时运行）。
   - bfloat16 CPU/CUDA（backend 支持时运行）。
   - autocast 下 Conv2d 和 Linear。

2. 用失败测试复现当前 `expected scalar type Float but found Half`。
3. 实现 dtype policy helper，统一决定：

   ```text
   input dtype -> adapter compute dtype -> output dtype
   ```

4. 验证 forward、backward、merge 和 state_dict 保存后重新加载。

**验收命令：**

```bash
python -m pytest -q tests/test_peft_adapters.py::TestPeftModelDeviceAndDtype
python -m pytest -q tests/test_molora.py tests/test_molora_supplementary.py
```

**停止条件：** 任意 dtype 路径产生非有限输出、参数 dtype 被意外改变、或 base layer 被解冻，停止进入下一任务。

### Task 1.2：修复 YOLO26-Master 配置构建

**Files:**

- Modify: `ultralytics/cfg/models/26/yolo26-master-n.yaml:25`
- Test: Create `tests/test_master_model_configs.py`

**Implementation decision:** 以实际 `SPPF(c1, c2, k=5)` 签名为准；先将该 YAML 的参数改为与同版本官方 YAML 和当前 `parse_model()` 一致的形式，不修改 `SPPF` 公共签名。

**Steps:**

1. 新增配置构建测试，至少覆盖：

   - `yolo26-master-n.yaml`。
   - `yolo26.yaml`。
   - `yolo26-seg.yaml`。
   - 三个 v0.10 MoA/MoT 配置。

2. 对每个配置执行 `YOLO(path, task=...)` 和最小 forward。
3. 检查 `SPPF` 输出通道、Detect 输入通道和 stride。
4. 对同目录中使用旧 SPPF 参数的配置进行扫描；若不在本次范围，登记为 follow-up，而不是静默修改。

**验收命令：**

```bash
python -m pytest -q tests/test_master_model_configs.py
```

### Task 1.3：建立 P0 健康门禁

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`（如需新增 pytest marker）

**Steps:**

1. 在 CI 中新增 `mixture-p0-regression` job。
2. 运行 MoLoRA dtype、模型配置构建和核心 MoE/MoA/MoT smoke tests。
3. 失败时输出完整 traceback，不使用 `tail` 截断唯一错误信息。

**验收：** P0 job 在本地和 CI 均能阻止回归。

## 5. Phase 2：配置、版本和模型治理

**目标：** 消除“配置可以写但不一定生效”的问题，建立稳定/实验/遗留版本边界。

### Task 2.1：清理 `default.yaml`

**Files:**

- Modify: `ultralytics/cfg/default.yaml:180-203`
- Modify: `ultralytics/cfg/default.yaml:262-276`
- Test: Create `tests/test_default_config_integrity.py`

**Steps:**

1. 删除重复的 `lora_tinit`、`lora_tfinal`、`lora_alpha_warmup`。
2. 确认 `moa_aux_gain` 只保留一个定义。
3. 增加：

   ```yaml
   molora_top_k_warmup: 0
   molora_domain_experts:
   molora_freeze_experts:
   ```

4. 明确 `None`、空 list、空 dict 在 `MoLoRAConfig.from_args()` 中的含义。
5. 测试 YAML key 唯一性、MoLoRA dataclass 字段映射和默认值类型。

**验收命令：**

```bash
python -m pytest -q tests/test_default_config_integrity.py
```

### Task 2.2：统一参数来源

**Files:**

- Modify: `ultralytics/nn/tasks.py:1915-1936`
- Modify: `ultralytics/engine/trainer.py:529-671`
- Create: `ultralytics/nn/modules/moe/config.py`
- Test: Create `tests/test_mixture_config_resolution.py`

**Implementation decision:** 保留 YAML 直接构造能力，但将 trainer/CLI 注入抽取为一个显式 `resolve_mixture_config(args, model)` 层；每个 module 在 setup 后只接受一次最终配置。

**Steps:**

1. 定义 `ResolvedMixtureConfig`，至少包含 MoE、MoA、MoT、MoLoRA 四类参数。
2. 规定优先级：显式 module/YAML 参数 > CLI/config 参数 > 安全默认值。
3. 生成 config audit，记录每个 module 最终生效值及来源。
4. 将现有 trainer 注入逻辑迁移到 resolver；旧函数保留兼容包装。
5. 测试同一配置在 YAML-only、CLI override、resume 三种路径下结果一致。

**验收：** 日志能回答“某个 router 的 temperature/balance/z-loss 最终是多少、来自哪里”。

### Task 2.3：建立模型 registry 和稳定性标签

**Files:**

- Create: `docs/governance/model-registry.yaml`
- Modify: `ultralytics/nn/modules/moe/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Test: `tests/test_model_registry.py`

**Registry 字段：**

```yaml
name: yolo-master-v0.10-moa-mot-n
task: detect
status: experimental
blocks: [VisualEnhancedAdaptiveGateMoE, C2fMoA, C2fMoT]
tests: [parse, forward, backward, ddp, export-onnx]
export: {onnx: partial, tensorrt: unverified}
last_verified_commit: <commit>
```

**验收：** 新增模型配置必须在 registry 中声明状态和最小验证集；未声明配置只能作为 legacy/experimental，不得进入 stable smoke job。

## 6. Phase 3：统一 Routing Auxiliary-Loss 契约

**目标：** 解决 registry、`last_aux_loss` 和 wrapper collector 并存造成的漏计/重复计数/stale graph 风险。

### Task 3.1：定义显式 publisher/collector 协议

**Files:**

- Create: `ultralytics/nn/modules/routing_protocol.py`
- Modify: `ultralytics/nn/modules/moe/protocol.py`
- Modify: `ultralytics/nn/modules/moe/_common.py`
- Test: Create `tests/test_routing_aux_contract.py`

**Protocol：**

```python
class RoutingAuxPublisher(Protocol):
    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor: ...
    def routing_snapshot(self) -> dict[str, Any]: ...
    def export_capabilities(self) -> dict[str, Any]: ...
```

**Rules：**

- 训练 forward 只发布一次 graph-connected aux loss。
- eval 不发布 graph-connected aux loss。
- collector 只读 publisher 的 canonical value。
- wrapper 不得默认再次把同一个 child loss 加入总 loss。
- registry 仅作为兼容传输层，不能成为唯一隐式协议。

### Task 3.2：接入 MoE、MoA、MoT

**Files:**

- Modify: `ultralytics/nn/modules/moe/modules.py`
- Modify: `ultralytics/nn/modules/moe/gated.py`
- Modify: `ultralytics/nn/modules/moe/integration.py`
- Modify: `ultralytics/nn/modules/moa/moa.py`
- Modify: `ultralytics/nn/modules/mot/mot.py`
- Modify: `ultralytics/utils/loss.py`

**Steps:**

1. 给每类模块增加 canonical publisher。
2. 让现有 `MOE_LOSS_REGISTRY`、`collect_moa_aux_loss()`、`collect_mot_aux_loss()` 作为兼容 adapter 调用 canonical publisher。
3. 在 collector 中输出 per-kind counts 和 scalar values，便于诊断。
4. 保留现有 independent gains 和 EMA normalization；先不改变数值公式。
5. 增加 nested wrapper 去重测试、DDP local/global gradient 测试和 eval no-write 测试。

**验收命令：**

```bash
python -m pytest -q tests/test_moe.py tests/test_moa.py tests/test_mot.py tests/test_routing_aux_contract.py
python -m pytest -q tests/test_moe_validation_collectives.py tests/test_moe_ddp_fixes.py
```

### Task 3.3：将 MoLoRA 接入同一契约

**Files:**

- Modify: `ultralytics/nn/peft/molora/layer.py`
- Modify: `ultralytics/nn/peft/molora/model.py`
- Modify: `ultralytics/nn/peft/molora/loss.py`
- Test: `tests/test_molora.py`

**Implementation decision:** 在 YOLO trainer 中只走统一 collector；`MoLoRAModel.compute_aux_loss()` 保留为独立 wrapper API，但明确标记为 standalone/manual mode，不能自动和统一 collector 同时生效。

**验收：** `share_moe_registry=False` 不再静默漏计；统一训练路径无重复计数；standalone wrapper 仍可独立使用。

## 7. Phase 4：Adapter IO 与 MoLoRA 收敛

**目标：** 让 LoRA、MoLoRA 和未来 PEFT variant 都能通过统一 adapter API 保存、加载、合并和验证。

### Task 4.1：定义 AdapterBackend 接口

**Files:**

- Create: `ultralytics/utils/lora/backend.py`
- Modify: `ultralytics/utils/lora/io.py`
- Modify: `ultralytics/utils/lora/__init__.py`
- Test: Create `tests/test_adapter_backend_contract.py`

**接口：**

```python
class AdapterBackend(Protocol):
    def can_handle(self, model) -> bool: ...
    def save(self, model, path: str | Path) -> bool: ...
    def load(self, model, path: str | Path) -> bool: ...
    def merge(self, model, *, calibration_data=None) -> bool: ...
    def metadata(self, model) -> dict[str, Any]: ...
```

注册至少两个 backend：标准 LoRA/PEFT、MoLoRA。

### Task 4.2：接入 trainer 和 YOLO model API

**Files:**

- Modify: `ultralytics/engine/trainer.py:1605-1612`
- Modify: `ultralytics/engine/model.py:406-455`
- Test: `tests/test_adapter_backend_contract.py`

**Steps:**

1. 将 `if getattr(..., "lora_enabled")` 改为 backend discovery。
2. 保留 `save_lora_only()` 作为兼容入口，内部委托给 `save_adapters()`。
3. 增加 `save_adapters()`、`load_adapters()`、`merge_adapters()`。
4. 保存 runtime metadata：backend、variant、targets、rank、router config、merge mode、schema version。
5. trainer 在 best、periodic、resume 三种路径都保存正确 adapter artifact。

**验收：**

- 标准 LoRA 的既有 adapter 文件格式仍可加载。
- MoLoRA 能保存独立 adapter checkpoint。
- 不启用任何 adapter 时不会产生空 adapter 目录。

### Task 4.3：明确 MoLoRA merge 语义

**Files:**

- Modify: `ultralytics/nn/peft/molora/layer.py:510-545`
- Modify: `ultralytics/nn/peft/molora/model.py:217-229`
- Create: `tests/test_molora_merge_semantics.py`
- Modify: `docs/LoRA_Quickstart.md`、`docs/molora_guide.md`

**决策：** 默认禁止把动态 MoLoRA 宣称为 exact merge。提供两个明确模式：

1. `merge(mode="uniform")`：仅用于明确的近似/实验用途，并在 metadata 中写入 `approximate=True`。
2. `merge(mode="calibrated", calibration_loader=...)`：用校准数据估计 router 权重，再生成固定 dense delta；metadata 写入校准集摘要、样本数和误差。

后续如果没有可靠 calibration 语义，不提供默认 merge，而是保留动态 router inference。

**Parity tests：**

- dynamic output vs unmerged output：应近似相等。
- unmerge 后恢复原始 base weights。
- calibrated merge 在校准集和 holdout 集分别报告误差。
- export 前后输出误差满足阈值。

## 8. Phase 5：导出、部署与恢复治理

**目标：** 让每个 mixture/adapter 组合都有明确的导出能力和失败方式。

### Task 5.1：建立 capability matrix

**Files:**

- Create: `ultralytics/cfg/export-capability-matrix.yaml`（canonical runtime source）
- Create: `docs/governance/export-capability-matrix.md`（由 canonical YAML 自动生成）
- Create: `ultralytics/utils/export_capabilities.py`
- Modify: `ultralytics/nn/modules/routing_protocol.py`
- Modify: `ultralytics/engine/exporter.py`
- Test: `tests/test_export_capability_matrix.py`、`tests/test_export_preflight.py`

**维度：**

- PyTorch eager FP32/FP16/BF16。
- torch.compile。
- TorchScript script/trace。
- ONNX。
- TensorRT。
- NCNN/MNN/edge C++ path。

每个模块必须声明：`supported`、`dense_fallback`、`requires_merge`、`known_error`。

### Task 5.2：统一 export preflight

**Files:**

- Create: `ultralytics/utils/export_preflight.py`
- Modify: `ultralytics/engine/exporter.py`
- Test: `tests/test_export_preflight.py`

**Steps:**

1. export 前扫描模型中所有 mixture/adapter modules。
2. 如果 backend 不支持动态 Top-K、grid_sample 或 data-dependent dispatch：
   - 有安全 dense fallback 时自动选择并记录。
   - 无安全 fallback 时提前报错，禁止导出后才失败。
3. 输出 JSON preflight report，包含模块、策略、fallback 和风险。

### Task 5.3：恢复机制与 runtime state 分离

**Files:**

- Modify: `ultralytics/engine/trainer.py:1506-1555`
- Modify: `ultralytics/nn/tasks.py:213-221`
- Test: `tests/test_moe_validation_collectives.py`、新增 `tests/test_runtime_state_reset.py`

**Steps:**

1. 将 registry、routing snapshot、debug counters 统一归入非持久 runtime state。
2. checkpoint 只保存可恢复的训练状态；恢复后显式清理 runtime state。
3. DDP rank 间同步“是否进入 recovery”的状态，禁止单 rank 独自继续。
4. 保留现有健康 checkpoint 和 nonfinite diagnostics。

**验收：** 模拟 NaN aux loss、NaN parameter、EMA nonfinite、validation router failure，所有 rank 结果一致且能恢复/明确失败。

## 9. Phase 6：真正稀疏 dispatch 与性能治理

**前置条件：** Phase 1-5 全部通过，且已有稳定输出 parity 基线。

### Task 6.1：MoT grouped sparse dispatch

**Files:**

- Modify: `ultralytics/nn/modules/mot/mot.py:859-913`
- Test: `tests/test_mot.py`
- Create: `tests/test_mot_sparse_parity.py`
- Create: `benchmarks/benchmark_mot_dispatch.py`

**设计：**

- 将 token/sample 按 expert index 打包。
- 每个 expert 只处理实际被选中的 token/sample。
- 使用 deterministic fallback 处理空 expert。
- 对训练 dense、训练 sparse、eval sparse、export dense 分别实现并测试。

**验收指标：**

- eval sparse 与 dense 输出误差低于设定阈值。
- 在目标 batch/分辨率上实际 expert FLOPs 至少下降 20%，否则不合并为默认路径。
- sparse path 不改变 router gradient 和 aux loss 统计。

### Task 6.2：MoLoRA grouped dispatch

**Files:**

- Modify: `ultralytics/nn/peft/molora/layer.py:460-498`
- Test: `tests/test_molora.py`
- Create: `benchmarks/benchmark_molora_dispatch.py`

**设计：**

- 按 expert 分组 batch index。
- 每个 expert 只执行被选择的样本。
- 保留小 batch 下的 dense fast path，避免 index/gather 开销超过收益。
- 对 Conv2d 和 Linear 分开 benchmark。

### Task 6.3：建立性能门禁

**Files:**

- Create: `benchmarks/mixture_baselines.yaml`
- Create: `docs/governance/performance-gates.md`
- Modify: `.github/workflows/ci.yml`（只在可用硬件 runner 上启用）

**必须记录：** 参数量、显存、p50/p95/p99 latency、实际 expert calls、FLOPs、吞吐和 mAP parity。

## 10. Phase 7：V-PEFT 在线化（独立项目轨）

**前置条件：** Phase 4 的 AdapterBackend 和 Phase 2 的 module registry 已稳定。

### Task 7.1：Graph-to-module 映射契约

**Files:**

- Modify: `ultralytics/vpeft/graph.py`
- Create: `ultralytics/vpeft/adapters.py`
- Test: Create `tests/test_vpeft_mapping.py`

**要求：** graph node 必须保存稳定的 `module_name`、operator、semantic role、shape、parent block 和 fingerprint；不得只依赖模块顺序或模糊 name heuristic。

### Task 7.2：Solver 结果落地

**Files:**

- Modify: `ultralytics/vpeft/solver.py`
- Modify: `ultralytics/vpeft/policy.py`
- Modify: `ultralytics/utils/lora/api.py`
- Test: Create `tests/test_vpeft_apply_plan.py`

**Steps:**

1. solver 输出 placement/rank/variant。
2. adapter orchestration 将其转换为 target modules 和 per-module rank。
3. 先以 `planner_backend="vpeft"` opt-in，不改变默认 Planner。
4. 记录 solver、约束、预算、fallback 和最终 adapter metadata。
5. OR-Tools 不可用时明确降级到 AO，并把降级写入 audit。

### Task 7.3：V-PEFT 可靠性门禁

**验收：**

- 同一 graph 多次 solve 结果可复现。
- hard constraints 永不违反。
- adapter 参数量不超过预算。
- target modules 与实际模型一一对应。
- resume、save/load、DDP 和 export preflight 不丢失 placement 信息。

## 11. CI、测试和发布门禁

### 11.1 推荐 CI job

| Job | 内容 | 阻断级别 |
|---|---|---|
| `lint-and-import` | Ruff、import、配置 schema | 阻断 |
| `mixture-p0-regression` | MoLoRA dtype、YOLO26 build、核心 smoke | 阻断 |
| `mixture-core` | MoE/MoA/MoT 全核心单测 | 阻断 |
| `peft-core` | LoRA/MoLoRA/adapter IO | 阻断 |
| `planner-core` | Planner/LOVO/V-PEFT mapping | 阻断 |
| `ddp-cpu-smoke` | 2 rank Gloo | 阻断 |
| `export-preflight` | 轻量 capability/preflight | 阻断 |
| `export-backend` | ONNX/TensorRT/MNN/NCNN | 非阻断或硬件专用 |
| `long-ablation` | 多 seed 训练和性能 | 夜间任务 |

### 11.2 发布前必须通过

```bash
python -m pytest -q tests/test_moa.py tests/test_mot.py tests/test_moe.py
python -m pytest -q tests/test_molora.py tests/test_molora_supplementary.py tests/test_peft_adapters.py
python -m pytest -q tests/test_planner.py tests/test_planner_enhancement.py tests/test_planner_integration.py
python -m pytest -q tests/test_master_model_configs.py tests/test_routing_aux_contract.py tests/test_adapter_backend_contract.py
python -m compileall -q ultralytics
ruff check ultralytics/ --select E,F,W --ignore E501
```

### 11.3 性能和数值门禁

- FP32/AMP forward 输出 finite。
- 训练第一步和恢复后第一步 loss finite。
- MoE/MoA/MoT/MoLoRA aux loss 不重复计数。
- eval 不写 graph-connected aux loss。
- adapter save/load 后输出 parity 在阈值内。
- merge/unmerge 后 base weight 恢复。
- export preflight 不允许未声明的动态控制流进入不支持 backend。

## 12. 里程碑与时间安排

时间是按连续开发工作日估算，实际应以门禁为准。

| 里程碑 | 内容 | 预计投入 |
|---|---|---:|
| M0 | 基线、分支、registry 草案 | 0.5-1 天 |
| M1 | MoLoRA dtype + YOLO26 配置修复 | 1-2 天 |
| M2 | default.yaml、配置 resolver、模型 registry | 1-2 天 |
| M3 | auxiliary-loss canonical contract | 2-4 天 |
| M4 | AdapterBackend、MoLoRA save/load/merge 语义 | 2-4 天 |
| M5 | export preflight、capability matrix、recovery state | 2-3 天 |
| M6 | sparse dispatch benchmark/prototype | 3-6 天 |
| M7 | V-PEFT online opt-in | 4-8 天 |

建议将 M0-M5 作为第一期稳定治理版本；M6-M7 作为第二期性能/研究增强版本。

## 13. 回滚和故障处理

### 13.1 变更隔离

推荐 commit 顺序：

```text
chore(governance): record baseline
fix(molora): make low-rank path dtype safe
fix(config): repair master model and default yaml drift
feat(config): add resolved mixture config audit
feat(routing): add canonical auxiliary-loss contract
feat(peft): add adapter backend orchestration
fix(molora): define merge semantics and adapter persistence
feat(export): add mixture export preflight
perf(mot): add grouped sparse dispatch
feat(vpeft): add opt-in solver application
```

### 13.2 回滚规则

- P0 修复引起模型输出变化：立即回滚实现 commit，保留失败测试和诊断日志。
- P1 contract 改造引起 loss 变化：切回兼容 adapter，比较 old/new auxiliary loss 分解后再继续。
- adapter schema 变更：必须提供 reader 兼容旧 schema，不能直接覆盖旧文件。
- merge 行为变化：默认保持 dynamic inference，不自动启用新 merge。
- sparse dispatch 低于性能门禁：保留为 opt-in，不进入默认路径。

### 13.3 失败模式

| 失败 | 处理 |
|---|---|
| 单 rank NaN | 全局同步 recovery flag，所有 rank 回退到健康 checkpoint |
| registry stale graph | forward 起始清理，collector 校验 step id，拒绝旧 step loss |
| dtype 不兼容 | preflight 报告 compute/input/output dtype，提前停止而非隐式 cast |
| adapter 结构不匹配 | schema/structure 校验失败，禁止部分加载 |
| export 不支持动态路由 | 选择声明过的 dense fallback，否则明确报错 |
| V-PEFT 无 OR-Tools | 降级 AO，并记录 `solver_fallback=true` |

## 14. 架构决策记录（计划阶段）

### ADR-001：采用分阶段兼容治理，而非一次性重写

**Status:** Proposed

**Context:** 项目存在多代 MoE、MoA、MoT 和 PEFT 实现，历史 YAML/权重仍需要保持可读。一次性重写会扩大 checkpoint、实验和部署回归面。

**Decision:** 新增 canonical protocol 和 backend orchestration，旧实现通过兼容 adapter 接入；只有在连续两个版本周期无使用后才考虑删除 legacy 实现。

**Consequences:** 稳定迁移成本较低，但短期内仍需维护兼容层。

**Alternatives:** 一次性重构；风险是实验复现和旧 checkpoint 断裂，因此不采用。

### ADR-002：动态 MoLoRA 默认不做 exact merge

**Status:** Proposed

**Context:** 动态 router 输出依赖输入，均匀平均专家 delta 无法等价重现训练/推理行为。

**Decision:** 默认保留动态 router；merge 只能显式选择 uniform approximate 或 calibrated approximate，并在 metadata 中声明。

**Consequences:** 部署可能保留 adapter/router 开销，但不会静默改变模型语义。

**Alternatives:** 强制均匀平均 merge；实现简单但有精度风险，因此不采用。

### ADR-003：统一 adapter backend，而不是继续增加 `*_enabled` 分支

**Status:** Proposed

**Context:** trainer/model 当前通过 `lora_enabled` 判断保存和 merge，MoLoRA 使用 `molora_enabled`，后续 variant 会继续增加分支。

**Decision:** 使用 `AdapterBackend` discovery 和统一 `save_adapters/load_adapters/merge_adapters` API。

**Consequences:** 需要一次 IO 迁移和 schema 版本管理，但后续 variant 接入成本下降。

**Alternatives:** 在 trainer 中继续添加属性分支；短期简单，长期会继续扩大耦合，因此不采用。

### ADR-004：V-PEFT 先以 opt-in backend 接入

**Status:** Proposed

**Context:** V-PEFT 的 graph/policy/solver 已有较大实现，但当前不是主链路，且 OR-Tools、mapping、resume/export 仍需验证。

**Decision:** 通过 `planner_backend="vpeft"` 显式开启，默认仍使用现有 Planner；完成 mapping、预算、约束和 parity 门禁后再评估默认化。

**Consequences:** 研究能力可以提前使用，生产默认行为保持稳定。

## 15. 最终交付标准

第一期治理版本只有同时满足以下条件才算完成：

1. P0 测试全部通过，MoLoRA dtype regression 已覆盖。
2. YOLO26-Master 和稳定模型 registry 中声明的配置均能 parse + forward。
3. `default.yaml` 无重复 key，MoLoRA CLI 字段完整。
4. MoE/MoA/MoT/MoLoRA auxiliary loss 有唯一 canonical collector，nested wrapper 不重复计数。
5. LoRA 和 MoLoRA 都能独立 save/load，且 metadata 有 schema/version/structure 校验。
6. MoLoRA merge 语义和误差边界在文档及测试中明确。
7. export preflight 能在导出前识别 unsupported dynamic routing。
8. DDP CPU smoke、AMP smoke、checkpoint recovery smoke 均通过。
9. 所有新增默认行为都有回滚开关或兼容路径。

第二期增强版本才要求：

- MoT/MoLoRA 真正 grouped sparse dispatch 达到性能门禁。
- V-PEFT opt-in placement 能落地到 adapter backend。
- 多 seed、多数据集、多 backend 的结果写入结构化实验 manifest。
