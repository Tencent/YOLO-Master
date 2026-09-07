# planner_solver_notes.md · V-PEFT / Planner 源码事实笔记

> 用途：为 C3 实验与最终报告提供**可复现引用的源码事实**（哪些护栏存在、哪条路径会审计、LOVO 是什么、如何记录 ΔmAP）。全部结论来自直接读源码（2026-09-07，commit `b0ead2b` 基线）。仓库根 = `ROOT`。
> 事实速查：三 solver（ao 默认 / dco / mip）；cap<8 抛错在 `placement_plan.py:117`；vpeft 接受后跳过 legacy planner audit；LOVO 是回归系数在线校准闭环。

## 1. `lora_planner_solver`：取值与默认

- 默认 **`"ao"`**。YAML：`ROOT/ultralytics/cfg/default.yaml:213`（注释标注取值域 ao, dco, mip）。代码侧同默认：`ROOT/ultralytics/utils/lora/config.py:143` `planner_solver: str = "ao"`。
- solver→类映射（未知值直接 `ValueError`）：`ROOT/ultralytics/utils/lora/api.py:425-433`
  ```python
  solver_name = str(getattr(config, "planner_solver", "ao") or "ao").lower()
  solver_cls = {"ao": AlternatingOptimizationSolver, "dco": DifferentiableOptimizationSolver,
                "mip": MIPRelaxationSolver}.get(solver_name)
  if solver_cls is None: raise ValueError(f"unsupported lora_planner_solver={solver_name!r}")
  ```
  （`mipr` 是 MIPRelaxationSolver 的别称，YAML 取值写作 `mip`。）
- solver 类与超参：
  | solver | 文件:行 | 关键约束 |
  |---|---|---|
  | `AlternatingOptimizationSolver` (AO) | `ultralytics/vpeft/solver.py:213`(init 224-241) | rank_min=4, rank_max=64, rank_step=4 → rank∈[4,8,…,64]；max_iter=15 |
  | `DifferentiableOptimizationSolver` (DCO) | `solver.py:470`(484-538) | 同上 rank 集；max_iter=200；optimize_variant 默认关 |
  | `MIPRelaxationSolver` (MIP) | `solver.py:770`(782-793) | rank_set=[4,8,12,16,32,64]；time_limit_ms=10_000；引擎 OR-Tools SCIP |

## 2. OR-Tools 是可选依赖；缺失时**抛错而非静默切换**

- docstring 明示（`solver.py:778-780`）：未装 OR-Tools 时 `solve()` 抛 `ImportError` 并推荐 `AlternatingOptimizationSolver`。
- 抛出点：`solver.py:906-913` `from ortools.linear_solver import pywraplp` → `ImportError(... "Consider using AlternatingOptimizationSolver ...")`。
- 仅当 OR-Tools 在但 SCIP 无可行解/超时时，内部迭代舍入兜底 `_iterative_rounding_fallback`（`solver.py:915-917, 961-962`）。
- **上层兜底（V-PEFT backend，`api.py:853-876`）**：`ImportError/ModuleNotFoundError`、`ValueError/TypeError`、其它 `Exception` 三类异常 —— `lora_vpeft_strict=True` 一律 **raise**；`strict=False` 才 `_record_vpeft_fallback(...)`（category=dependency/configuration/internal）并把 `planner_backend` 降为 `"legacy"`。
  → 结论：**C3 实验用 strict=True，不会静默降级**，失败即报错，证据链干净。若要"降级护栏演示"可配 strict=False 观察 fallback 记录。

## 3. 层容量护栏与 cap<8 已知缺陷

- 容量定义：`cap = min(in_c, out_c)`（Conv2d 用 channels，Linear 用 features）。硬校验：`ultralytics/vpeft/placement_plan.py:113-118`
  ```python
  capacity = min(...in..., ...out...)
  if rank > capacity: raise ValueError(f"PlacementPlan rank {rank} for {target.name!r} exceeds layer capacity {capacity}")
  ```
- 已知缺陷场景：对 YOLO-Master-EsMoE-N，部分层容量 < AO 最小候选 rank 4，例如
  - `0.conv`：stem 首层（若 planner 选中且 rank>cap 即抛错；默认 `lora_skip_stem=True` 会跳过 stem，见 §4）
  - `{3,6,9,12}.routing.routing_network.2`：MoE 路由的线性层（in/out 很小，cap≈3）
  - `25.dfl.conv`：DFL 头（cap≈1）
- vpeft solver 侧 `SemanticProtectionConstraint` 会把 DFL 标记为 always-protected（不参与放置），故 DFL 报错主要出现在 legacy `validate_model`/目标检测路径而非 vpeft solver。
- **受控处理（沿用 smoke 方案）**：`lora_exclude_modules="routing_network.2, dfl.conv, 0.conv"`（子串匹配，`0.conv` 会连带排除 `10.conv` 等，如实记录）；`lora_skip_stem=True` 默认跳过 stem。不关闭 strict，误伤清单写入结果 evidence。

## 4. 过滤/排除相关参数

- `lora_exclude_modules`（`cfg/default.yaml:239`，list[str]，子串匹配目标名过滤）。
- `lora_skip_stem=True` 默认跳过未归一化的 stem（`default.yaml:245` 注释：防止首层 conv FP16 NaN）。
- `lora_min_channels`（`default.yaml:246`）、`lora_last_n`/`lora_from_layer`/`lora_to_layer`（240-242）、`lora_include_moe`/`lora_include_attention`/`lora_only_backbone`（235-237）。

## 5. Planner 审计日志：文件位置与 JSON 字段

- 写出类：`DecisionAudit`（`ultralytics/utils/lora/planner.py:661-689`），`to_dict()` 字段 = timestamp / model_name / fingerprint / variant / requested_rank / decision_status / recommended_variant / recommended_rank / predicted_delta / refusal_reason / safety_overrides / metadata / evidence / target_modules_count。
- 落盘：`planner.py:707-741`，默认目录 `runs/planner_audit/`（相对进程 cwd），文件名 `planner_audit_<ts>.json`，最多保留 100 个（轮转）。
- 状态值：`PlacementDecision.status` ∈ {ACCEPT, REFUSE, ADAPT}（`planner.py:610-628`）。`predicted_delta` = 回归预测 ΔmAP。
- **重要**：legacy Planner（PEFTPlanner.plan，走 DecisionAudit JSON）仅在 `planner_requested and not vpeft_plan_active` 时执行；**V-PEFT backend 接受放置后 legacy planner 被跳过**（`api.py:883-884`）。故 C3 主策略 vpeft 的"审计证据"来自：
  1. `PlannerResult.from_placement_plan(...)` 挂到 model（`api.py:834-837`）；
  2. `config.target_modules / rank_pattern` 为放置结果（846-847）；
  3. 日志 `[V-PEFT] {status}: selected {n} targets with ranks=...`（849-852）；
  4. strict 失败直接异常，strict=False 才有 `_record_vpeft_fallback`。
  → 采集 vpeft 决策时，从 `args.yaml`（planner 开/后端/strict/exclude/budget）与运行日志的 `[V-PEFT] ...` 行提取，勿依赖 `planner_audit/`（vpeft 路径可能不生成）。

## 6. LOVO：语义、位置、用法

- LOVO 源码概念（全部在 `ultralytics/utils/lora/planner.py`）：
  - `LOVODataCollector` / `LOVOValidator` / `LOVOValidationResult` / `LOVODataPoint`。
  - planner 构造参数 `lovo_collector / lovo_validator / lovo_persist_path`（1413-1415）；collector ≥5 点且未 fit 时 `_maybe_fit_from_lovo` 自动拟合（1461-1479），validator 给出 `lovo_r2 / lovo_rmse / n_samples`。
  - 在线学习闭环：训练后调用 `planner.record_training_result(model, variant, rank, delta_mAP, ...)`（1481-1529）→ collector.add → 下次 plan() 自动 re-fit；配 `lovo_persist_path` 跨 run 持久化。
  - 回归模型：默认 5 系数（beta0=0.0656 截距 …），扩展 12 系数（深度/宽度/头/残差/范数/log(r)/φ_attn²，默认 0，由 LOVO fit 激活），`DEFAULT_COEFFS`（1383-1396）。REFUSE 阈值 `REFUSE_THRESHOLD = -0.05`（1406），按预测 ΔmAP 拒绝。
  - 论文基线：`PAPER_COEFFS`（1399）；Table 2 LOVO metrics accuracy 86.7% / recall 0.944 / F1 0.850（注释 1405）。
- **任务书"记录 ΔmAP/LOVO"的执行口径**：
  1. 每次策略跑完记录实测 `ΔmAP = 策略 mAP − 基线 mAP`（基线 = 不训练/相同随机种子的 cold model 或预训练 zero-shot 值，取 full_sft? 更合理为同一数据下 seed-一致的 base 结果，报告中注明定义）；
  2. 用 `record_training_result` 回填 → 观察 planner 是否 auto re-fit 与 `LOVO R²/RMSE` 日志；
  3. 如无法实例化带 collector 的 planner（训练 CLI 不暴露），则以"evidence 字段记录 predicted_delta vs actual ΔmAP（预测−实测回归护栏）"交付，并在报告中注明与源码 LOVO 的对应关系。

## 7. 训练产物落盘位置（runs/<strategy>）

- `args.yaml`（resolved config，运行器提取 resolved_config 的来源）、`results.csv`（每 epoch 指标，末行为最后状态）、`weights/`、`lora_adapter/`（`lora_save_adapters=True` + `lora_adapter_dir`，`default.yaml:220-221`）。汇总 `summary.json` 由运行器生成（含 gpu 前后快照/exit_code/elapsed）。
- planner audit JSON 位置与 legacy/vpeft 区别见 §5。

## 8. cfg/default.yaml 核心键快照（planner/训练相关）

| 键 | 默认 | 说明 |
|---|---|---|
| lora_r | 0 | 0=关闭；>0 启用 |
| lora_planner_enabled | False | 开 planner |
| lora_planner_solver | "ao" | ao/dco/mip |
| lora_planner_backend | "legacy" | legacy 或 vpeft（opt-in） |
| lora_vpeft_strict | False | strict=True 时内部错一律 raise |
| lora_adapter_budget | (空) | C3 用 2_100_000 |
| lora_include_head | False | head 是否纳入 LoRA |
| lora_exclude_modules | (空) | 子串排除清单（C3 固定） |
| lora_skip_stem | True | 默认跳过 stem |
| lora_min_channels | 0 | 0=禁用 |
| lora_include_moe/attention | False/False | 纳入 MoE/Attention |
| lora_few_shot_mode | False | few-shot 正则（本章节 C3 对比可用性待验证） |
| freeze | (空,int=0) | freeze 前 N 层（full 对照组=0，frozen_backbone=11） |

- 完整参数行见 `cfg/default.yaml:205-295`（lora_*）与 `:39`（freeze）。
