#!/usr/bin/env python3
"""
agent/runtime/cli/stability.py

Model/Query 无关的稳定性校验层。

解决问题：
  切换模型（yolo11n → yolo11s）、切换 VLM provider（gpt-4.1 → qwen-vl）、
  变换 query，响应中的具体数值会变化，但以下三类属性应始终成立：
    1. 结构合约（Schema）  — 必需字段存在且类型正确
    2. 行为语义（Behavior）— status/recovery/guardrail 等业务逻辑不变
    3. 单调性约束（Monotonic）— map50 ∈ [0,1]、token_count ≥ 0 等数学约束

使用方式：
  从 validate.py 的 build_result() 中调用 stability_check(case, payload)，
  把结果追加到 result["stability_checks"] 中。
  也可单独在 CI 脚本中调用 StabilityChecker.run_all(payload, rules)。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _dotted_get(value: Any, path: str) -> Any:
    """支持 dot-notation 的安全取值，路径不存在返回 _MISSING。"""
    _MISSING = object.__new__(object)  # sentinel
    current = value
    for part in path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                return _MISSING
            idx = int(part)
            if idx >= len(current):
                return _MISSING
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        else:
            return _MISSING
    return current


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, object) and type(v).__name__ == "object" and not hasattr(v, "__dict__"))


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _is_list(v: Any) -> bool:
    return isinstance(v, (list, tuple))


def _is_dict(v: Any) -> bool:
    return isinstance(v, dict)


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any] | tuple[Any, ...]:
    return v if isinstance(v, (list, tuple)) else []


# ---------------------------------------------------------------------------
# 单条 Check 结果
# ---------------------------------------------------------------------------

def _ok(rule: str, path: str = "", detail: str = "") -> dict[str, Any]:
    return {"rule": rule, "path": path, "ok": True, "detail": detail}


def _fail(rule: str, path: str = "", detail: str = "") -> dict[str, Any]:
    return {"rule": rule, "path": path, "ok": False, "detail": detail}


# ---------------------------------------------------------------------------
# 规则类型
# ---------------------------------------------------------------------------

class StabilityChecker:
    """
    针对单个 dispatcher 响应 payload，运行一组模型/query 无关的稳定性规则。

    规则分三层：
      schema    — 字段存在性与类型
      behavior  — 业务语义（状态机、recovery、guardrail 等）
      monotonic — 数值约束（范围、非负、非减等）
    """

    def __init__(self, payload: dict[str, Any], skill: str = ""):
        self.payload = payload
        self.skill = skill or str(payload.get("skill", ""))
        self._results: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 公开入口
    # -----------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """运行所有与当前 skill 相关的规则，返回 check 结果列表。"""
        self._results = []
        self._check_universal_schema()
        self._check_universal_behavior()
        if self._get("dry_run") is True:
            return self._results

        if self.skill in ("yolo.train", "yolo.val"):
            self._check_train_val_schema()
            self._check_train_val_monotonic()
        if self.skill == "yolo.train":
            self._check_train_behavior()
        if self.skill in ("yolo.multimodal.infer", "yolo.multimodal.evaluate"):
            self._check_multimodal_schema()
            self._check_multimodal_behavior()
        if self.skill == "yolo.multimodal.evaluate":
            self._check_multimodal_evaluate_schema()
            self._check_multimodal_evaluate_monotonic()
        if self.skill in ("yolo.lora.diagnose",):
            self._check_lora_diagnose_schema()
        if self.skill in ("yolo.eval.peft_compare",):
            self._check_peft_compare_schema()
        if self.skill in ("yolo.pipeline.experiment",):
            self._check_pipeline_schema()

        return self._results

    @classmethod
    def run_all(cls, payload: dict[str, Any], skill: str = "") -> dict[str, Any]:
        """便捷类方法，返回 {passed, failed, total, checks} 汇总。"""
        checker = cls(payload, skill)
        checks = checker.run()
        passed = sum(1 for c in checks if c["ok"])
        return {
            "passed": passed,
            "failed": len(checks) - passed,
            "total": len(checks),
            "checks": checks,
        }

    # -----------------------------------------------------------------------
    # 内部 helper
    # -----------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        return _dotted_get(self.payload, path)

    def _add(self, result: dict[str, Any]) -> None:
        self._results.append(result)

    def _require_field(self, path: str, types: tuple | None = None, rule_prefix: str = "schema") -> bool:
        """字段必须存在，可选类型检查。返回是否通过。"""
        v = self._get(path)
        if v is None and _is_missing(v):
            self._add(_fail(f"{rule_prefix}.required", path, f"field missing: {path}"))
            return False
        if types and not isinstance(v, types):
            self._add(_fail(f"{rule_prefix}.type", path,
                            f"expected {types}, got {type(v).__name__}: {repr(v)[:60]}"))
            return False
        self._add(_ok(f"{rule_prefix}.required", path))
        return True

    def _require_in(self, path: str, allowed: set, rule_prefix: str = "behavior") -> bool:
        v = self._get(path)
        ok = v in allowed
        if ok:
            self._add(_ok(f"{rule_prefix}.enum", path, f"value={v!r}"))
        else:
            self._add(_fail(f"{rule_prefix}.enum", path,
                            f"expected one of {sorted(allowed)}, got {v!r}"))
        return ok

    def _require_range(self, path: str, lo: float, hi: float, rule_prefix: str = "monotonic") -> bool:
        v = _to_float(self._get(path))
        if v is None:
            self._add(_fail(f"{rule_prefix}.range", path, "field missing or non-numeric"))
            return False
        ok = lo <= v <= hi
        if ok:
            self._add(_ok(f"{rule_prefix}.range", path, f"{v} ∈ [{lo}, {hi}]"))
        else:
            self._add(_fail(f"{rule_prefix}.range", path, f"{v} ∉ [{lo}, {hi}]"))
        return ok

    def _require_nonneg(self, path: str, rule_prefix: str = "monotonic") -> bool:
        v = _to_float(self._get(path))
        if v is None:
            self._add(_fail(f"{rule_prefix}.nonneg", path, "field missing or non-numeric"))
            return False
        ok = v >= 0
        if ok:
            self._add(_ok(f"{rule_prefix}.nonneg", path, f"{v} ≥ 0"))
        else:
            self._add(_fail(f"{rule_prefix}.nonneg", path, f"{v} < 0"))
        return ok

    def _require_nonempty(self, path: str, rule_prefix: str = "schema") -> bool:
        v = self._get(path)
        ok = bool(v)
        if ok:
            self._add(_ok(f"{rule_prefix}.nonempty", path))
        else:
            self._add(_fail(f"{rule_prefix}.nonempty", path, f"empty or missing: {v!r}"))
        return ok

    def _require_implies(self, cond_path: str, cond_val: Any,
                         then_path: str, rule: str = "behavior.implication") -> bool:
        """如果 cond_path == cond_val，则 then_path 必须存在且非空。"""
        cond = self._get(cond_path)
        if cond != cond_val:
            self._add(_ok(rule, then_path, f"condition {cond_path}={cond!r} not met, skip"))
            return True
        v = self._get(then_path)
        ok = bool(v)
        if ok:
            self._add(_ok(rule, then_path, f"implied by {cond_path}={cond_val!r}"))
        else:
            self._add(_fail(rule, then_path,
                            f"when {cond_path}={cond_val!r}, {then_path} must be nonempty"))
        return ok

    # -----------------------------------------------------------------------
    # Layer 1: 通用 Schema（所有 skill 共享）
    # -----------------------------------------------------------------------

    def _check_universal_schema(self) -> None:
        """任何 skill 的响应都必须满足的最小字段集合。"""
        self._require_field("skill", (str,))
        self._require_field("status", (str,))
        self._require_field("summary", (str,))
        # manifest 可以是路径字符串或 None，但键必须存在
        if "manifest" not in self.payload:
            self._add(_fail("schema.required", "manifest", "manifest key missing"))
        else:
            self._add(_ok("schema.required", "manifest"))
        # usage 结构
        self._require_field("usage", (dict,))
        usage = _as_dict(self._get("usage"))
        if isinstance(usage, dict):
            self._require_field("usage.tokens", (dict,), rule_prefix="schema")
        # cost_estimate 结构
        self._require_field("cost_estimate", (dict,))

    # -----------------------------------------------------------------------
    # Layer 2: 通用行为语义（所有 skill 共享）
    # -----------------------------------------------------------------------

    def _check_universal_behavior(self) -> None:
        # status 必须是合法枚举
        self._require_in("status",
                         {"ok", "running", "blocked", "failed", "partial"},
                         rule_prefix="behavior")
        # dry_run 时必须有 plan 字段
        is_dry = self._get("dry_run")
        if is_dry is True:
            v = self._get("plan")
            ok = _is_dict(v) and bool(v)
            if ok:
                self._add(_ok("behavior.dry_run_has_plan", "plan"))
            else:
                self._add(_fail("behavior.dry_run_has_plan", "plan",
                                "dry_run=true but plan is missing or empty"))
        # recovery 字段存在时，其 attempted 必须是 bool
        recovery = self._get("recovery")
        if _is_dict(recovery):
            attempted = recovery.get("attempted")
            if not isinstance(attempted, bool):
                self._add(_fail("behavior.recovery_attempted_is_bool", "recovery.attempted",
                                f"expected bool, got {type(attempted).__name__}"))
            else:
                self._add(_ok("behavior.recovery_attempted_is_bool", "recovery.attempted"))
            # recovery.attempted=true 时必须有 from_device / to_device
            if attempted is True:
                for sub in ("from_device", "to_device", "strategy"):
                    v = recovery.get(sub)
                    if v:
                        self._add(_ok("behavior.recovery_fields", f"recovery.{sub}"))
                    else:
                        self._add(_fail("behavior.recovery_fields", f"recovery.{sub}",
                                        "recovery.attempted=true but field missing"))
        # next_actions 如果存在必须是 list
        na = self._get("next_actions")
        if _is_missing(na):
            return
        if na is not None and not _is_list(na):
            self._add(_fail("behavior.next_actions_is_list", "next_actions",
                            f"expected list, got {type(na).__name__}"))
        elif na is not None:
            self._add(_ok("behavior.next_actions_is_list", "next_actions"))

    # -----------------------------------------------------------------------
    # Layer 1/2: train / val
    # -----------------------------------------------------------------------

    def _check_train_val_schema(self) -> None:
        if self._get("status") != "ok":
            return
        self._require_field("evaluation", (dict,))
        eval_block = _as_dict(self._get("evaluation"))
        for key in ("map50", "map50_95", "precision", "recall"):
            path = f"evaluation.{key}"
            v = _to_float(_dotted_get(eval_block, key))
            if v is None:
                self._add(_fail("schema.required", path, "numeric field missing"))
            else:
                self._add(_ok("schema.required", path))
        self._require_field("evaluation.speed_ms", (dict,))
        self._require_field("artifacts", (list,))

    def _check_train_val_monotonic(self) -> None:
        if self._get("status") != "ok":
            return
        for key in ("map50", "map50_95", "precision", "recall"):
            self._require_range(f"evaluation.{key}", 0.0, 1.0)
        # speed_ms 各项非负
        speed = _as_dict(self._get("evaluation.speed_ms"))
        for k in ("preprocess", "inference", "postprocess"):
            v = _to_float(speed.get(k))
            if v is not None:
                ok = v >= 0
                if ok:
                    self._add(_ok("monotonic.nonneg", f"evaluation.speed_ms.{k}", f"{v}"))
                else:
                    self._add(_fail("monotonic.nonneg", f"evaluation.speed_ms.{k}", f"{v} < 0"))

    def _check_train_behavior(self) -> None:
        if self._get("status") != "ok":
            return
        # artifacts 中必须有 checkpoint 类型
        arts = _as_list(self._get("artifacts"))
        has_ckpt = any(isinstance(a, dict) and a.get("kind") == "checkpoint" for a in arts)
        if has_ckpt:
            self._add(_ok("behavior.train_has_checkpoint", "artifacts"))
        else:
            self._add(_fail("behavior.train_has_checkpoint", "artifacts",
                            "no artifact with kind=checkpoint found"))
        # job 字段
        self._require_field("job", (dict,))
        self._require_nonempty("job.save_dir")

    # -----------------------------------------------------------------------
    # Layer 1/2: multimodal（infer + evaluate 共享）
    # -----------------------------------------------------------------------

    def _check_multimodal_schema(self) -> None:
        self._require_field("multimodal", (dict,))
        mm = _as_dict(self._get("multimodal"))
        # vlm 块
        if _is_dict(mm.get("vlm")):
            vlm = mm["vlm"]
            for f in ("status", "model", "api_mode"):
                if f not in vlm:
                    self._add(_fail("schema.required", f"multimodal.vlm.{f}",
                                    "vlm field missing"))
                else:
                    self._add(_ok("schema.required", f"multimodal.vlm.{f}"))

    def _check_multimodal_behavior(self) -> None:
        status = self._get("status")
        # blocked 时必须有 blocked_reason
        if status == "blocked":
            v = self._get("multimodal.vlm.blocked_reason")
            if v:
                self._add(_ok("behavior.blocked_has_reason", "multimodal.vlm.blocked_reason"))
            else:
                self._add(_fail("behavior.blocked_has_reason", "multimodal.vlm.blocked_reason",
                                "status=blocked but blocked_reason missing"))
        # fusion 策略字段
        fusion = _as_dict(self._get("multimodal.fusion"))
        if _is_dict(fusion):
            if "strategy" in fusion:
                self._add(_ok("schema.required", "multimodal.fusion.strategy"))
            # guardrail policy 一致性：add_only 不允许 suppress_count > 0
            policy = fusion.get("policy", "")
            suppress_count = _to_float(fusion.get("suppress_count", 0)) or 0
            if policy == "add_only" and suppress_count > 0:
                self._add(_fail("behavior.fusion_policy_consistency",
                                "multimodal.fusion.suppress_count",
                                f"policy=add_only but suppress_count={suppress_count}"))
            elif policy == "add_only":
                self._add(_ok("behavior.fusion_policy_consistency",
                               "multimodal.fusion.suppress_count",
                               "add_only, suppress_count=0 ✓"))

    # -----------------------------------------------------------------------
    # Layer 1/2: multimodal.evaluate 专有
    # -----------------------------------------------------------------------

    def _check_multimodal_evaluate_schema(self) -> None:
        if self._get("status") not in ("ok", "partial"):
            return
        self._require_field("evaluation", (dict,))
        evl = _as_dict(self._get("evaluation"))
        for f in ("images_processed", "verdict_parse_rate"):
            v = evl.get(f)
            if v is None:
                self._add(_fail("schema.required", f"evaluation.{f}", "field missing"))
            else:
                self._add(_ok("schema.required", f"evaluation.{f}"))
        # metric_guardrail
        self._require_field("metric_guardrail", (dict,))
        mg = _as_dict(self._get("metric_guardrail"))
        if _is_dict(mg):
            if mg.get("selected") not in (None, "yolo_only", "fused_preview"):
                self._add(_fail("behavior.guardrail_selected_enum",
                                "metric_guardrail.selected",
                                f"unknown value: {mg.get('selected')!r}"))
            else:
                self._add(_ok("behavior.guardrail_selected_enum",
                               "metric_guardrail.selected"))

    def _check_multimodal_evaluate_monotonic(self) -> None:
        if self._get("status") not in ("ok", "partial"):
            return
        evl = _as_dict(self._get("evaluation"))
        # verdict_parse_rate ∈ [0, 1]
        vpr = _to_float(evl.get("verdict_parse_rate"))
        if vpr is not None:
            self._require_range("evaluation.verdict_parse_rate", 0.0, 1.0)
        # images_processed ≥ 0
        ip = _to_float(evl.get("images_processed"))
        if ip is not None:
            ok = ip >= 0
            if ok:
                self._add(_ok("monotonic.nonneg", "evaluation.images_processed", f"{ip}"))
            else:
                self._add(_fail("monotonic.nonneg", "evaluation.images_processed", f"{ip} < 0"))
        # metric_preview delta 方向性：若 guardrail 选 fused_preview，delta 必须 >= 0
        mg = _as_dict(self._get("metric_guardrail"))
        if _is_dict(mg) and mg.get("selected") == "fused_preview":
            delta = _to_float(_dotted_get(self.payload, "evaluation.metric_preview.delta.map50_95"))
            if delta is not None and delta < 0:
                self._add(_fail("monotonic.guardrail_fused_implies_positive_delta",
                                "evaluation.metric_preview.delta.map50_95",
                                f"guardrail selected fused_preview but delta={delta} < 0"))
            else:
                self._add(_ok("monotonic.guardrail_fused_implies_positive_delta",
                               "evaluation.metric_preview.delta.map50_95",
                               f"delta={delta}"))

    # -----------------------------------------------------------------------
    # Layer 1: lora.diagnose
    # -----------------------------------------------------------------------

    def _check_lora_diagnose_schema(self) -> None:
        if self._get("status") != "ok":
            return
        self._require_field("lora_diagnose", (dict,))
        diag = _as_dict(self._get("lora_diagnose"))
        for f in ("adapter_path", "layers_analyzed"):
            if f not in diag:
                self._add(_fail("schema.required", f"lora_diagnose.{f}", "field missing"))
            else:
                self._add(_ok("schema.required", f"lora_diagnose.{f}"))
        # effective_rank 必须 ∈ [0, max_rank]
        er = _to_float(diag.get("effective_rank"))
        if er is not None:
            ok = er >= 0
            msg = f"effective_rank={er}"
            if ok:
                self._add(_ok("monotonic.nonneg", "lora_diagnose.effective_rank", msg))
            else:
                self._add(_fail("monotonic.nonneg", "lora_diagnose.effective_rank", msg))

    # -----------------------------------------------------------------------
    # Layer 1: peft_compare
    # -----------------------------------------------------------------------

    def _check_peft_compare_schema(self) -> None:
        if self._get("status") not in ("ok", "partial"):
            return
        self._require_field("peft_compare", (dict,))
        pc = _as_dict(self._get("peft_compare"))
        # ranking 必须是非空 list
        ranking = pc.get("ranking")
        if not _is_list(ranking) or len(ranking) == 0:
            self._add(_fail("schema.nonempty", "peft_compare.ranking",
                            "ranking must be a non-empty list"))
        else:
            self._add(_ok("schema.nonempty", "peft_compare.ranking",
                           f"{len(ranking)} variants"))
            # 每个 ranking 项必须有 name 和 rank_metric_value
            for i, item in enumerate(ranking):
                if not _is_dict(item):
                    self._add(_fail("schema.type", f"peft_compare.ranking.{i}",
                                    "item must be a dict"))
                    continue
                for f in ("name", "rank_metric_value"):
                    if f not in item:
                        self._add(_fail("schema.required", f"peft_compare.ranking.{i}.{f}",
                                        "field missing"))
                    else:
                        self._add(_ok("schema.required", f"peft_compare.ranking.{i}.{f}"))

    # -----------------------------------------------------------------------
    # Layer 1: pipeline
    # -----------------------------------------------------------------------

    def _check_pipeline_schema(self) -> None:
        if self._get("status") not in ("ok", "partial", "failed"):
            return
        self._require_field("pipeline", (dict,))
        pipe = _as_dict(self._get("pipeline"))
        for f in ("stages_requested", "stages_completed", "stages_failed"):
            if f not in pipe:
                self._add(_fail("schema.required", f"pipeline.{f}", "field missing"))
            else:
                self._add(_ok("schema.required", f"pipeline.{f}"))
        # stages_completed + stages_failed <= stages_requested
        req = _to_float(pipe.get("stages_requested", 0)) or 0
        done = _to_float(pipe.get("stages_completed", 0)) or 0
        fail = _to_float(pipe.get("stages_failed", 0)) or 0
        if done + fail <= req:
            self._add(_ok("monotonic.pipeline_stage_counts",
                           "pipeline.stages_*",
                           f"completed={done}+failed={fail}≤requested={req}"))
        else:
            self._add(_fail("monotonic.pipeline_stage_counts",
                            "pipeline.stages_*",
                            f"completed={done}+failed={fail} > requested={req}"))
