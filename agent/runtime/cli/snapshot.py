#!/usr/bin/env python3
"""
agent/runtime/cli/snapshot.py

环境快照与跨环境一致性对比。

解决问题：
  测试环境 → 生产环境迁移时，以下因素可能导致行为漂移：
    - 模型换成更大/更小版本
    - VLM provider 切换（OpenAI → DashScope/Qwen）
    - query/prompt 变化
    - 硬件变化（MPS → CUDA → CPU）

  本模块提供：
    1. EnvironmentSnapshot  — 捕获当前运行环境指纹
    2. PayloadSnapshot      — 从响应 payload 提取「结构摘要」（剔除会漂移的具体值）
    3. SnapshotDiff         — 对比两次快照，分类哪些差异是「预期的」哪些是「异常的」

用法示例：
  # 测试环境录制
  snap = PayloadSnapshot.capture(payload, skill="yolo.train")
  snap.save("agent/logs/snapshots/train_baseline.json")

  # 生产环境回放对比
  baseline = PayloadSnapshot.load("agent/logs/snapshots/train_baseline.json")
  current  = PayloadSnapshot.capture(prod_payload, skill="yolo.train")
  diff = SnapshotDiff.compare(baseline, current)
  assert diff.regression_count == 0, diff.summary()
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _dotted_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 环境指纹
# ---------------------------------------------------------------------------

class EnvironmentSnapshot:
    """捕获当前 Python/Torch/设备环境，用于标注快照采集时的运行上下文。"""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    @classmethod
    def capture(cls) -> "EnvironmentSnapshot":
        import sys
        import platform
        data: dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "machine": platform.machine(),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            import torch
            data["torch"] = torch.__version__
            data["mps_available"] = torch.backends.mps.is_available()
            data["cuda_available"] = torch.cuda.is_available()
        except Exception:
            data["torch"] = None
        try:
            import ultralytics
            data["ultralytics"] = ultralytics.__version__
        except Exception:
            data["ultralytics"] = None
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvironmentSnapshot":
        return cls(d)


# ---------------------------------------------------------------------------
# Payload 结构摘要（剔除会漂移的具体值）
# ---------------------------------------------------------------------------

# 哪些字段是「结构性字段」（必须跨环境一致）
STRUCTURAL_FIELDS: dict[str, type] = {
    "skill": str,
    "status": str,
    "summary": str,
    "manifest": (str, type(None)),
    "usage.tokens.total": int,          # 存在性，不比较具体值
    "cost_estimate.currency": str,
}

# 哪些字段是「行为字段」（枚举、bool — 必须一致）
BEHAVIORAL_FIELDS: dict[str, Any] = {
    "dry_run": bool,
    "recovery.attempted": bool,
    "metric_guardrail.selected": str,
    "multimodal.fusion.policy": str,
}

# 哪些字段是「数值字段」（只检查符号方向和范围，不比较具体值）
NUMERIC_RANGE_FIELDS: dict[str, tuple[float, float]] = {
    "evaluation.map50": (0.0, 1.0),
    "evaluation.map50_95": (0.0, 1.0),
    "evaluation.precision": (0.0, 1.0),
    "evaluation.recall": (0.0, 1.0),
    "evaluation.verdict_parse_rate": (0.0, 1.0),
    "evaluation.images_processed": (0.0, 1e9),
}


class PayloadSnapshot:
    """
    从 dispatcher 响应中提取「模型/query 无关的结构摘要」。

    具体数值（mAP、token count）不保存绝对值，只保存：
      - 字段是否存在
      - 枚举类字段的值（status、policy 等）
      - 数值类字段的「范围是否合法」（bool）
      - artifacts 的 kind 列表（不含路径）
    """

    def __init__(self, skill: str, data: dict[str, Any],
                 env: EnvironmentSnapshot | None = None):
        self.skill = skill
        self.data = data
        self.env = env
        self.captured_at = data.get("_captured_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @classmethod
    def capture(cls, payload: dict[str, Any], skill: str = "",
                env: EnvironmentSnapshot | None = None) -> "PayloadSnapshot":
        skill = skill or str(payload.get("skill", ""))
        summary: dict[str, Any] = {
            "_captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "_skill": skill,
        }

        # 1. 结构字段：记录「存在/不存在」和「类型匹配与否」
        for path, expected_type in STRUCTURAL_FIELDS.items():
            v = _dotted_get(payload, path)
            if expected_type is str:
                summary[f"_exists.{path}"] = isinstance(v, str) and bool(v)
            elif expected_type == (str, type(None)):
                summary[f"_exists.{path}"] = True  # None 也 ok
            else:
                summary[f"_exists.{path}"] = v is not None

        # 2. 行为字段：记录枚举值本身（这些值跨模型应稳定）
        for path, _ in BEHAVIORAL_FIELDS.items():
            v = _dotted_get(payload, path)
            summary[f"_behavior.{path}"] = v

        # 3. 数值字段：只记录「是否在合法范围内」
        for path, (lo, hi) in NUMERIC_RANGE_FIELDS.items():
            v = _to_float(_dotted_get(payload, path))
            if v is None:
                summary[f"_range.{path}"] = None  # 字段缺失
            else:
                summary[f"_range.{path}"] = lo <= v <= hi

        # 4. artifacts 种类列表（不含绝对路径）
        arts = payload.get("artifacts") or []
        if isinstance(arts, list):
            summary["_artifact_kinds"] = sorted(
                {a.get("kind", "?") for a in arts if isinstance(a, dict)}
            )

        # 5. next_actions 存在性
        na = payload.get("next_actions")
        summary["_has_next_actions"] = isinstance(na, list) and len(na) > 0

        # 6. multimodal 块摘要
        mm = payload.get("multimodal")
        if isinstance(mm, dict):
            summary["_mm.has_vlm"] = "vlm" in mm
            summary["_mm.has_fusion"] = "fusion" in mm
            summary["_mm.has_llm_refine"] = "llm_refine" in mm
            vlm = mm.get("vlm") or {}
            summary["_mm.vlm.status"] = vlm.get("status")

        # 7. recovery 摘要
        recovery = payload.get("recovery") or {}
        if isinstance(recovery, dict):
            summary["_recovery.attempted"] = recovery.get("attempted", False)
            if recovery.get("attempted"):
                summary["_recovery.strategy"] = recovery.get("strategy")

        return cls(skill=skill, data=summary, env=env)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.data)
        if self.env:
            payload["_env"] = self.env.to_dict()
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PayloadSnapshot":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        skill = str(data.get("_skill", ""))
        env_data = data.pop("_env", None)
        env = EnvironmentSnapshot.from_dict(env_data) if env_data else None
        return cls(skill=skill, data=data, env=env)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.data)
        if self.env:
            result["_env"] = self.env.to_dict()
        return result


# ---------------------------------------------------------------------------
# 快照对比
# ---------------------------------------------------------------------------

class DiffItem:
    def __init__(self, key: str, baseline: Any, current: Any,
                 is_regression: bool, note: str = ""):
        self.key = key
        self.baseline = baseline
        self.current = current
        self.is_regression = is_regression
        self.note = note

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "baseline": self.baseline,
            "current": self.current,
            "is_regression": self.is_regression,
            "note": self.note,
        }


class SnapshotDiff:
    """
    对比 baseline 快照和 current 快照，分类差异。

    差异分两类：
      - regression（回归）：行为/结构字段不一致，如 status 从 ok 变 failed
      - drift（漂移/预期变化）：数值范围符合约束但具体值不同（换模型正常）
    """

    def __init__(self, items: list[DiffItem], baseline_skill: str, current_skill: str):
        self.items = items
        self.baseline_skill = baseline_skill
        self.current_skill = current_skill

    @property
    def regression_count(self) -> int:
        return sum(1 for it in self.items if it.is_regression)

    @property
    def drift_count(self) -> int:
        return sum(1 for it in self.items if not it.is_regression)

    @classmethod
    def compare(cls, baseline: PayloadSnapshot, current: PayloadSnapshot) -> "SnapshotDiff":
        items: list[DiffItem] = []
        all_keys = set(baseline.data.keys()) | set(current.data.keys())

        for key in sorted(all_keys):
            if key.startswith("_captured_at"):
                continue  # 时间戳差异忽略
            bv = baseline.data.get(key)
            cv = current.data.get(key)

            if bv == cv:
                continue  # 完全一致，跳过

            # 判断是否是 regression
            if key.startswith("_exists.") or key.startswith("_behavior.") or key.startswith("_mm.") or key.startswith("_recovery."):
                # 这些字段必须与 baseline 一致
                is_regression = bv != cv
                note = "structural/behavioral field changed"
            elif key.startswith("_range."):
                # range 字段：从 True/None 变成 False = regression
                if bv is True and cv is False:
                    is_regression = True
                    note = "numeric field went out of valid range"
                elif bv is None and cv is False:
                    is_regression = True
                    note = "numeric field appeared but out of range"
                elif bv is False and cv is True:
                    is_regression = False
                    note = "numeric field recovered into valid range (improvement)"
                else:
                    is_regression = False
                    note = "numeric field presence changed (model/data difference)"
            elif key == "_artifact_kinds":
                # artifact 种类集合不能减少
                bset = set(bv or [])
                cset = set(cv or [])
                missing = bset - cset
                if missing:
                    is_regression = True
                    note = f"artifact kinds disappeared: {missing}"
                else:
                    is_regression = False
                    note = f"artifact kinds changed: {bset} → {cset}"
            elif key == "_has_next_actions":
                is_regression = bv is True and cv is False
                note = "next_actions disappeared" if is_regression else "next_actions presence changed"
            else:
                is_regression = False
                note = "unknown key diff"

            items.append(DiffItem(key=key, baseline=bv, current=cv,
                                  is_regression=is_regression, note=note))
        return cls(items=items,
                   baseline_skill=baseline.skill,
                   current_skill=current.skill)

    def summary(self) -> str:
        lines = [
            f"SnapshotDiff: baseline_skill={self.baseline_skill!r} current_skill={self.current_skill!r}",
            f"  regressions={self.regression_count}  drifts={self.drift_count}  total_diffs={len(self.items)}",
        ]
        for it in self.items:
            tag = "REGRESSION" if it.is_regression else "drift"
            lines.append(f"  [{tag}] {it.key}: {it.baseline!r} → {it.current!r}  ({it.note})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_skill": self.baseline_skill,
            "current_skill": self.current_skill,
            "regression_count": self.regression_count,
            "drift_count": self.drift_count,
            "total_diffs": len(self.items),
            "items": [it.to_dict() for it in self.items],
        }
