#!/usr/bin/env python3
"""F11 · T6R.5 机制分析脚本(负结果定位)。

依据 06 文档 §10.3(任务书允许负结果,但必须给证据链):
  - 软目标质量:JS 距离 vs 阈值
  - 同质化:load_std vs 阈值
  - 温度:foundation.router_loss_weight 提取
  - 坍塌:专家负载极差 / 单专家 > 90% 标记

输入:
  --summary     runs/real/t4_ablation/real_100ep/f11_ablation_summary.json
  --routing     runs/real/t5_routing/routing_analysis.json
  --student-routing 可选(若 T5 已生成学生侧 routing;否则跳过)
  --out         输出报告路径

用法:
  python scripts/analyze_f11_mechanism.py ^
    --summary runs/real/t4_ablation/real_100ep/f11_ablation_summary.json ^
    --routing runs/real/t5_routing/routing_analysis.json ^
    --out runs/real/t6_extension/mechanism/t6r5_mechanism_report.json

参考:
  - 06-F11全部任务分析与设计.md §10.3(负结果的处理)
  - 08-F11结果汇总 §8.5(5 指标 vs 6 文档判读线)

文档版本:v1.0(2026-09-06)| Owner:张伟林(Zviolin)| F11 T6R.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> list[dict]:
    """读取 JSON 文件(列表形式)。"""
    if not path.exists():
        print(f"[F11][T6R.5][WARN] 缺失: {path}")
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"格式错误: {path} 应为 list,实为 {type(data).__name__}")
    return data


def find_by_name(rows: list[dict], name: str) -> dict | None:
    """按 name / run 字段匹配。"""
    for r in rows:
        if r.get("name") == name or r.get("run") == name:
            return r
    return None


def analyze_dimension(
    dim_name: str,
    a_metric: float | None,
    c_metric: float | None,
    delta_threshold: float,
    inverse: bool = False,
) -> dict:
    """单维度分析:返回证据项。

    inverse=True 表示"反向越好"(如 load_std / js_mean 越小越好)
    """
    if a_metric is None or c_metric is None:
        return {
            "dimension": dim_name,
            "status": "MISSING_DATA",
            "a": a_metric,
            "c": c_metric,
            "delta": None,
            "judgment": "数据缺失,无法判定",
        }
    delta = c_metric - a_metric
    if inverse:
        passed = delta < -delta_threshold
    else:
        passed = delta > delta_threshold
    return {
        "dimension": dim_name,
        "status": "PASS" if passed else "FAIL",
        "a": a_metric,
        "c": c_metric,
        "delta": delta,
        "judgment": (
            f"C vs A Δ={delta:+.4f} "
            f"{'优于' if (delta < 0 if inverse else delta > 0) else '劣于'} A "
            f"(阈值 {'<' if inverse else '>'} {delta_threshold:+.4f})"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="F11 · T6R.5 机制分析(负结果定位)")
    parser.add_argument("--summary", required=True, help="三方对照 summary json")
    parser.add_argument("--routing", required=True, help="T5R 路由分析 json")
    parser.add_argument("--student-routing", default=None, help="(可选) 学生侧路由分析 json")
    parser.add_argument("--out", required=True, help="输出报告 json")
    args = parser.parse_args()

    summary = load_json(Path(args.summary).resolve())
    routing = load_json(Path(args.routing).resolve())
    student_routing = (
        load_json(Path(args.student_routing).resolve()) if args.student_routing else []
    )

    # 取 A 组与 C 组(可能 name 不同,统一匹配)
    a_summary = find_by_name(summary, "a-baseline")
    c_summary = find_by_name(summary, "c-router-kd-3") or find_by_name(summary, "c-router-kd")
    a_routing = find_by_name(routing, "A-baseline")
    c_routing = find_by_name(routing, "C-router-kd")

    if not (a_summary and c_summary and a_routing and c_routing):
        missing = []
        if not a_summary:
            missing.append("summary.a-baseline")
        if not c_summary:
            missing.append("summary.c-router-kd*")
        if not a_routing:
            missing.append("routing.A-baseline")
        if not c_routing:
            missing.append("routing.C-router-kd")
        raise SystemExit(f"关键数据缺失: {missing}")

    report: dict = {
        "task": "T6R.5-mechanism-analysis",
        "version": "v1.0",
        "date": "2026-09-06",
        "evidence_chain": [],
        "verdict": {},
    }

    # 维度 1:软目标质量(JS 距离 — 越小越接近教师分布,阈值 0.05)
    js_mean = c_routing.get("js_mean")
    js_std = c_routing.get("js_std")
    if js_mean is None:
        status = "MISSING_DATA"
        judgment = "数据缺失,无法判定"
    else:
        status = "PASS" if js_mean < 0.05 else "FAIL"
        std_str = f"{js_std:.4f}" if js_std is not None else "N/A"
        threshold_msg = "< 0.05 阈值 → 软目标对齐 OK" if status == "PASS" else ">= 0.05 阈值 → 软目标偏离"
        judgment = f"C 组 JS={js_mean:.4f} (std={std_str}) {threshold_msg}"
    report["evidence_chain"].append(
        {
            "dimension": "软目标质量(JS 距离:越小越好,阈值 0.05)",
            "status": status,
            "a": None,
            "c": js_mean,
            "delta": None,
            "judgment": judgment,
        }
    )

    # 维度 2:同质化(load_std:越小越同质)
    a_load_std = a_routing.get("load_std")
    c_load_std = c_routing.get("load_std")
    report["evidence_chain"].append(
        analyze_dimension(
            dim_name="同质化(load_std:越小越均衡)",
            a_metric=a_load_std,
            c_metric=c_load_std,
            delta_threshold=0.05,
            inverse=True,
        )
    )

    # 维度 3:温度坍塌(switch_rate 越低 → 单 expert 越固化)
    a_switch = a_routing.get("top1_switch_rate")
    c_switch = c_routing.get("top1_switch_rate")
    report["evidence_chain"].append(
        analyze_dimension(
            dim_name="路由切换稳定性(switch_rate 越高越平滑)",
            a_metric=a_switch,
            c_metric=c_switch,
            delta_threshold=0.05,
            inverse=False,
        )
    )

    # 维度 4:坍塌检查(C 组专家负载极差)
    c_load_ratio = c_routing.get("load_ratio") or []
    collapse_max = max(c_load_ratio) if c_load_ratio else None
    collapse_min = min(c_load_ratio) if c_load_ratio else None
    report["evidence_chain"].append(
        {
            "dimension": "坍塌检查(单专家选中率 < 90% 通过)",
            "status": (
                "FAIL"
                if collapse_max is not None and collapse_max > 0.9
                else "PASS"
                if collapse_max is not None
                else "MISSING_DATA"
            ),
            "c_load_ratio": c_load_ratio,
            "c_max_load": collapse_max,
            "c_min_load": collapse_min,
            "judgment": (
                f"C 组专家负载 {collapse_max:.1%} / {collapse_min:.1%} "
                f"{'→ 已坍塌' if collapse_max and collapse_max > 0.9 else '→ 未坍塌'}"
                if collapse_max is not None
                else "数据缺失"
            ),
        }
    )

    # mAP 维度(从 summary 拿)
    a_map = a_summary.get("mAP50-95")
    c_map = c_summary.get("mAP50-95")
    report["evidence_chain"].append(
        analyze_dimension(
            dim_name="mAP 提升(C ≥ A - 0.5% 通过)",
            a_metric=a_map,
            c_metric=c_map,
            delta_threshold=-0.005,
            inverse=False,
        )
    )

    # 汇总判定
    failures = [e for e in report["evidence_chain"] if e.get("status") == "FAIL"]
    passes = [e for e in report["evidence_chain"] if e.get("status") == "PASS"]
    missing = [e for e in report["evidence_chain"] if e.get("status") == "MISSING_DATA"]
    report["verdict"] = {
        "total": len(report["evidence_chain"]),
        "pass": len(passes),
        "fail": len(failures),
        "missing": len(missing),
        "fail_dimensions": [e["dimension"] for e in failures],
        "conclusion": (
            "KD 有效"
            if not failures
            else f"KD 部分失效({len(failures)}/{len(report['evidence_chain'])} 维度未通过): "
            + "; ".join(e["dimension"] for e in failures)
        ),
    }

    # 落盘
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台输出
    print(f"\n[F11][T6R.5] ==== 机制分析报告 ====")
    print(f"输入:summary={args.summary}")
    print(f"输入:routing={args.routing}")
    print(f"输入:student_routing={args.student_routing or '(未提供)'}")
    print(f"输出:{out}\n")
    for e in report["evidence_chain"]:
        print(f"  [{e['status']:<6}] {e['dimension']}")
        print(f"           判定: {e['judgment']}")
    print(
        f"\n[F11][T6R.5] 结论:{report['verdict']['conclusion']}\n"
        f"             通过 {report['verdict']['pass']}/{report['verdict']['total']} 维度,"
        f"缺失 {report['verdict']['missing']}"
    )
    print(f"\n[F11][T6R.5] [OK] 报告已保存:{out}")


if __name__ == "__main__":
    main()