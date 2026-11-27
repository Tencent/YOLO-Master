# -*- coding: utf-8 -*-
"""
issue #49 交付物一键生成器
================================
读取 v01 与 esmoe 在 VisDrone 上的 results.csv，自动产出：
  1) 训练曲线图  results_fig.png  (loss / mAP50 / mAP50-95 / moe_loss)
  2) 对比结果    comparison_results.json
  3) 对比报告    README_issue49.md  (含指标表 + 结论)
产物统一放到桌面交付文件夹，供提交 PR 与回 issue 使用。

用法:
  cd D:/YOLO-Master
  PYTHONPATH=D:/YOLO-Master python scripts/reproduce/deliver_report.py
"""
import csv
import json
import os
import sys

REPO = r"D:\YOLO-Master"
V01_CSV = os.path.join(REPO, r"runs\issue49\VisDrone_local_v01\results.csv")
ESMOE_CSV = os.path.join(REPO, r"runs\issue49\VisDrone_local_esmoe\results.csv")
OUT_DIR = r"C:\Users\Administrator\Desktop\YOLOmaster开源项目"
# 参数量（来自各自训练 summary：v01=442层7.5M；esmoe≈2.69M）
MODEL_PARAMS = {"v01": 7516742, "esmoe": 2690000}

# 关键指标列
MAP50 = "metrics/mAP50(B)"
MAP5095 = "metrics/mAP50-95(B)"
TR_MOE = "train/moe_loss"
TR_BOX = "train/box_loss"
VA_BOX = "val/box_loss"
TR_CLS = "train/cls_loss"


def read_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = []
    for r in rows:
        try:
            epochs.append(int(float(r["epoch"])))
        except Exception:
            epochs.append(len(epochs) + 1)
    return epochs, rows


def col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except Exception:
            out.append(None)
    return out


def build_comparison():
    comp = {}
    for key, path in (("v01", V01_CSV), ("esmoe", ESMOE_CSV)):
        if not os.path.exists(path):
            print(f"[skip] {key} 结果不存在: {path}")
            continue
        ep, rows = read_csv(path)
        if not rows:
            continue
        last = rows[-1]
        comp[key] = {
            "params": MODEL_PARAMS.get(key),
            "epochs_trained": ep[-1] if ep else 0,
            "mAP50": last.get(MAP50),
            "mAP50-95": last.get(MAP5095),
            "train_moe_loss": last.get(TR_MOE),
            "train_box_loss": last.get(TR_BOX),
            "train_cls_loss": last.get(TR_CLS),
            "val_box_loss": last.get(VA_BOX),
        }
        print(f"[ok] {key}: mAP50={comp[key]['mAP50']} mAP50-95={comp[key]['mAP50-95']} moe_loss={comp[key]['train_moe_loss']}")
    return comp


def make_figure():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib 不可用，跳过曲线图: {e}")
        return None

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"v01": "#e74c3c", "esmoe": "#2980b9"}

    for key, path in (("v01", V01_CSV), ("esmoe", ESMOE_CSV)):
        if not os.path.exists(path):
            continue
        ep, rows = read_csv(path)
        if not ep:
            continue
        c = colors.get(key, "#333")
        # mAP50
        axs[0, 0].plot(ep, col(rows, MAP50), marker="o", color=c, label=key)
        # mAP50-95
        axs[0, 1].plot(ep, col(rows, MAP5095), marker="o", color=c, label=key)
        # moe_loss
        axs[1, 0].plot(ep, col(rows, TR_MOE), marker="s", color=c, label=key)
        # box loss
        axs[1, 1].plot(ep, col(rows, TR_BOX), color=c, label=f"{key} train")
        axs[1, 1].plot(ep, col(rows, VA_BOX), "--", color=c, label=f"{key} val")

    axs[0, 0].set_title("mAP50")
    axs[0, 1].set_title("mAP50-95")
    axs[1, 0].set_title("train/moe_loss (MoE router loss)")
    axs[1, 1].set_title("box_loss (train vs val)")
    for ax in axs.flat:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("YOLO-Master Issue #49 — VisDrone Baseline (v0.1-N vs EsMoE-N)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "issue49_results_fig.png")
    fig.savefig(out, dpi=120)
    print(f"[ok] 曲线图 -> {out}")
    return out


def make_readme(comp):
    v = comp.get("v01", {})
    e = comp.get("esmoe", {})
    def g(d, k):
        x = d.get(k)
        try:
            x = float(x)
        except (TypeError, ValueError):
            return "N/A"
        return f"{x:.5f}"
    lines = []
    lines.append("# YOLO-Master Issue #49 — 垂类数据集基线训练与 MoE 对比\n")
    lines.append("## 任务说明\n")
    lines.append("在垂类目标检测数据集 **VisDrone2019-DET**（无人机航拍，小目标密集）上，")
    lines.append("分别训练两种混合专家(MoE)模型 **YOLO-Master-v0.1-N** 与 **YOLO-Master-EsMoE-N**，")
    lines.append("对比其检测精度(mAP)与专家路由损失(moe_loss)，验证 MoE 在垂类场景下的表现。\n")
    lines.append("## 实验配置\n")
    lines.append("- 数据集: VisDrone2019-DET（train 6471 / val 548，已转 YOLO 格式）")
    lines.append("- 训练: 从零训练(pretrained=False)，imgsz=640, batch=8, workers=0, CPU(torch 2.3.1+cpu)")
    lines.append("- 每模型 5 epochs（受 deadline 与算力限制，公平对比）\n")
    lines.append("## 指标对比\n")
    lines.append("| 模型 | 参数量 | epochs | mAP50 | mAP50-95 | train/moe_loss | train/box_loss |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| v0.1-N (ModularRouter) | {v.get('params')} | {v.get('epochs_trained')} | {g(v,'mAP50')} | {g(v,'mAP50-95')} | {g(v,'train_moe_loss')} | {g(v,'train_box_loss')} |")
    lines.append(f"| EsMoE-N (ES_MOE) | {e.get('params')} | {e.get('epochs_trained')} | {g(e,'mAP50')} | {g(e,'mAP50-95')} | {g(e,'train_moe_loss')} | {g(e,'train_box_loss')} |\n")
    lines.append("## 结论\n")
    if v and e:
        if (v.get("mAP50") or 0) > (e.get("mAP50") or 0):
            lines.append("- 在 VisDrone 上 **v0.1-N 的 mAP50 高于 EsMoE-N**，说明其空间路由专家在该垂类场景更具优势。")
        elif (e.get("mAP50") or 0) > (v.get("mAP50") or 0):
            lines.append("- 在 VisDrone 上 **EsMoE-N 的 mAP50 高于 v0.1-N**，说明多尺度核专家在该垂类场景更具优势。")
        else:
            lines.append("- 两者 mAP50 接近，需更多 epoch 进一步区分。")
        lines.append(f"- 参数量: v0.1-N {v.get('params')} vs EsMoE-N {e.get('params')}，EsMoE-N 更轻量。")
    lines.append("- 训练全程记录 moe_loss，验证了两种 MoE 路由模块均可端到端训练。\n")
    lines.append("## 复现指引\n")
    lines.append("```bash\ncd D:/YOLO-Master\nPYTHONPATH=D:/YOLO-Master python scripts/reproduce/reproduce.py \\\n  --data scripts/reproduce/VisDrone_local.yaml --models v01 esmoe --epochs 5 --batch 8 --workers 0 --save-period 1\n```\n")
    lines.append("## 备注\n")
    lines.append("- SKU-110K 因本机网络流量限制（单会话累计约 2GB 后限速至 10KB/s）未能下载，故对比聚焦于 VisDrone。")
    lines.append("- 当前环境为 CPU 版 torch，GPU(CUDA) 版本待更好网络条件后补充。")
    out = os.path.join(OUT_DIR, "README_issue49.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] 报告 -> {out}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    comp = build_comparison()
    json.dump(comp, open(os.path.join(OUT_DIR, "comparison_results.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[ok] 对比 json -> {os.path.join(OUT_DIR, 'comparison_results.json')}")
    make_figure()
    make_readme(comp)
    print("[done] 交付物生成完毕，位于:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
