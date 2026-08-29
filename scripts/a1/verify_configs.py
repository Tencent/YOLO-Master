"""A1 S1 阶段0配置自检：验证 A/C 两格 head 构建与 criterion 差异。

检查点（依据 experiments/A1/A1-moe-end2end-方案设计.md 已核实事实）：
- C 格 yolo26.yaml（end2end=True）：head 有 one2one 分支、criterion 为 E2EDetectLoss
- A 格 yolo26-end2end-false.yaml（end2end=False）：head 无 one2one 分支、criterion 为 v8DetectionLoss
- 两格均能构建并完成一次前向

用法：
    python scripts/a1/verify_configs.py
"""

from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]  # YOLO-Master 仓库根

CASES = [
    ("C", ROOT / "ultralytics/cfg/models/26/yolo26.yaml", True),
    ("A", ROOT / "ultralytics/cfg/models/26/yolo26-end2end-false.yaml", False),
]


def check(case: str, yaml_path: Path, expect_end2end: bool) -> None:
    model = YOLO(yaml_path).model
    head = model.model[-1] if hasattr(model, "model") else model[-1]
    head_module = model.model[-1]

    has_one2one = hasattr(head_module, "one2one_cv2") and hasattr(head_module, "one2one_cv3")
    # criterion 为惰性初始化（首次 model.loss 调用时才构建），自检时显式调用 init_criterion
    criterion = model.init_criterion()
    criterion_name = type(criterion).__name__
    model_end2end = bool(getattr(model, "end2end", False))

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))

    assert model_end2end == expect_end2end, f"[{case}] end2end 标志不符：期望 {expect_end2end}"
    if expect_end2end:
        assert has_one2one, f"[{case}] 应存在 one2one 分支"
        # 注：tasks.py:659 实际注入 E2ELoss（E2EDetectLoss 的增益调度增强版），
        # 两者本质相同：criterion 含 one2many + one2one 两个子 loss
        assert criterion_name in {"E2ELoss", "E2EDetectLoss"}, (
            f"[{case}] criterion 应为 E2E 双分支，实际 {criterion_name}"
        )
        assert hasattr(criterion, "one2one"), f"[{case}] criterion 应含 one2one 子 loss"
        assert isinstance(out, dict) and "one2one" in out, f"[{case}] 训练 forward 应返回 one2one"
    else:
        assert not has_one2one, f"[{case}] 不应存在 one2one 分支"
        assert criterion_name == "v8DetectionLoss", f"[{case}] criterion 应为 v8DetectionLoss，实际 {criterion_name}"
        assert not hasattr(criterion, "one2one"), f"[{case}] criterion 不应含 one2one 子 loss"
        assert not (isinstance(out, dict) and "one2one" in out), f"[{case}] 训练 forward 不应返回 one2one"

    print(
        f"[{case}] PASS  end2end={model_end2end}  one2one_branch={has_one2one}  "
        f"criterion={criterion_name}  forward_ok=True"
    )


if __name__ == "__main__":
    for case, yaml_path, expect in CASES:
        check(case, yaml_path, expect)
    print("配置自检全部通过")
