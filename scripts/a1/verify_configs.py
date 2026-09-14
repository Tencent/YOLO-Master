"""A1 实验格配置自检：验证 A/B/C/D 四格 head 构建、criterion 与 MoE 结构差异。

检查点（依据 experiments/A1/A1-moe-end2end-方案设计.md 已核实事实）：
- end2end=True 格（C/D）：head 有 one2one 分支、criterion 为 E2E 双分支
- end2end=False 格（A/B）：head 无 one2one 分支、criterion 为 v8DetectionLoss
- MoE on 格（B/D）：backbone 含 A2C2fMoE 模块
- 四格均能构建并完成一次前向

用法：
    python scripts/a1/verify_configs.py
"""

from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]  # YOLO-Master 仓库根

# (格, yaml 路径, 期望 end2end, 期望 MoE)
CASES = [
    ("A", ROOT / "ultralytics/cfg/models/26/yolo26-end2end-false.yaml", False, False),
    ("B", ROOT / "ultralytics/cfg/models/26/yolo26-master-end2end-false.yaml", False, True),
    ("C", ROOT / "ultralytics/cfg/models/26/yolo26.yaml", True, False),
    ("D", ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml", True, True),
]


def check(case: str, yaml_path: Path, expect_end2end: bool, expect_moe: bool) -> None:
    model = YOLO(yaml_path).model
    head_module = model.model[-1]

    has_one2one = hasattr(head_module, "one2one_cv2") and hasattr(head_module, "one2one_cv3")
    # criterion 为惰性初始化（首次 model.loss 调用时才构建），自检时显式调用 init_criterion
    # MoE 格（有路由模块）会被 CompositeCriterion 包装，检查其内部 native criterion
    criterion = model.init_criterion()
    if type(criterion).__name__ == "CompositeCriterion":
        criterion = criterion.native_criterion
    criterion_name = type(criterion).__name__
    model_end2end = bool(getattr(model, "end2end", False))
    n_moe = sum(1 for m in model.modules() if type(m).__name__ == "A2C2fMoE")

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))

    assert model_end2end == expect_end2end, f"[{case}] end2end 标志不符：期望 {expect_end2end}"
    assert (n_moe > 0) == expect_moe, f"[{case}] MoE 模块不符：期望 {expect_moe}，实际 {n_moe} 个"
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
        assert not (isinstance(out, dict) and "one2one" in out), f"[{case}] 训练 forward 不应返回 one2one"

    print(
        f"[{case}] PASS  end2end={model_end2end}  moe_blocks={n_moe}  one2one={has_one2one}  "
        f"criterion={criterion_name}  forward_ok=True"
    )


if __name__ == "__main__":
    for case, yaml_path, expect_e2e, expect_moe in CASES:
        check(case, yaml_path, expect_e2e, expect_moe)
    print("四格配置自检全部通过")
