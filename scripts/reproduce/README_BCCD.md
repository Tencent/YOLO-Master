# 🎯 Issue #49 Reproduction: YOLO-Master Baseline & EsMoE on BCCD
**【2026犀牛鸟开源人才专属】模型训练专项-垂类数据集基线训练复现**

> **Language**: [English](#english-version) | [中文版](#中文版)

---

<h2 id="english-version">🇬🇧 English Version</h2>

### 1. Dataset & Rationale
According to the Issue #49 guidelines ("可选，换用其他的垂类公开数据集" / optional: use other vertical public datasets), this reproduction utilizes the **BCCD (Blood Cell Count and Detection)** dataset. It provides an excellent and efficient environment for evaluating dense, tiny-object detection without exceeding computational limits.

### 2. Training Commands
We provide two separate scripts to reproduce the results without overriding existing repository files. Run the following commands in your terminal:

```bash
# Train YOLO-Master-v0.1-N (Baseline)
python scripts/reproduce/reproduce_bccd_v01.py

# Train YOLO-Master-EsMoE-N (Mixture of Experts)
python scripts/reproduce/reproduce_bccd_esmoe.py
```

### 3. Results & Visualization
| Model | Config | Epochs | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: |
| `YOLO-Master-v0.1-N` | `.../v0_1/.../yolo-master-n.yaml` | 100 | 0.9096 | 0.6110 |
| `YOLO-Master-EsMoE-N` | `.../v0/.../yolo-master-n.yaml` | 100 | 0.8911 | 0.5931 |

*The following charts demonstrate the convergence. Note that `train/moe_loss` correctly activates during the EsMoE-N training, verifying the router's load-balancing mechanism.*

| YOLO-Master-v0.1-N (Baseline) | YOLO-Master-EsMoE-N (MoE Activated) |
| :---: | :---: |
| <img src="../../assets/bccd_v01_results.jpg" width="450"> | <img src="../../assets/bccd_esmoe_results.jpg" width="450"> |

### 4. 🐛 Newly Discovered Issue & Bug Fix (Monkey Patch)
**Symptom:** When the `EsMoE-N` model transitions to `final_eval`, it frequently crashes with `MoERouterError: Router input contains NaN/Inf values`.
**Mechanism:** In `ultralytics/nn/autobackend.py`, `model.warmup()` allocates a dummy tensor using `torch.empty()`. If the GPU memory contains uninitialized dirty data, the extremely strict `_validate_router_input` inside the MoE router will catch the `NaN/Inf` and terminate the evaluation.
**Solution:** A zero-invasion Monkey Patch is implemented in `reproduce_bccd_esmoe.py` to dynamically override the `warmup` method, replacing `torch.empty` with `torch.zeros` to ensure a clean evaluation tensor.

---

<h2 id="中文版">🇨🇳 中文版</h2>

### 1. 数据集说明
依据 Issue #49 中 *“可选，换用其他的垂类公开数据集”* 的规则，考虑到算力限制与快速验证的需求，本次提交采用了 **BCCD (血细胞计数)** 这一密集小目标垂类数据集进行完整复现。

### 2. 训练命令
为了避免与主分支已有文件冲突，本次提交新增了针对 BCCD 数据集的专属运行脚本：

```bash
# 运行基础版基线模型
python scripts/reproduce/reproduce_bccd_v01.py

# 运行混合专家模型（带自动防崩溃补丁）
python scripts/reproduce/reproduce_bccd_esmoe.py
```

### 3. 预期结果与图表
| 模型名称 | 配置文件所在目录 | 训练轮数 | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: |
| `YOLO-Master-v0.1-N` | `.../v0_1/.../yolo-master-n.yaml` | 100 | 0.9096 | 0.6110 |
| `YOLO-Master-EsMoE-N` | `.../v0/.../yolo-master-n.yaml` | 100 | 0.8911 | 0.5931 |

*注：下方对比图清晰展示了 `train/moe_loss` 在 EsMoE 模型中被成功激活并收敛，证明混合专家路由分发机制运行正常。*

| YOLO-Master-v0.1-N (无专家路由) | YOLO-Master-EsMoE-N (专家路由激活) |
| :---: | :---: |
| <img src="../../assets/bccd_v01_results.jpg" width="450"> | <img src="../../assets/bccd_esmoe_results.jpg" width="450"> |

### 4. 🐛 已知问题与底层框架修复 (Monkey Patch)
* **问题发现**：在训练 `YOLO-Master-EsMoE-N` 结束并触发 `final_eval` 时，发现底层 `AutoBackend.warmup` 使用 `torch.empty` 申请了未初始化的显存。产生的脏数据 (NaN/Inf) 会直接触发 MoE 路由器的严格输入校验，导致抛出 `MoERouterError` 崩溃。
* **解决方案**：在 `reproduce_bccd_esmoe.py` 脚本中，已提供无侵入式的 **Monkey Patch（猴子补丁）** 解决方案。通过动态将 `torch.empty` 替换为安全的 `torch.zeros`，确保模型平滑通过热身评估，且无需修改官方框架源码。