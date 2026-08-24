# 08 - 风险与降级

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：风险与降级

## 已识别风险

### R1：无 NVIDIA GPU（本机硬限制）
- **影响**：无法跑 TensorRT FP16 / INT8 真实部署；无法挂 COCO val2017 完整集
- **降级**：按任务规则 "无卡降级 ONNX Runtime CPU 后端"，本次仅做 FP32 PyTorch → FP32 ONNX → FP32 ORT CPU
- **声明位置**：`01_environment_install.md`（环境声明）+ 本文件
- **后续**：P1 阶段挂云端 GPU 补做

### R2：COCO 完整 val2017 跑全集耗时不经济
- **影响**：本机 CPU 单卡跑 5000 张 val 需数小时，8.24 时间窗不允许
- **降级**：使用 COCO8（4 train + 4 val）作为精度差基线
- **声明位置**：`02_baseline_minimum_task.md`（任务边界）+ `06_result_evidence/README.md`（局限）
- **后续**：P1 阶段挂云端 GPU 重做

### R3：网络限制（GitHub 直连不通）
- **影响**：无法直接 `git clone` / `pip install` 走 GitHub
- **降级**：用 ghfast.top 镜像（GitHub 协议层代理）+ hf-mirror.com（HuggingFace 镜像）+ pypi.tuna.tsinghua.edu.cn（PyPI 镜像）
- **声明位置**：`01_environment_install.md`（镜像源声明）
- **后续**：无需解决

### R4：examples 子包存在版本号笔误
- **影响**：`examples/RTDETR-ONNXRuntime-Python/requirements.txt` 中 `onnxruntime==1.19.2` 是上游笔误（该版本号未发布），导致 `pip install -e .` 失败
- **降级**：就地修复为 `onnxruntime>=1.20.0`
- **声明位置**：`01_environment_install.md`（已知问题）
- **后续**：可在 fork 提交 PR 给 upstream 修复

### R5：MoT / MoA / MoA-MOT 缺公开预训练权重
- **影响**：仅有 yaml 配置，无对应 .pt 端到端验证
- **降级**：本次 smoke 仅用 EsMoE-N（已有公开权重）走通端到端
- **声明位置**：`02_baseline_minimum_task.md`（任务边界）
- **后续**：P1 阶段可从 yaml 随机初始化 + 自训小样本补做

### R6：fork 与 upstream 分叉
- **影响**：本地工作目录有 3710 个本地文件 untracked（主要是 .venv / 产物 / smoke/），push 时需注意
- **降级**：仅 push `smoke/a3/` 目录相关文件 + 必要的 .gitignore / README；.venv/、*.pt、*.onnx 通过 .gitignore 排除
- **声明位置**：`.gitignore` 策略 + 本文件
- **后续**：N/A

### R7：immutable tag 不能 force-push
- **影响**：tag 一旦创建，规则声明不允许 force-push 改写
- **降级**：tag 内容在创建前应经过充分校验（本地能 `git diff` / `git log` 通过才打 tag）
- **声明位置**：`02_baseline_minimum_task.md`（tag 策略）+ 本文件
- **后续**：N/A

## 风险等级总览

| 风险 | 等级 | 是否在 P0 范围内解决 |
|---|---|---|
| R1 无 GPU | 高 | ❌ 降级到 CPU 后端 |
| R2 COCO 完整集耗时 | 中 | ❌ 降级到 COCO8 |
| R3 网络限制 | 中 | ❌ 用镜像源 |
| R4 笔误 | 低 | ✅ 已修复 |
| R5 MoT 缺权重 | 中 | ❌ P1 阶段补 |
| R6 fork 分叉 | 中 | ✅ 用 .gitignore 隔离 |
| R7 tag 保护 | 低 | ✅ 规则声明 |

## 不可作证项声明

按规则 3.2，本报告**不依赖**以下证据：
- 截图（仅做辅助，PNG 曲线图 + 检测图用于视觉佐证，不作为唯一证据）
- 口头说明
- 不可访问的私有仓库
- 后补 README（本文档为同步撰写，不是后补）
- 只有结论没有原始记录的报告（`05_full_logs.md` 与 `06_result_evidence/val_metrics.json` 提供原始记录）
