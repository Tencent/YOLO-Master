# YOLO-Master Agent

`agent/` 是本仓库的智能代理运行层, 用于把 YOLO-Master / Ultralytics 的训练、验证、推理、导出、多模态评估与开放世界分析能力包装成稳定的 agent skill。

这个目录不是一个独立项目, 而是仓库内的代理适配层。核心原则是: 先安装并使用本地 Ultralytics 框架, 优先通过 `yolo` CLI 执行主流程; 当 CLI 无法覆盖某些增强能力时, 再由 `runtime/` 中的 Python 模块补足。

## Directory Layout

```text
agent/
├── SKILL.md                 # 给 AI agent 读取的技能入口与操作规约
├── README.md                # 给开发者和维护者阅读的目录说明
├── assets/                  # skill 使用的数据、prompt、分类词表与轻量测试集
├── logs/                    # AutoTrain / 多模态评估报告与运行缓存
├── metadata/                # skill 展示、入口与界面元数据
├── references/              # 架构、接口、thinking-with-image 等长文档
├── scripts/                 # 仅保留可执行薄封装
└── runtime/                 # 可复用实现库
    ├── cli/                 # dispatcher、contract、pipeline、LoRA/PEFT 工具、validator
    ├── evaluation/          # 指标预览、guardrail、融合评估
    ├── multimodal/          # VLM/LLM 调用、视觉标注与融合逻辑
    └── open_world/          # LVIS/V3Det 分类学与开放世界归一化
```

## Execution Surface

日常调用应从 `scripts/` 进入。这里的脚本只负责设置 import path 并委托给 `runtime/cli/`, 避免把业务逻辑继续堆在可执行入口中。

环境检查:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.system","action":"doctor","params":{"ensure_cli":true}}' --pretty
```

快速回归验证:

```bash
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

真实 YOLO CLI 调用仍应优先使用仓库内安装出的 `yolo`:

```bash
python -m pip install -e .
yolo version
```

## Runtime Responsibilities

- `runtime/cli/dispatcher.py`: 统一解析 skill request 并路由到具体 runtime 模块; 新逻辑优先放入同级模块, 避免继续膨胀入口文件。
- `runtime/cli/executor.py`, `device.py`, `normalize.py`, `dataset.py`: CLI 执行/恢复、设备环境、请求规范化、数据集与图像解析的共享实现, 保持 dispatcher 只负责路由和 handler 编排。
- `runtime/cli/system_handlers.py`: `yolo.system` 的 doctor、install、settings、cfg 等系统管理动作。
- `runtime/cli/model_handlers.py`: `yolo.model.inspect` 的模型元信息、names、device、task map 与轻量模型操作。
- `runtime/cli/core_handlers.py`: `train/val/predict/track/export/benchmark/tune` 与 LoRA adapter 的主干 handler, 通过依赖注入复用 dispatcher 的模型与 CLI 兼容入口。
- `runtime/cli/launcher_handlers.py`: `yolo.solutions.run` 与 `yolo.ui.launch` 的 launcher-style handler, 包含 Solutions 参数组装和 Gradio/Streamlit 启动。
- `runtime/cli/contract.py`: 统一响应 envelope、manifest、`usage.tokens` 与 `cost_estimate`。
- `runtime/cli/async_jobs.py`: 为 `policy.async=true` 的长任务提交子进程 job, 返回 `job_id`, status 文件、stdout/stderr 和可 tail 的 progress 路径。
- `runtime/cli/job_handlers.py`: `yolo.job.status` / `yolo.job.cancel` 的异步任务查询与取消入口。
- `runtime/cli/multimodal_handlers.py`: `yolo.multimodal.infer` / `yolo.multimodal.evaluate` 的 handler 编排, 通过依赖注入保留 dispatcher 兼容 monkeypatch 入口。
- `runtime/cli/pipeline.py`: `yolo.pipeline.experiment` 的端到端阶段编排, 支持 train/val/export/benchmark 以及 LoRA/MoE/PEFT 诊断阶段, 并为长流程写出可 tail 的 `progress.jsonl`。
- `runtime/cli/lora_tools.py`: `yolo.lora.diagnose`, 包含 effective rank、LoRA A/B 范数与 delta-W 谱预览。
- `runtime/cli/moe_tools.py`: `yolo.moe.diagnose` / `yolo.moe.prune`, 包含专家使用率、保留/裁剪专家列表、裁剪比例与验证状态预览。
- `runtime/cli/peft_compare.py`: `yolo.eval.peft_compare`, 用统一请求结构编排多个 LoRA/DoRA/LoHA/Full-SFT 变体。
- `runtime/cli/sahi_compare.py`: `yolo.eval.sparse_sahi_compare`, 将标准推理、完整 SAHI 和集成 Sparse SAHI 对比纳入统一 skill 契约。
- `runtime/cli/validate.py`: AutoTrain 风格用例执行器, 支持 `quick`, `contract`, `cli-smoke`, `deep-smoke`, `extended` 等验证套件。
- `runtime/multimodal/`: 负责 OpenAI-compatible VLM/LLM 请求、thinking-with-image prompt、reasoning 摘要、marked image、crop/zoom 视觉搜索与结构化结果解析; provider 配置位于 `runtime/multimodal/providers/*.yaml`。
- `runtime/evaluation/`: 负责 YOLO-only 与融合结果的指标预览、分类/检测/分割 delta、COCO/label record 构建和 metric guardrail。
- `runtime/open_world/`: 负责 LVIS/V3Det taxonomy 匹配、open-world profile、IoU relabel、未匹配标签兜底与 verified list 合并。

## Skill Contract

`SKILL.md` 是 agent 的主要读取入口, 应保持短而可执行; 长说明放入 `references/`。当前 `.codex/skills/yolo-master-agent/` 只作为兼容桥接目录存在, 真实实现和维护入口都在这里。

请求结构遵循:

```json
{
  "skill": "yolo.train",
  "runtime": {"prefer_cli": true, "prefer_mps": true},
  "inputs": {"model": "yolo11n.pt", "data": "coco128.yaml"},
  "params": {"epochs": 1, "imgsz": 640},
  "policy": {"dry_run": false}
}
```

新增能力时优先补 `runtime/` 模块和验证用例, 再让 `scripts/` 暴露薄入口。不要把长逻辑重新写回 `scripts/`。

AutoTrain case 默认从 `assets/autotrain_cases/` 目录读取, 按 skill 分组维护; 旧的 `assets/autotrain_cases.json` 仍保留为兼容入口。
