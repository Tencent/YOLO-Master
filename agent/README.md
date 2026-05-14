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
    ├── cli/                 # dispatcher、validator、报告命令
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

- `runtime/cli/dispatcher.py`: 统一解析 skill request, 选择 CLI/Python 执行路径, 注入设备与 macOS/MPS 默认值, 写出 manifest。
- `runtime/cli/validate.py`: AutoTrain 风格用例执行器, 支持 `quick`, `contract`, `cli-smoke`, `deep-smoke`, `extended` 等验证套件。
- `runtime/multimodal/`: 负责 OpenAI-compatible VLM/LLM 请求、thinking-with-image prompt、marked image、crop/zoom 视觉搜索与结构化结果解析。
- `runtime/evaluation/`: 负责 YOLO-only 与融合结果的指标预览、分类/检测/分割 delta、metric guardrail。
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
