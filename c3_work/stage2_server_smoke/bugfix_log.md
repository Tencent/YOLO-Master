# bugfix_log.md · 复现中发现并修复的问题（素材库）

## BF-01 共享服务器上裸 `yolo` 命中错误的 ultralytics（已修复）

- **现象**：`server_run.py`（及准入 `smoke/c3/run_smoke.py`）用 `subprocess.run(["yolo", ...])` 调 CLI。在本 8×A800 共享机上，PATH 中 `yolo` 解析到 `/mnt/pfs/zitao_team/miniconda3/bin/yolo`（base conda，且其 import 链路加载了另一用户 `chenzhong1/xuexiji_ocr/ultralytics` 的副本）。
- **症状**：`SyntaxError: 'lora_exclude_modules' is not a valid YOLO argument` 等 8 个 lora_* 全部不可识别，训练 7.8s 内 exit 1。**与仓库实际代码无关**，纯环境污染，极易误判为代码/版本问题。
- **根因**：repo 的 LoRA/planner 参数只在**本仓库 editable 安装**的 ultralytics 中注册；裸 `yolo` 走 PATH，可复现性被破坏。
- **修复**：运行器用 `Path(sys.executable).parent / "yolo"`（当前解释器同 env 的 yolo），不依赖 PATH。涉及 `c3_work/stage2_server_smoke/server_run.py` 与 `smoke/c3/run_smoke.py`。
- **泛化结论（写进复现包 README）**：所有 CLI 调用必须 `$ENV/bin/yolo` 或显式 `source activate <env>` 后执行；禁止依赖全局 PATH。
- 验证：修复后 vpeft 冒烟正常进入训练（resolved config 完整含 lora_planner_*）。

## 待补充（观测后更新）

- V-PEFT strict 模式实际决策（accept targets/ranks）与 MPS 已见 cap<8 缺陷在本机 8 卡版的复现情况。
