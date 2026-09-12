# C3 stage5: PR 与结项交付

从 stage1-4 证据汇总为可提交 PR 包（远端 fork: `github.com/lycyhrc/YOLO-Master` 分支 `c3-vpeft-smoke`）。
另含一条**对齐官方 main 的独立源码修复分支** `fix/vpeft-capacity-guard`（基线 `af961b9`）。

## 交付物清单（逐个 PR 提交）

| PR | 内容 | 状态 |
|---|---|---|
| PR-1 env+data | stage1(环境重建/数据下载/转换/划分/许可与 SHA) | 已 push 到 fork `c3-vpeft-smoke`(tip 见 fork 最新提交) |
| PR-2 server smoke | stage2(服务器化 runner/paths.env/planner solver notes/A800 冒烟 evidence) | 已 push(同上) |
| PR-3 matrix | stage3(matrix/queue_runner/collect_evidence/18 单元结果) | 已 push(同上) |
| PR-4 analysis | stage4(统计四维同表/planner 审计/p2 素材) | 已 push(同上) |
| PR-5 report | stage5(复现包/结项报告/汇报 pptx) | 已 push + report_final 定稿 + pptx 已重生成(末页含链接区) |
| **整体交付 PR** | `c3-vpeft-smoke` 全部交付物(stage1-5) → base `Tencent:main` | **复用仓库 PR `#279`**（重开）：正文 `pr_body_c3_overall.md` 已就绪；分支内容需先换成交付分支（见下） |
| 上游源码修复 PR | `fix/vpeft-capacity-guard`: C_cap 容量硬约束 + capacity_excluded 审计 + 未适配层冻结 + 单测（2 commit, 8 文件） | 已 push(tip `daed306`)；**改为新建 PR**（编号待创建），正文 `pr_body_capacity_guard.md` |

> 同门写法:仓库同门 PR 是「`[犀牛鸟-Xx]：` 前缀标题 + `Summary` / `Problem` / `Validation` / `Limitations` 分节正文」
> （已合并的源码修复参照 #240 `fix(lora): ...`、#267 `fix(mixture): ...`；带前缀的软件修复参照 #253 `[犀牛鸟-A2]：Fix ...`）。
> 本目录两个正文文件都已按该风格写好，标题与正文可直接粘贴。
>
> 归属约定（2026-09-11 更新）：**`#279` 用作整体交付 PR**，源码修复单独开新 PR。
> 注意 GitHub **不允许修改已有 PR 的 head 分支**，所以 #279 的 head 分支（现为 `fix/vpeft-capacity-guard`）
> 必须先改名为 `c3-vpeft-smoke` 再推交付内容，详见下方「整体交付 PR」。

## 复现包结构 `reproduction/`

```
reproduction/
├── data_and_env_from_stage1.md   # stage1 README：环境重建 + 数据集准备流程
├── datasets/
│   ├── manifest.md               # 数据来源/许可/统计/SHA 留档位置（权威说明）
│   ├── prepare_neu_det.py
│   └── prepare_deeppcb.py
├── configs/
│   ├── paths.env                 # 服务器化路径模板(替换为部署路径)
│   ├── env_setup.sh              # conda env 重建脚本
│   └── train_commands_example.sh # 三策略启动命令(同预算 epochs100 batch8 imgsz640 amp=false)
├── results/
│   ├── comparison_tables.md      # stage4 输出
│   ├── planner_audit_summary.md  # stage4 输出
│   ├── analysis_stats.py / collect_evidence.py
│   ├── matrix.json               # stage3 输出
│   └── evidence_summary.csv/json # stage3 输出
└── limitations.md                # 已知局限/许可风险/seed 稳定性观察
```

## 已知局限（提前记录，见 limitations.md）

- NEU-DET 镜像源无显式 LICENSE(数据许可风险已记录 SHA256 快照)。
- full/vpeft/frozen 主结论在 3 seed 层面给出; vpeft/frozen 在 NEU 上存在 seed
  稳定性差异(见 stage4, outlier: neu_vpeft_s2024 / neu_frozen_s2024 / pcb_frozen_s824),
  补充单元(neu_*_s2025, pcb_frozen_s824b)验证确定性。
- planner cap<8 已知缺陷：本轮通过 exclude 清单规避；已修复并提交上游候选 PR
  (`fix/vpeft-capacity-guard`: C_cap 容量硬约束 + capacity_excluded 审计 + 未适配层冻结)，消融未重跑。
- YOLO-Master n-scale 全参仅 2.8M,参数效率叙事弱于大模型; vpeft 的优势量化到
  4.15% adapter / 16.5% effective trainable(含类别 mismatch 解冻 head)。

## 收尾检查单

- [x] stage4 定稿(含补充单元)
- [x] reproduction/ 组装完整(收集各 stage 小文件)
- [x] 结项报告 report_final.md(结论/证据表/局限/复现)
- [x] 汇报 pptx(三策略对照 + planner 决策 + 证据链 + 末页链接区)
- [x] `fix/vpeft-capacity-guard` 已在 fork 上(源码修复, tip `daed306`)
- [x] **#279 改造为整体交付 PR**：fork 上删掉旧的 `c3-vpeft-smoke`，把 `fix/vpeft-capacity-guard` 改名成 `c3-vpeft-smoke`（#279 head 自动跟随）—— 2026-09-11 完成
- [x] `c3-vpeft-smoke` 已 force push 回 fork（本地 20 条 Conventional Commits 覆盖旧 11 条历史）：fork tip = `e41e147`
- [x] 源码修复分支已推回 fork 同名分支 `fix/vpeft-capacity-guard`（tip `daed306`），base `Tencent:main`
- [x] commit 信息风格对齐上游: `c3-vpeft-smoke` 已统一为 Conventional Commits(`feat(c3): …` / `docs(c3): …`), 与 `origin/main` 一致
- [ ] 重填 #279 的标题与正文（整体交付），然后点 `Reopen pull request`
- [ ] 源码修复 PR（新建）：base `Tencent:main` ← `fix/vpeft-capacity-guard`
- [x] PPT 用 Google Drive 链接交付: PR 描述末尾新增 `## PPT` 一节, 只放该链接(仓库 `c3_work/stage5_pr_delivery/` 保留同源副本, 未重新导出)
- [ ] (可选) 交付的 pptx 末页①写的是「上游修复 PR #279」，改号后与网页不一致；若要重出 deck 可运行 `gen_slides.py`（文案已同步）

## 整体交付 PR（复用仓库 `#279`）

- 标题：`[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付`
- 正文：`pr_body_c3_overall.md`（中文，覆盖 stage1-5 全部交付 + 上游修复概述 + P0/P1/P2 达成情况，末尾带 `## PPT`），从 `## 交付内容` 起粘贴。
- 分支：`lycyhrc:YOLO-Master:c3-vpeft-smoke` → base `Tencent:main`（即 #279 的 head）。

GitHub 改不了已有 PR 的 head 分支，所以要让 #279 显示交付内容，得先把它的 head 分支换成 `c3-vpeft-smoke`：

1. 网页 fork → Branches：删除 `c3-vpeft-smoke`（旧的 11 条旧格式历史，不是任何 PR 的 head），
   然后把 `fix/vpeft-capacity-guard` **改名**为 `c3-vpeft-smoke`。#279 的 head 会自动跟随改名，内容暂时还是那 2 个修复 commit，属正常。
2. 推送交付分支覆盖它（期望值 `daed306` = 改名后分支的当前 tip）：
   ```bash
   GIT_SSH_COMMAND='ssh -i /tmp/c3_push_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new' \
     git push --force-with-lease=refs/heads/c3-vpeft-smoke:daed306 \
     git@github.com:lycyhrc/YOLO-Master.git c3-vpeft-smoke:refs/heads/c3-vpeft-smoke
   ```
3. 打开 https://github.com/Tencent/YOLO-Master/pull/279 → 改标题/正文 → `Reopen pull request`。

> 不想折腾分支名也行（不推荐）：直接 `git push --force-with-lease=refs/heads/fix/vpeft-capacity-guard:daed306 <fork> c3-vpeft-smoke:fix/vpeft-capacity-guard`
> 把交付内容推进 #279 的现有 head 分支，代价是 PR 头部显示的分支名与内容不符。

## 上游源码修复 PR（新建）

- 标题：`[犀牛鸟-C3]：修复 V-PEFT 容量约束 —— 窄于最小候选 rank 的层不再被选为适配目标`
- 正文：`pr_body_capacity_guard.md` 里 `## 概述` 之后的全部内容（文件开头那段 HTML 注释是本地说明，不要一起粘）。
- 分支：`lycyhrc:YOLO-Master:fix/vpeft-capacity-guard` → base `Tencent:main`（2 commit，tip `daed306`；改名步骤后该分支名空闲，普通 push 即可重建）。

PPT（两个 PR 的正文末尾 `## PPT` 一节都只放此链接）：

```text
https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
```

> 发起方式：`Tencent/YOLO-Master` → New pull request → base `main` ← compare `lycyhrc:YOLO-Master:fix/vpeft-capacity-guard`。
> 若改成先建这个 PR 再处理 #279，pptx 末页①的链接要同步换成新编号。
