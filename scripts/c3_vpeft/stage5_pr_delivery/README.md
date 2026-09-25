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
| 源码修复 PR（可选，新开） | `fix/vpeft-capacity-guard`: C_cap 容量硬约束 + capacity_excluded 审计 + 未适配层冻结 + 单测 | 分支已 push(tip `daed306`)；PR 待**新开**（编号待创建），正文用 `pr_body_capacity_guard.md` |
| **最终交付 PR（新建）** | `c3-vpeft-smoke` 全部交付物(stage1-5) → base `Tencent:main` | 分支已 push(tip `456ae92`)；标题与正文见下方，待网页新建 |

> 源码修复 PR 的写法说明:仓库同门 PR 的写法是「`[犀牛鸟-Xx]：` 前缀标题 + `Summary` / `Problem` / `Validation` / `Limitations` 分节正文」
> （已合并的源码修复参照 #240 `fix(lora): ...`、#267 `fix(mixture): ...`；带前缀的软件修复参照 #253 `[犀牛鸟-A2]：Fix ...`）。
> 本目录的 `pr_body_capacity_guard.md` 已按该风格重写，标题与正文可直接粘贴。
>
> **分支归属（2026-09-11 最终）**：`#279` 的 head 是 `fix/vpeft-capacity-guard`，而 GitHub 不允许改已有 PR 的 head 分支，
> 所以**最终交付 PR 只能从 `c3-vpeft-smoke` 新建**；`#279` 保持 `closed` 不再使用（源码修复要提就另开新 PR）。

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
- [x] `c3-vpeft-smoke` 已 push 回 fork（本地历史重写为 Conventional Commits，覆盖旧格式历史）：fork tip = `456ae92`
- [x] commit 信息风格对齐上游: `c3-vpeft-smoke` 已统一为 Conventional Commits(`feat(c3): …` / `docs(c3): …`), 与 `origin/main` 一致
- [ ] **最终交付 PR（唯一必做）**：New pull request → base `Tencent:main` ← `lycyhrc:YOLO-Master:c3-vpeft-smoke`，标题与正文见下方
- [ ] (可选) 源码修复另开新 PR：base `Tencent:main` ← `fix/vpeft-capacity-guard`，正文粘 `pr_body_capacity_guard.md`
- [x] PPT 用 Google Drive 链接交付: 只在整体交付 PR 描述末尾的 `## PPT` 一节放该链接(仓库 `scripts/c3_vpeft/stage5_pr_delivery/` 保留同源副本, 未重新导出)
- [ ] (可选) pptx 末页②仍是占位符 `<<EVIDENCE_LINK_PLACEHOLDER>>`, 若要重出可填同一 Drive 链接

## 最终交付 PR（`c3-vpeft-smoke` → `Tencent:main`，**唯一必做**）

标题：

```text
[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付
```

正文：粘 `pr_body_c3_overall.md` 里 `## SUMMARY｜C3 阶段交付` 到文件末尾的全部内容。
骨架按同门 PR #281 重写：`SUMMARY｜…` → P0/P1/P2 总览表 → 三个阶段分节 → `REVIEWER GUIDE｜交付物与证据入口` → `VALIDATION` → 已知局限 → `说明` → `PPT`。

发起方式（网页）：`Tencent/YOLO-Master` → New pull request → base `Tencent:main` ← compare `lycyhrc:YOLO-Master:c3-vpeft-smoke`。

> 两个正文文件的用途：`pr_body_c3_overall.md` = 最终交付 PR（本课题全部工作，末尾带 `## PPT` 链接）；
> `pr_body_capacity_guard.md` = 只讲源码修复的另一个 PR（2 commit，8 文件，正文不带 PPT，可选提交）。

## 最终交付 PR 的提交方式（只需网页操作）

`https://github.com/Tencent/YOLO-Master/pulls` → `New pull request` → base `Tencent/YOLO-Master:main`
← compare `lycyhrc/YOLO-Master:c3-vpeft-smoke`，然后：

1. 标题填 `[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付`；
2. 描述框粘 `pr_body_c3_overall.md` 里 `## SUMMARY｜C3 阶段交付` 到末尾的全部内容（开头 HTML 注释是本地说明，不要粘）；
3. 末尾 `## PPT` 一节保留这一行：

```text
https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
```

> `#279` 不用再管：它的 head 是 `fix/vpeft-capacity-guard`，而 GitHub 不允许改已有 PR 的 head 分支，
> 重开也只能提出源码修复那 8 个文件，挂不上整体交付。pptx 末页①的链接在 PR 号确定后如需再填，用 `gen_slides.py` 重出。

## 源码修复 PR（另开，可选）

标题：

```text
[犀牛鸟-C3]：修复 V-PEFT 容量约束 —— 窄于最小候选 rank 的层不再被选为适配目标
```

base `Tencent:main` ← head `lycyhrc/YOLO-Master:fix/vpeft-capacity-guard`；正文粘 `pr_body_capacity_guard.md` 里
`## SUMMARY｜问题与修复` 之后的全部内容（同样不粘开头的 HTML 注释），正文不带 PPT 章节。
不打算提这个 PR 也不影响最终交付。
