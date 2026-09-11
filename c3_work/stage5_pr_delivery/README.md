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
| PR-6 上游源码 | `fix/vpeft-capacity-guard`: C_cap 容量硬约束 + capacity_excluded 审计 + 未适配层冻结 + 单测 | 已 push(tip `daed306`)；PR **#279** 当前 `closed`、标题为分支名派生、正文为空 → 重开后按 `pr_body_capacity_guard.md` 填 |
| **整体 PR** | `c3-vpeft-smoke` 全部交付物(stage1-5, 14 commit) → base `Tencent:main` | 分支已 push；正文 `pr_body_c3_overall.md` 已就绪, 待网页发起 |

> PR-6 说明:仓库同门 PR 的写法是「`[犀牛鸟-Xx]：` 前缀标题 + `Summary` / `Problem` / `Validation` / `Limitations` 分节正文」
> （已合并的源码修复参照 #240 `fix(lora): ...`、#267 `fix(mixture): ...`；带前缀的软件修复参照 #253 `[犀牛鸟-A2]：Fix ...`）。
> 本目录的 `pr_body_capacity_guard.md` 已按该风格重写，标题与正文可直接粘贴。

## 复现包结构 `reproduction/`

```
reproduction/
├── env.txt              # conda env export(yolo_master)
├── data_manifest.json   # 两数据集来源/sha256/划分(来自 stage1)
├── configs/
│   ├── paths.env        # 服务器化路径模板(替换为部署路径)
│   └── budget_*.sh      # 三策略启动命令(同预算 epochs100 batch8 imgsz640 amp=false)
├── results/
│   ├── comparison_tables.md   # stage4 输出
│   ├── evidence_summary.csv   # stage3 输出
│   └── planner_audit_summary.md
└── limitations.md       # 已知局限/许可风险/seed 稳定性观察
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
- [x] 两个分支已 push fork: `c3-vpeft-smoke`(证据包, tip 见 fork 最新提交)、`fix/vpeft-capacity-guard`(源码修复, tip `daed306`)
- [x] commit 信息风格对齐上游: `c3-vpeft-smoke` 15 条已统一为 Conventional Commits(`feat(c3): …` / `docs(c3): …`), 与 `origin/main` 一致
- [ ] 重填 PR #279 的标题与正文(见下方「PR-6 提交方式」), 然后点 `Reopen pull request`
- [x] PPT 用 Google Drive 链接交付: PR 描述末尾新增 `## PPT` 一节, 只放该链接(仓库 `c3_work/stage5_pr_delivery/` 保留同源副本, 未重新导出)
- [ ] (可选) pptx 末页②仍是占位符 `<<EVIDENCE_LINK_PLACEHOLDER>>`, 若要重出可填同一 Drive 链接

## 整体交付 PR（`c3-vpeft-smoke` → `Tencent:main`）

正文：`pr_body_c3_overall.md`（中文，覆盖 stage1-5 全部交付 + 上游修复概述，末尾带 `## PPT`）。
标题建议：

```text
[犀牛鸟-C3]：V-PEFT 小样本工业缺陷检测结项交付
```

发起方式（网页）：`Tencent/YOLO-Master` → New pull request → base `main` ← compare `lycyhrc:YOLO-Master:c3-vpeft-smoke`。

> 两个正文文件的用途：`pr_body_c3_overall.md` = 整体交付 PR（本课题全部工作）；
> `pr_body_capacity_guard.md` = 只讲源码修复的上游 PR #279（2 commit，8 文件）。

## PR-6 提交方式（只剩网页操作）

标题（替换 GitHub 上由分支名派生的 `Fix/vpeft capacity guard`）：

```text
[犀牛鸟-C3]：修复 V-PEFT 容量约束 —— 窄于最小候选 rank 的层不再被选为适配目标
```

正文：把 `pr_body_capacity_guard.md` 里 `## Summary` 之后的全部内容粘进描述框（文件开头那段 HTML 注释是本地说明，不要一起粘）。

PPT（正文末尾 `## PPT` 一节只放此链接）：

```text
https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
```

然后在本 PR 页面点 `Reopen pull request`（分支还在，直接重开即可）：

https://github.com/Tencent/YOLO-Master/pull/279

> 若选择新建 PR 而不是重开，pptx 末页①的链接要同步换成新编号。
