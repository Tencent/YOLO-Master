# C3 stage5: PR 与结项交付

从 stage1-4 证据汇总为可提交 PR 包（远端 fork: `github.com/lycyhrc/YOLO-Master` 分支 `c3-vpeft-smoke`）。
另含一条**对齐官方 main 的独立源码修复分支** `fix/vpeft-capacity-guard`（基线 `af961b9`）。

## 交付物清单（逐个 PR 提交）

| PR | 内容 | 状态 |
|---|---|---|
| PR-1 env+data | stage1(环境重建/数据下载/转换/划分/许可与 SHA) | 已提交 commit, 待 push |
| PR-2 server smoke | stage2(服务器化 runner/paths.env/planner solver notes/A800 冒烟 evidence) | 已提交 commit, 待 push |
| PR-3 matrix | stage3(matrix/queue_runner/collect_evidence/18 单元结果) | 已提交 commit, 待 push |
| PR-4 analysis | stage4(统计四维同表/planner 审计/p2 素材) | 定稿 commit, 待 push |
| PR-5 report | stage5(复现包/结项报告/汇报 pptx) | report_final 定稿 + pptx 已重生成(末页含链接占位) + reproduction 已组装 |
| PR-6 上游源码 | `fix/vpeft-capacity-guard`: C_cap 容量硬约束 + capacity_excluded 审计 + 未适配层冻结 + 单测 | 2 commit 已提交, 待 push 并发起上游 PR |

> push 由本人手动执行(仓库 credentials 当前不可自动用)。本地 commit 均已留好。
> PR-6 文案：`pr_body_capacity_guard.md`；链接占位见 pptx 末页①。

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

- [x] stage4 定稿(含补充单元) ｜ push 待本人执行
- [x] reproduction/ 组装完整(收集各 stage 小文件)
- [x] 结项报告 report_final.md(结论/证据表/局限/复现)
- [x] 汇报 pptx(三策略对照 + planner 决策 + 证据链 + 末页 <<...>> 链接占位)
- [ ] 逐个 PR 四节说明 + PR-6 源码修复分支 push fork(本人执行, 见下方命令)
- [ ] 用实际链接替换 pptx 末页 <<PR_LINK_PLACEHOLDER>> / <<EVIDENCE_LINK_PLACEHOLDER>>

## 待本人执行的两条 push

```bash
cd research/TX_yolo/YOLO-Master
git push -u fork c3-vpeft-smoke                      # 证据包(PR-1~PR-5)
git push -u fork fix/vpeft-capacity-guard            # 上游源码修复(PR-6)
# 发起 PR: https://github.com/Tencent/YOLO-Master/compare/main...lycyhrc:YOLO-Master:fix/vpeft-capacity-guard?expand=1
```
