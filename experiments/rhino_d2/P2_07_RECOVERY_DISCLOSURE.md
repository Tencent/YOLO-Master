> **Archived research document.** Statements about copying records, running tests and delivery dates below describe the original packaging stage, not actions performed by this integration commit. Referenced evidence stays in the archive; it is not bundled here.

Public access: [public derivative of this document](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/P2_07_RECOVERY_DISCLOSURE.md). Original document identity remains local at packaging SHA `781a6f148a46d59a89ea40806de8e10eb7b98941`. The public package is a sanitized derivative, not the original Git history; its PROVENANCE.json distinguishes byte-preserved files from path-sanitized records. Local-only references below are not public links. Import edits: this notice and pinned link destinations only; archived figures are linked rather than embedded. Scientific text and numbers are unchanged. The protocol's original implementation is distinct from post-freeze public-API hardening in integration Commit 1 `3a62b6f480e2855c3034f9788f69b804b81a5b56`.

---

# P2-07 / B24 人工中断恢复披露

本页是事后整理，不冒充事前注册或当时的授权记录；不改写最终科学锚点 e3e11d0。

原始隔离目录：`E:/2026YOLO/P2-07-interrupted-B24-20260909-224418`。
本次保留原目录未动，复制非权重日志/配置/CSV/逐批证据到
`results/p2_terminal_freeze/recovery/`（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_terminal_freeze/recovery/`），每一项复制前后 SHA-256 一致。
权重不重复入Git，仅在 audit.json（本地原始归档，未收入公开包：`781a6f148a46d59a89ea40806de8e10eb7b98941:experiments/rhino_d2/results/p2_terminal_freeze/audit.json`） 中记录完整相对路径、大小与SHA-256。

## 时序（北京时间）

- 2026-09-09 20:41:08：原队列启动；当前 arm 为B24。
- 原运行未完成；隔离 CSV 含33个完整 epoch。原队列状态仍写running，不能用它推断中断之后仍在运行。
- 22:44:39：recovery note 记录 `manual_process_termination`、`interrupted_external_manual`，计划从epoch0重启。
- 22:45:32：一次恢复启动因已有 per-arm evidence 被拒绝；错误为“禁止覆盖或静默resume”。未开始训练。
- 22:50:02/06：note 对同一 `restart_attempt_1` 重复记载；不据此虚构两个额外训练尝试。
- 22:50:43：新队列启动；完整B24 manifest 开始时间22:50:46，仍使用aa6354d实现。
- 2026-09-10 10:15:47：最终队列及次指标完成，完成证据锚定e3e11d0。

来源：[原始 note](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_terminal_freeze/recovery/RECOVERY_NOTE.txt)、
[原队列状态](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_terminal_freeze/recovery/queue_status.interrupted.json)、
[被拒绝恢复状态](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_terminal_freeze/recovery/queue_status.restart_attempt_failed.json)、
[完整B24 manifest](https://github.com/gao-666/YOLO-Master/blob/fedc360e73cdbfeaed930b17365458f632b1abc8/experiments/rhino_d2/results/p2_normalized_response_formal/formal/formal-B-s20260824/manifest.json)。

## 对复现及解释的影响

恢复是 fresh restart，不是中途 checkpoint resume。中断段的epoch、验证结果和权重不拼接、不进入最终paired统计。
正式三组审计只覆盖保留的六个完整arm；机器实际总计算还包含中断B24及工程pilot，不能将二者隐藏在matched-compute称谓下。
原 `INTERRUPTION_HASHES.txt` 显示的路径被格式化截断，本次另生成可机器读取的完整路径清单，不修改原记录。
档案支持当时记录了人工中断和恢复，但不能独立证明操作者身份、授权过程或未查看中间指标；如评审要求，当时操作记录仍应补充。

这是一项已披露的协议执行偏离。恢复沿用原注册seed，中断段不进入最终统计；现有档案不足以独立排除中间指标对恢复决策的影响。它不是允许今后自动重试的先例。
