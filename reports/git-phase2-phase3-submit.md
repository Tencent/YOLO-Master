# Git Phase 2/3 提交推送核验报告

## 核验结论

截至 2026-08-02，**尚未完成 push**。执行 `git fetch --prune` 后，本地与 upstream 已分叉：本地领先 3 个提交、落后 32 个提交。远端分支不包含本地 `HEAD`。

## 分支与远端

- 当前分支：`codex/mixture-optimization-batches`
- Upstream：`origin/codex/mixture-optimization-batches`
- Ahead/behind（`HEAD...@{upstream}`）：`3 32`
- 远端包含本地 HEAD：否
- Fetch：成功；`origin/codex/mixture-optimization-batches` 从 `31aa643` 更新至 `0818eff`

## 尚未推送的提交

| Commit | Subject |
|---|---|
| `9f581948a7061d180e182785cc1df2246e46e118` | `fix: clarify sparse routing diagnostics` |
| `11f542396af46551a3897833c5b39b91d7956e1c` | `fix: harden mixture routing correctness` |
| `31be313e379dc9c05a65a5ee9aea6830933459fd` | `fix: stabilize MoA layouts and MPS numerics` |

## Push 状态

- 本轮仅按要求执行最终核验，未执行 `git push`。
- 当前普通 push 将因远端领先 32 个提交而成为 non-fast-forward；需要先安全整合远端变更。
- 工作区同时存在未提交源码改动，因此未擅自 rebase、merge、stash、reset 或创建提交。

## `git status -sb`

```text
## codex/mixture-optimization-batches...origin/codex/mixture-optimization-batches [ahead 3, behind 32]
 M tests/test_config_drift_detector.py
 M tests/test_latent_mixture.py
 M tests/test_molora_routing_aware_merge.py
 M tests/test_molora_sparse_dispatch.py
 M tests/test_peft_adapters.py
 M tests/test_placement_plan_schema.py
 M tests/test_vpeft_lora_e2e.py
 M tools/config_drift_detector.py
 M ultralytics/cfg/default.yaml
 M ultralytics/nn/modules/latent_mixture.py
 M ultralytics/nn/modules/moe/__init__.py
 M ultralytics/nn/modules/moe/protocol.py
 M ultralytics/nn/peft/molora/layer.py
 M ultralytics/utils/lora/api.py
 M ultralytics/utils/lora/config.py
 M ultralytics/vpeft/__init__.py
 M ultralytics/vpeft/placement_plan.py
?? .session_tmps/
?? .workbuddy/
?? agent/logs/
?? output/
?? outputs/
?? tests/test_moe_facade_protocol.py
```

## 建议后续动作

在保留未提交用户改动的前提下，先完成或隔离剩余逻辑批次，然后将远端 32 个提交安全整合到当前分支，解决潜在冲突并重跑相关测试；确认 ahead/behind 后再普通 push。禁止 force push。
