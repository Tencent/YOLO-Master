# ADMISSION — C3 | V-PEFT 工业缺陷小样本实战 | 准入检查 (2026-08-26)

## 1. 本次验收结论

| # | 验收项 | 结果 | 证据位置 |
|---|---|---|---|
| 1 | 环境安装 | PASS | §2, §4.1 |
| 2 | 基线/最小任务定义 | PASS | §3 |
| 3 | V-PEFT 最小训练跑通(planner 开启) | PASS | §5, `smoke/c3/runs/c3_vpeft_smoke_20260826_161145/` |
| 4 | 能解释一次 planner 输出 | PASS | §5.1(ACCEPT, 95 targets, 预算使用 10.6%) |
| 5 | 配置文件留档 | PASS | `train/vpeft_on/args.yaml` |
| 6 | 完整日志 | PASS | `smoke/c3/logs/smoke_run.log`(SHA-256 见 §6) |
| 7 | 结果证据 | PASS | `evidence.json` + results.csv/png,§5 |
| 8 | 风险与降级登记 | PASS | §7 |

限定:本机 Apple Silicon(MPS),coco8 微型数据集,1 epoch。**本机 smoke 只证明最小闭环跑通;真实三方案同预算对照将在 GPU(RTX 4090 / A800)上执行(脚本已交付,§4.2)。**

## 2. 锁定环境与数据

| 项 | 锁定值 |
|---|---|
| OS | macOS(darwin), Apple Silicon M1 Pro |
| Python | 3.11(conda env `yolo_master`) |
| YOLO-Master | 本地 editable 安装(8.4.101,`ultralytics/__init__.py`) |
| PyTorch | 2.13.0(MPS 可用) |
| 推理/训练设备 | `mps`(本机无 CUDA;GPU 对照固定 CUDA) |
| AMP | 本机 smoke 为默认 `amp=True`;GPU 对照固定 `amp=false`(见 §7 风险) |
| 权重 | `YOLO-Master-EsMoE-N.pt`,SHA-256 `29e1b93f...88c32` |
| 数据 | `coco8.yaml`(Ultralytics 内置 4 图微型数据集) |

> 数据许可:本轮 coco8 为 Ultralytics 内置演示数据,许可随包分发。

NEU-DET 数据已锁定(2026-08-26):
- 来源:GitHub 镜像 `Marfbin/NEU-DET-with-yolov8`(YOLOv8 格式,1800 张/6 类,`crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`),**仓库无显式 LICENSE,学术使用,正式 P0 前建议确认官方 NEU Surface Defect Database 许可条款**。
- 划分:train=1620 / test=180(用作 val,与源 `data.yaml` 一致)。
- few-shot:**每类 k 张**分层采样(k5=30, k10=60, k50=300, k100=600),seed=824,确保 6 类全覆盖。
- 产物:`~/datasets/NEU-DET-yolo/neu_det_yolo_v3/`(`neu_det.yaml`、`shots/k{k}/train.txt`、`split_report.json` 含样本 SHA-256)。

## 3. 设计说明与代码边界

- **复用不改算法**:直接使用 `yolo detect train`、Trainer、LoRA 与 V-PEFT 既有实现;未修改 `ultralytics/`、`tests/`、原有 `scripts/`。
- **新增仅限 `smoke/c3/`**:
  - `smoke_c3_vpeft.py` — 准入 smoke 入口(planner + 1 epoch + 证据采集)
  - `run_smoke.py` — GPU 三策略统一运行器(vpeft / full_sft / frozen_backbone)
  - `prepare_neu_det.py` — NEU-DET 数据转换 + few-shot(5/10/50/100)划分
  - `docs/ADMISSION_20260826.md` — 本文档
- **最小任务三要素**:数据=coco8(4 图);资源=MPS, batch=4, imgsz=320;条件=1 epoch, `lora_planner_enabled=True`, `lora_planner_backend="vpeft"`, `lora_adapter_budget=2,000,000`, rank=8。
- **禁止覆盖**:所有运行目录按时间戳/`--name` 隔离,复跑不覆盖。

## 4. 复现命令

### 4.1 环境安装(本机)

```bash
conda create -n yolo_master python=3.11 -y
conda install -n yolo_master pytorch torchvision -c pytorch -y
conda run -n yolo_master pip install -r requirements.txt
conda run -n yolo_master pip install -e .
```

### 4.2 准入 smoke(本机 MPS)

```bash
cd YOLO-Master
conda run -n yolo_master python smoke/c3/smoke_c3_vpeft.py --epochs 1 --imgsz 320 --batch 4 --device mps
# 输出: smoke/c3/runs/c3_vpeft_smoke_<ts>/evidence.json 及完整训练产物
```

### 4.3 GPU 三方案同预算对照(4090 / A800,已交付,下一步执行)

```bash
# 1) 准备 NEU-DET 数据 + few-shot 划分
python smoke/c3/prepare_neu_det.py --src <NEU-DET原始目录> --out <neu_det_yolo> --shots 5,10,50,100 --seed 824

# 2) 三策略各跑一次(拒绝覆盖同名目录,必须换 --name)
python smoke/c3/run_smoke.py --strategy vpeft           --data <neu_det_yolo>/neu_det.yaml --name k5_vpeft    --epochs 1 --batch 8 --imgsz 640 --device 0 --amp false
python smoke/c3/run_smoke.py --strategy full_sft        --data <neu_det_yolo>/neu_det.yaml --name k5_full     --epochs 1 --batch 8 --imgsz 640 --device 0 --amp false
python smoke/c3/run_smoke.py --strategy frozen_backbone --data <neu_det_yolo>/neu_det.yaml --name k5_frozen   --epochs 1 --batch 8 --imgsz 640 --device 0 --amp false
```

每个运行目录含:实际命令、完整日志、`resolved_config`、指标、显存采样、退出码。

## 5. 结果证据

| 维度 | 数值 | 说明 |
|---|---|---|
| Planner 决策 | **ACCEPT** | 预算内找到可行放置,未回退 |
| Planner 目标数 | 95 个模块,全部 rank=8, variant=lora | 含 stem/backbone/neck/head、MoE routing 网络 |
| 适配器预算 | 上限 2,000,000,实际 **212,696**(10.6%) | AO 求解器 3.11s,效用 47.5 |
| 实际适配器参数 | **164,864(占总参数 5.478%)** | 训练日志 `[LoRA] Stats` |
| 总可训练参数 | 544,400(18.089%) | 适配器 164,864 + 检测头 unfreeze 379,536 |
| 模型规模 | 360 layers, 2,844,648 params | YOLO-master-n |
| 单 epoch 耗时 | 108.5s(含 MPS 编译与验证) | coco8, batch=4, imgsz=320 |
| mAP50 / mAP50-95 | 0.0 / 0.0 | 见免责声明 ↓ |
| 内存(RSS) | 0.27 GB → 0.80 GB | MPS 统一内存,无独立显存 |

**免责声明**:单 epoch、单 seed、4 图 coco8 的指标只证明最小闭环跑通,**不用于评价收敛、稳定收益或方案优劣**;方案对照与收敛分析全部留到 GPU 上的 NEU-DET/DeepPCB 任务(§9)。

### 5.1 Planner 输出解释(本次验收核心)

```
[planner 输出解释] status=ACCEPT -> 约束下找到可行放置:按规划的 rank 注入 LoRA 适配器。
  planner_backend=vpeft, solver=ao
  adapter_budget(max_params)=2,000,000
  放置适配器数=95, 合计 rank=760(95×8)
  目标模块示例:0.conv, 1.conv, 2.cv1.conv, ..., 24.m.0.m.1.cv2.conv(含 MoE routing_network)
  约束: hard=[C_op, C_sem, C_budget, C_deploy, C_compat, C_moe, C_div], soft=[C_budget, C_deploy]
```

解读要点(可对答辩/导师陈述):
1. **决策是 ACCEPT 而非 REFUSE**:预算 200 万下,planner 认为 95 个卷积/路由模块都可以放 LoRA 且满足全部硬约束(算子兼容 C_op、语义 C_sem、预算 C_budget、部署 C_deploy、兼容 C_compat、MoE 结构 C_moe、多样性 C_div)。
2. **变体选择被优化**:求解器在 `ia3` 与 `lora` 两个候选间选择(AO 求解器 `n_variant_candidates=2, optimize_variant=true`),最终统一选择 `lora`。
3. **预算使用率仅 10.6%**:说明在 rank=8 下该小模型预算宽松;下一步可在 GPU 上提高 rank 或预算观察效用-成本曲线。
4. **自动排除 30 个 grouped conv**(`[LoRA] Automatically excluded 30 incompatible grouped conv layers (r=8)`),这是结构约束 C_compat 的体现——planner 不会强行在 grouped conv 上放置 rank=8。
5. **95 targets 中 rank 全部为 8**:当前配置下求解器没有进行 rank 分化;后续 P1 对照中可记录 `lora_rank_pattern` 变化。

## 6. 完整日志与证据索引(仓库相对路径)

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 完整运行日志 | `smoke/c3/logs/smoke_run.log` | `a14595d2eb5f7204c85fb7de137498a84b23ce0379b7c26d77530d05d69631f9` |
| 结构化证据 | `smoke/c3/runs/c3_vpeft_smoke_20260826_161145/evidence.json` | `6ff5fb5ed4825ef06424324130c0c645ac4b0f58f3ef8b68d4b57cf4c256e78d` |
| 最终配置 | `.../train/vpeft_on/args.yaml` | `e8efe9f1fc3c51ecd483f4081c210ead621ac7b4fe4d91275e398f61fcccf4b6` |
| 训练曲线 | `.../train/vpeft_on/results.png`、`results.csv` | 随运行目录 |
| 权重 | `YOLO-Master-EsMoE-N.pt` | `29e1b93f09b16c8cf7c402f36dcaafc19d4812155631ed45b769e941e4c88c32` |

## 7. 风险与降级

| 风险 | 本次状态 | 降级与恢复规则 |
|---|---|---|
| 本机无 CUDA GPU | 已解除(MPS 跑通) | 真实三方案对照在 4090/A800 执行;设备不可用则报障,不以 CPU 结果顶替 |
| MPS 非确定性警告(`index_put_with_accumulate_mps`) | 已记录 | GPU 对照固定 `seed=824` + CUDA;若仍有差异以 `yolo checks` 与多 seed 复测处理 |
| AMP 非有限值 | 不在本机范围(默认 amp=True 未异常) | GPU 对照固定 `amp=false`,AMP 结果不混入判定 |
| V-PEFT 回退(FALLBACK) | 未发生(ACCEPT) | strict 模式(GPU 脚本 vpeft 策略 `lora_vpeft_strict=True`):若 strict 失败/后端非 vpeft/目标数 0 → 判定失败并提交 issue |
| grouped conv 不兼容(30 层) | 自动排除,受控 | 不关闭 strict 掩盖;在文档中如实记录排除清单 |
| planner 对工业缺陷任务有隐含假设 | 待 GPU 任务验证 | 按任务书:提交 issue,降级为手动 V-PEFT 配方(固定 rank/变体) |
| 数据许可 | coco8 内置;NEU-DET/DeepPCB 待确认 | 只认官方页面/许可条款,数据归档记 SHA-256 |
| 结果过度解读 | 已控制 | 本 smoke 不发布任何收敛/收益结论 |
| 日志/版本漂移 | 已控制 | 运行目录不覆盖;全链路哈希留证 |

## 8. 最终结论

规定范围内全部 PASS。V-PEFT 最小闭环已在 coco8(1 epoch,MPS)跑通,planner 输出可解释(ACCEPT / 95 targets / 预算使用 10.6%),环境与产物已哈希锁定。

## 9. 下一阶段 P0(不纳入本次验收)

- [x] NEU-DET 数据准备完成:`~/datasets/NEU-DET-yolo/neu_det_yolo_v3/`,few-shot 每类 5/10/50/100(seed=824,6 类全覆盖)
- [ ] NEU-DET 官方许可条款确认(当前为无 LICENSE 的 GitHub 镜像,正式 P0 前完成)
- [ ] GPU(4090/A800)上 NEU-DET 三方案同预算对照:vpeft / full_sft / frozen_backbone,FP32,1 epoch(脚本:`run_smoke.py`)
- [ ] 记录四维证据:可训练参数比、显存峰值、耗时、mAP50-95(≥3 seed)
- [ ] DeepPCB 数据集接入,达到任务书 P0(双数据集各跑通一次 V-PEFT)
- [ ] 将本 smoke 与结果 PR 到个人 fork(`smoke/c3/`),更新登记表
