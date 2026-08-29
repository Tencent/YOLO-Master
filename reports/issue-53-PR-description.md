# PR: MoA 边界测试补全与垂类训练验证 (Issue #53)

## 概述

系统补全 Mixture of Attention (MoA) 模块的边界测试，提升测试覆盖率从 ~85% 到 **97%**，并完成 VisDrone 垂类训练验证。

## 变更内容

### 1. 测试补全 (`tests/test_moa.py`)

新增 **30 项**边界测试（约 500 行），覆盖所有 4 类要求场景：

#### 1.1 NeckMoAFusion 跨刻度尺寸不匹配
- `test_neck_moa_fusion_extreme_upsample` — 6 组极端 case（64× 上采样、1×1 lo、非方形 19×23、奇数尺寸 11×27）
- `test_neck_moa_fusion_equal_spatial_dims_no_interpolate` — hi=lo 时跳过插值
- `test_neck_moa_fusion_large_channel_mismatch` — 大通道差 (256→32→128)
- ✅ 所有用例前向稳定、形状保持、梯度正常

#### 1.2 MoABlock 温度梯度数值稳定性
- `test_moa_block_extreme_temperature_gradient` — temp < 1e-8，验证梯度不爆炸（每个参数 grad_max < 100）
- `test_moa_router_temperature_gradient_smoothness` — 9 档温度 (1e-8 ~ 100)，梯度平滑不消失
- `test_moa_router_softmax_prob_sum_robustness` — 极端 logit 下 softmax 概率和恒为 1
- ✅ 无 NaN、梯度稳定

#### 1.3 _safe_groups 边界 — num_heads 非整除
- `test_safe_groups_prime_channels_local_head` — dim∈{17,19,23,29,31,37,41} 质数通道，_safe_groups 正确降级到 1
- `test_safe_groups_prime_channels_global_head` — GlobalHead 同样覆盖
- `test_safe_groups_composite_channels` — 13 组 (channels, desired) → expected 验证
- `test_regional_attn_head_pool_stride_larger_than_spatial` — stride=3 > H=2
- ✅ GroupNorm→LayerNorm 降级正确，无 IndexError

#### 1.4 C2fMoA aux_loss 防重复计数
- `test_c2fmoa_triple_nesting_no_double_count` — 3 层嵌套（Outer→Mid→C2fMoA），covered_modules 正确跳过子模块
- `test_c2fmoa_aux_loss_independent_of_module_wrapping_order` — 包装顺序不变性
- ✅ 已通过 `publish_aux_loss(covered_modules=self.m)` 机制，无 MOE_LOSS_REGISTRY 式双计数

### 2. 缺陷排查

**未发现严重缺陷。** 所有 95 个边界测试通过：
- softmax 在 temp=1e-12 时数值稳定
- 路由梯度在温度全范围内不爆炸不消失
- 质数通道数、stride>spatial 等极端参数处理正确
- FP16/BF16 混合精度路径无 NaN

### 3. 覆盖率报告

| 指标 | Before | After |
|------|--------|-------|
| 测试数 | 65 | **95** |
| MoA 模块覆盖率 | ~85% | **97%** |
| 未覆盖行 | ~80 | 18 |

详见 `reports/issue-53-coverage-report.md`

### 4. 垂类训练验证

- 数据集：VisDrone2019-DET (6471 train, 548 val)
- 模型：yolo-master-moa-n (3.55M) vs yolo-master-n (3.42M)
- CPU 验证训练：1 epoch @ imgsz=256，管线通过（数据加载、优化器 router 组识别、MoA forward/backward、aux_loss 收集）
- 完整 GPU 训练脚本：`scripts/issue53/train_visdrone_moa_vs_baseline.py`
- 训练日志：`reports/issue-53-training-validation.md`

## 文件清单

```
tests/test_moa.py                                    (+500 lines, 30 tests)
reports/issue-53-coverage-report.md                  (updated)
reports/issue-53-training-validation.md              (updated)
reports/issue-53-PR-description.md                   (this file)
scripts/issue53/train_visdrone_moa_vs_baseline.py    (new)
scripts/download_visdrone_local.py                   (new)
```

## 测试命令

```bash
# 运行全部 MoA 测试 + 覆盖率
cd YOLO-Master
pip install pytest-cov polars
python -m pytest tests/test_moa.py --cov=ultralytics/nn/modules/moa -v

# GPU 训练 (需要 VisDrone 数据集)
python scripts/issue53/train_visdrone_moa_vs_baseline.py \
    --epochs 50 --imgsz 640 --batch 16 --device 0
```
