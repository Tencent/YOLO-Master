# 🐧Please note that this file has been modified by Tencent on 2026/07/29. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Cross-Scale Expert Sharing MoE: 跨尺度专家共享 MoE 模块.

创新点:
  - 多个 MoE 块共享同一组 expert pool
  - 减少参数 (~25-50% reduction)
  - 跨尺度特征共享通过 expert pool 自然实现
  - 论文价值: 讨论"参数共享 vs 性能"权衡

使用:
  在 YAML 中, 多个 SharedExpertMoE 块使用相同 pool_id 即可共享 experts:
  ```yaml
  backbone:
    - [-1, 1, SharedExpertMoE, [512, 4, 2, 0.5, "shared_512"]]  # 创建 pool
    - [-1, 1, SharedExpertMoE, [512, 4, 2, 0.5, "shared_512"]]  # 复用 pool
  ```
"""

from typing import Dict

from .modules import LowRankHybridAdaptiveGateMoE

# Build-time expert pool registry. Model parsing clears it at model boundaries;
# modules keep the shared expert object as a registered child after construction.
# Key: pool_id, Value: dict{fused_experts, in_channels, out_channels, num_experts, top_k, bottleneck_ratio}
_SHARED_EXPERT_POOLS: Dict[str, dict] = {}


class SharedExpertMoE(LowRankHybridAdaptiveGateMoE):
    """
    Cross-Scale Expert Sharing MoE (v0.8 variant).

    多个 MoE 块共享同一组 expert pool, 实现跨尺度特征共享和参数效率.

    关键创新:
      1. Expert Pool 共享: 相同 pool_id 的 MoE 块共享 expert 参数
      2. 跨尺度特征传递: 浅层 P3 expert 可被深层 P4 复用
      3. 参数效率: 相比独立 MoE 块, 参数量减少 25-50%
      4. 训练稳定: 每个 expert 被多个尺度训练, 鲁棒性更强

    约束:
      - 共享 pool 的 MoE 块必须有相同 in_channels, out_channels, num_experts, top_k
      - 不同 channel size 应使用不同 pool_id
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
        pool_id: str = "shared",
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
            bottleneck_ratio,
        )
        self.pool_id = pool_id
        self._is_pool_owner = False
        self._setup_shared_pool()

    def _setup_shared_pool(self):
        """设置共享 expert pool. 第一个该 pool_id 的块是 owner, 后续块复用."""
        pool_key = self.pool_id
        pool_signature = {
            "in_channels": self.dynamic_channels,
            "out_channels": self.out_dynamic,
            "num_experts": self.fused_experts.num_experts if hasattr(self.fused_experts, "num_experts") else 0,
            "top_k": self.top_k,
            "bottleneck_ratio": self.bottleneck_ratio,
        }

        if pool_key in _SHARED_EXPERT_POOLS:
            existing = _SHARED_EXPERT_POOLS[pool_key]
            # 验证参数兼容性
            for k, v in pool_signature.items():
                if k in existing and existing[k] != v:
                    raise ValueError(
                        f"SharedExpertMoE pool '{pool_key}' parameter mismatch: "
                        f"{k} expected {existing[k]}, got {v}. "
                        f"共享 pool 的 MoE 块必须有相同的 channels/num_experts/top_k."
                    )
            # 复用现有 expert pool
            self.fused_experts = existing["fused_experts"]
            self._is_pool_owner = False
        else:
            # 注册新的 expert pool
            _SHARED_EXPERT_POOLS[pool_key] = {
                **pool_signature,
                "fused_experts": self.fused_experts,
            }
            self._is_pool_owner = True

    @classmethod
    def reset_shared_pools(cls):
        """Clear the build-time registry before or after constructing a model."""
        _SHARED_EXPERT_POOLS.clear()

    def get_pool_info(self):
        """获取当前 pool 的信息 (用于诊断)."""
        return {
            "pool_id": self.pool_id,
            "is_owner": self._is_pool_owner,
            "num_experts": self.fused_experts.num_experts if hasattr(self.fused_experts, "num_experts") else 0,
            "top_k": self.top_k,
            "dynamic_channels": self.dynamic_channels,
        }
