# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation-driven dynamic top-k annealing for MoE inference efficiency.

Existing MoE schedulers tune the auxiliary *balance-loss coefficient*. This
module instead anneals the number of *active* experts (``top_k``) during
training, guided jointly by validation-mAP saturation and expert-usage balance
(Gini). Once accuracy has plateaued and load is balanced, dropping the lowest-
weighted expert per token costs little accuracy while cutting inference FLOPs -
the core "compute-on-demand" premise of ES-MoE. A rollback guard reverts a step
that hurts mAP, so capacity is never removed unsafely.

Complements (does not replace) the balance-loss schedulers: use both together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


def apply_top_k(model: torch.nn.Module, k: int) -> int:
    """Set the active-expert count on every compatible MoE module. Returns count."""
    updated = 0
    k = int(k)
    for module in model.modules():
        if hasattr(module, "top_k"):
            # clamp to the module's own expert budget when known
            n = int(getattr(module, "num_experts", k))
            module.top_k = max(1, min(k, n))
            updated += 1
    return updated


@dataclass
class TopKAnnealConfig:
    """Configuration for :class:`DynamicTopKScheduler` (all opt-in, defaults safe)."""

    enabled: bool = False          # off by default -> fully backward compatible
    init_top_k: int = 2
    min_top_k: int = 1
    window: int = 3                # sliding window (val epochs) for saturation
    map_eps: float = 1e-3          # mAP gain below this over a window == saturated
    gini_ceiling: float = 0.30     # only anneal when load is at-or-below this Gini
    hold: int = 2                  # consecutive qualifying epochs before a step
    rollback_drop: float = 5e-3    # mAP drop after a step that triggers rollback
    ema_momentum: float = 0.8


@dataclass
class TopKAnnealState:
    """Serializable snapshot emitted each step (for logging / results CSV)."""

    top_k: int
    ema_gini: float
    val_map: float
    saturated: bool
    balanced: bool
    action: str                    # "hold" | "anneal" | "rollback" | "floor"

    def to_dict(self) -> dict:
        return {
            "topk_top_k": self.top_k, "topk_ema_gini": self.ema_gini,
            "topk_val_map": self.val_map, "topk_saturated": self.saturated,
            "topk_balanced": self.balanced, "topk_action": self.action,
        }


@dataclass
class DynamicTopKScheduler:
    """Anneal ``top_k`` from val-mAP saturation + expert-balance (Gini).

    Formula (evaluated once per validation epoch):
        saturated_t = max(mAP[t-w+1:t+1]) - max(mAP[t-2w+1:t-w+1]) < map_eps
        balanced_t  = ema_gini_t <= gini_ceiling
        if saturated_t and balanced_t for `hold` consecutive epochs and k>min_k:
            k <- k - 1                         # anneal: fewer active experts
        if mAP dropped by > rollback_drop right after an anneal:
            k <- k + 1 and freeze              # safety rollback

    ``ema_gini_t = m*ema_gini_{t-1} + (1-m)*gini_t`` smooths routing noise.
    """

    cfg: TopKAnnealConfig = field(default_factory=TopKAnnealConfig)
    top_k: int = 0
    ema_gini: float | None = None
    _maps: List[float] = field(default_factory=list)
    _qualify: int = 0
    _last_action_annealed: bool = False
    _map_before_step: float = 0.0
    _frozen: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            self.top_k = int(self.cfg.init_top_k)

    def _saturated(self) -> bool:
        w = self.cfg.window
        if len(self._maps) < 2 * w:
            return False
        recent = max(self._maps[-w:])
        prev = max(self._maps[-2 * w:-w])
        return (recent - prev) < self.cfg.map_eps

    def step(self, val_map: float, gini: float, model: torch.nn.Module | None = None) -> TopKAnnealState:
        """Advance one validation epoch; optionally apply the new top_k to ``model``."""
        val_map = float(val_map)
        gini = float(max(gini, 0.0))
        m = self.cfg.ema_momentum
        self.ema_gini = gini if self.ema_gini is None else m * self.ema_gini + (1 - m) * gini
        self._maps.append(val_map)

        action = "hold"
        if not self.cfg.enabled or self._frozen:
            return self._emit(val_map, action)

        # rollback: previous epoch we annealed and mAP dropped materially
        if self._last_action_annealed and (self._map_before_step - val_map) > self.cfg.rollback_drop:
            self.top_k = min(self.top_k + 1, int(self.cfg.init_top_k))
            self._frozen = True
            self._last_action_annealed = False
            action = "rollback"
            self._apply(model)
            return self._emit(val_map, action)
        self._last_action_annealed = False

        saturated = self._saturated()
        balanced = self.ema_gini <= self.cfg.gini_ceiling
        if saturated and balanced:
            self._qualify += 1
        else:
            self._qualify = 0

        if self._qualify >= self.cfg.hold and self.top_k > self.cfg.min_top_k:
            self._map_before_step = val_map
            self.top_k -= 1
            self._qualify = 0
            self._last_action_annealed = True
            action = "anneal"
            self._apply(model)
        elif self.top_k <= self.cfg.min_top_k:
            action = "floor"

        return self._emit(val_map, action, saturated, balanced)

    def _apply(self, model) -> None:
        if model is not None:
            apply_top_k(model, self.top_k)

    def _emit(self, val_map, action, saturated=False, balanced=False) -> TopKAnnealState:
        return TopKAnnealState(
            top_k=int(self.top_k), ema_gini=float(self.ema_gini or 0.0),
            val_map=val_map, saturated=bool(saturated), balanced=bool(balanced), action=action,
        )

    def state_dict(self) -> dict:
        return {"top_k": self.top_k, "ema_gini": self.ema_gini, "maps": list(self._maps),
                "qualify": self._qualify, "frozen": self._frozen}

    def load_state_dict(self, s: dict) -> None:
        self.top_k = int(s.get("top_k", self.top_k))
        self.ema_gini = s.get("ema_gini")
        self._maps = list(s.get("maps", []))
        self._qualify = int(s.get("qualify", 0))
        self._frozen = bool(s.get("frozen", False))


# ---- 训练集成:val-epoch 回调(镜像 utils/callbacks/moe_diag.py 的防御式模式)----

def _val_map_from_trainer(trainer) -> float:
    """从 trainer 取验证 mAP(优先 mAP50-95,回退 fitness)。"""
    m = getattr(trainer, "metrics", None) or {}
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50(B)"):
        if isinstance(m, dict) and k in m:
            return float(m[k])
    fit = getattr(trainer, "fitness", None)
    return float(fit) if fit is not None else 0.0


def create_topk_anneal_callback(cfg: TopKAnnealConfig | None = None):
    """返回 on_fit_epoch_end(trainer) 回调:每验证 epoch 按 mAP饱和+Gini 退火 top_k。

    防御式:无 MoE 模块 / 缺依赖时静默跳过,不影响训练。opt-in(cfg.enabled)。
    """
    sched = DynamicTopKScheduler(cfg=cfg or TopKAnnealConfig())

    def on_fit_epoch_end(trainer):
        try:
            if not sched.cfg.enabled:
                return
            from ultralytics.nn.modules.moe.schedule import mean_usage_gini_from_model
            gini = mean_usage_gini_from_model(trainer.model)
            val_map = _val_map_from_trainer(trainer)
            st = sched.step(val_map, gini, trainer.model)
            setattr(trainer, "_topk_anneal_state", st.to_dict())
            if st.action in ("anneal", "rollback"):
                msg = f"[TopKAnneal] epoch={getattr(trainer,'epoch',0)} {st.action} -> top_k={st.top_k} (mAP={val_map:.4f} gini={st.ema_gini:.3f})"
                log = getattr(trainer, "console", None) or __import__("logging").getLogger("ultralytics")
                (log.info if hasattr(log, "info") else print)(msg)
        except Exception as exc:  # 永不因调度器中断训练
            setattr(trainer, "_topk_anneal_error", str(exc))

    on_fit_epoch_end._scheduler = sched  # 便于外部读状态/存 ckpt
    return on_fit_epoch_end
