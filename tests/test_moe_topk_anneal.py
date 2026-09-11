# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the dynamic top-k annealing MoE scheduler."""

from ultralytics.nn.modules.moe.topk_anneal import (
    DynamicTopKScheduler,
    TopKAnnealConfig,
    create_topk_anneal_callback,
    _val_map_from_trainer,
)


def test_disabled_is_backward_compatible():
    s = DynamicTopKScheduler(cfg=TopKAnnealConfig(enabled=False, init_top_k=2))
    for _ in range(20):
        st = s.step(val_map=0.4, gini=0.1)
    assert s.top_k == 2 and st.action == "hold"


def test_anneals_on_saturation_and_balance():
    cfg = TopKAnnealConfig(enabled=True, init_top_k=3, min_top_k=1, window=2,
                           map_eps=1e-3, gini_ceiling=0.3, hold=2, ema_momentum=0.0)
    s = DynamicTopKScheduler(cfg=cfg)
    actions = [s.step(val_map=0.40, gini=0.1).action for _ in range(10)]
    assert "anneal" in actions and 1 <= s.top_k < 3


def test_no_anneal_when_imbalanced():
    cfg = TopKAnnealConfig(enabled=True, init_top_k=3, window=2, gini_ceiling=0.3, hold=2, ema_momentum=0.0)
    s = DynamicTopKScheduler(cfg=cfg)
    for _ in range(10):
        s.step(val_map=0.40, gini=0.6)
    assert s.top_k == 3


def test_no_anneal_while_improving():
    cfg = TopKAnnealConfig(enabled=True, init_top_k=3, window=2, map_eps=1e-3, gini_ceiling=0.3, hold=2, ema_momentum=0.0)
    s = DynamicTopKScheduler(cfg=cfg)
    m = 0.30
    for _ in range(10):
        m += 0.02
        s.step(val_map=m, gini=0.1)
    assert s.top_k == 3


def test_rollback_on_map_drop():
    cfg = TopKAnnealConfig(enabled=True, init_top_k=3, min_top_k=1, window=2, map_eps=1e-3,
                           gini_ceiling=0.3, hold=2, rollback_drop=5e-3, ema_momentum=0.0)
    s = DynamicTopKScheduler(cfg=cfg)
    k_after = None
    for _ in range(8):
        st = s.step(val_map=0.40, gini=0.1)
        if st.action == "anneal":
            k_after = s.top_k
            break
    assert k_after is not None and k_after < 3
    st = s.step(val_map=0.30, gini=0.1)
    assert st.action == "rollback" and s.top_k == k_after + 1
    for _ in range(5):
        s.step(val_map=0.40, gini=0.1)
    assert s.top_k == k_after + 1  # frozen


def test_state_dict_roundtrip():
    s = DynamicTopKScheduler(cfg=TopKAnnealConfig(enabled=True, init_top_k=2))
    for _ in range(4):
        s.step(val_map=0.4, gini=0.1)
    s2 = DynamicTopKScheduler(cfg=TopKAnnealConfig(enabled=True))
    s2.load_state_dict(s.state_dict())
    assert s2.top_k == s.top_k and s2._maps == s._maps


class _MockModel:
    def modules(self):
        return []


class _MockTrainer:
    def __init__(self):
        self.model = _MockModel()
        self.epoch = 0
        self.metrics = {"metrics/mAP50-95(B)": 0.4}
        self.fitness = 0.4


def test_callback_disabled_is_noop():
    cb = create_topk_anneal_callback(TopKAnnealConfig(enabled=False))
    t = _MockTrainer()
    for i in range(10):
        t.epoch = i
        cb(t)
    assert cb._scheduler.top_k == cb._scheduler.cfg.init_top_k
    assert not hasattr(t, "_topk_anneal_error")


def test_callback_never_raises():
    cb = create_topk_anneal_callback(TopKAnnealConfig(enabled=True, init_top_k=3))
    t = _MockTrainer()
    for i in range(6):
        t.epoch = i
        cb(t)  # must never raise, even if optional deps are missing
    assert cb._scheduler is not None


def test_val_map_extraction():
    class T:
        metrics = {"metrics/mAP50-95(B)": 0.37}
        fitness = 0.2
    assert abs(_val_map_from_trainer(T()) - 0.37) < 1e-9

    class T2:
        metrics = {}
        fitness = 0.29
    assert abs(_val_map_from_trainer(T2()) - 0.29) < 1e-9
