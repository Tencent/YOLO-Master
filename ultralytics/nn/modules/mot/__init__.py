# 🐧 YOLO-Master MoT Module — Mixture of Transformers
# Copyright (C) 2026 Tencent. All rights reserved.
"""Mixture-of-Transformers (MoT) package.

Provides three Transformer expert variants and their C2f-style wrapper:
  - MoTBlock   : core MoT building block (3-expert routing)
  - C2fMoT     : C2f-style YAML-compatible wrapper (drop-in for C3k2/A2C2f)
  - collect_mot_aux_loss  : helper to collect router aux losses
  - anneal_mot_temperature: temperature annealing utility
"""

from .mot import MoTBlock, C2fMoT, collect_mot_aux_loss, anneal_mot_temperature

__all__ = ("MoTBlock", "C2fMoT", "collect_mot_aux_loss", "anneal_mot_temperature")
