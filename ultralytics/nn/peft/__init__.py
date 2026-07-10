# 🐧 YOLO-Master PEFT Package — Parameter-Efficient Fine-Tuning
# Copyright (C) 2026 Tencent. All rights reserved.
"""Parameter-Efficient Fine-Tuning (PEFT) subpackage for YOLO-Master.

Currently provides:
- **MoLoRA** (Mixture-of-LoRA): Multi-expert LoRA with learned routing,
  spatial/linear/hybrid routers, MoE-aware rank allocation, and
  auxiliary load-balancing / diversity losses.

Import path::

    from ultralytics.nn.peft.molora import get_peft_molora_model, MoLoRAConfig
"""

from ultralytics.nn.peft.molora import *  # noqa: F401,F403
from ultralytics.nn.peft.molora import __all__ as _molora_all

__all__ = _molora_all
