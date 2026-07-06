# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import math
import weakref
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from .api import _compute_param_stats, _is_adapter_param

class LoraTrainingStrategy:
    """
    Advanced training strategies for LoRA fine-tuning.

    Provides 4 complementary strategies:
    1. Layer-wise Decay: Reduce LR for deeper layers (stabilizes early training)
    2. Alpha Warmup: Gradually increase lora_alpha (prevents initial instability)
    3. Orthogonal Regularization: Penalize rank collapse in A/B matrices
    4. Dynamic Dropout Scheduling: Increase dropout as training progresses
    """

    def __init__(self, model, config=None, epochs=100):
        self.model = model
        self.config = config or getattr(model, 'lora_config', None)
        self.epochs = epochs
        # Store original alpha values per layer.
        # Uses weakref.ref as key to survive model deepcopy / serialization,
        # unlike id(module) which becomes stale after copy.deepcopy() or
        # pickle round-trip.
        self._original_alphas: Dict[weakref.ref, Dict[str, Any]] = {}
        self._strategy_active = False
        self._warmup_required = False  # True when area-attention needs warmup

    # ── Strategy 1: Layer-wise LR decay ──
    @staticmethod
    def get_layer_decay_factors(model, total_layers=None, decay_rate=0.85) -> Dict[str, float]:
        """
        Compute per-layer LR multipliers with exponential decay by depth.

        Args:
            model: LoRA-enabled model
            total_layers: Total number of YOLO backbone+head blocks (auto-detected if None).
                For YOLO, this is the count of top-level Sequential children (typically ~23).
                NOT the count of all nn.Module descendants.
            decay_rate: Multiplicative factor per layer depth (0.8~0.95 typical)

        Returns:
            Dict mapping parameter name -> lr_multiplier
        """
        if total_layers is None:
            # Auto-detect from YOLO structure. YOLO wraps blocks as a nn.Sequential
            # under `.model` (or `.model.model` if wrapped by PeftProxy).
            # We count the top-level numbered blocks (0..N), not all descendants.
            candidate_roots = []
            for root_attr in ("model", "base_model"):
                cur = getattr(model, root_attr, None)
                # Descend through nested wrappers
                for _ in range(4):
                    if cur is None:
                        break
                    if hasattr(cur, "__len__"):
                        try:
                            n = len(cur)
                            if n > 1:
                                candidate_roots.append(n)
                                break
                        except TypeError:
                            pass
                    cur = getattr(cur, "model", None) or getattr(cur, "base_model", None)

            if candidate_roots:
                total_layers = max(candidate_roots)
            else:
                # Fallback: extract max top-level index from adapter parameter names
                max_idx = 0
                for name, _ in model.named_parameters():
                    if not _is_adapter_param(name):
                        continue
                    for p in name.split("."):
                        if p.isdigit():
                            max_idx = max(max_idx, int(p))
                            break
                total_layers = max(max_idx + 1, 10)  # Minimum 10 layers

        factors = {}
        for name, param in model.named_parameters():
            if not _is_adapter_param(name):
                continue
            # Extract layer index from name (e.g., "model.23.cv3.0.conv.lora_A.weight")
            parts = name.split(".")
            layer_idx = 0
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break

            # Normalize to [0, 1]
            normalized_depth = min(layer_idx / max(total_layers, 1), 1.0)
            # Exponential decay: shallow layers get higher LR
            factor = decay_rate ** normalized_depth
            factors[name] = factor

        return factors

    def apply_layer_decay_to_optimizer(self, optimizer, decay_rate=0.85) -> int:
        """
        Apply layer-wise LR decay to existing optimizer param groups.
        
        This function REPLACES the single LoRA param group with multiple
        param groups, each with a different LR based on layer depth.
        
        PyTorch optimizer requires one param_group per unique LR, so we group
        parameters by their layer index and create one param_group per layer.

        Returns:
            Number of parameters whose LR was adjusted
        """
        # Guardrail: very small decay_rate (e.g. 0.01) is almost always a config mistake.
        # Typical range is 0.8 ~ 0.95; anything below 0.5 collapses all layers to ~0 lr
        # and defeats the purpose of layer-wise decay.
        if decay_rate <= 0.0 or decay_rate > 1.0:
            LOGGER.warning(
                f"[LoRA-Strategy] ⚠️ Invalid lora_layer_decay={decay_rate}. "
                f"Must be in (0, 1]. Skipping layer decay."
            )
            return 0
        if decay_rate < 0.5:
            LOGGER.warning(
                f"[LoRA-Strategy] ⚠️ lora_layer_decay={decay_rate} is very aggressive. "
                f"Recommended range is 0.8~0.95. Deep layers will receive near-zero LR, "
                f"which typically causes adapter under-training (mAP collapse)."
            )

        factors = self.get_layer_decay_factors(self.model, decay_rate=decay_rate)
        if not factors:
            return 0

        # Find the LoRA param group index and its base_lr.
        # Build a name lookup once to avoid O(N*M) scan; then collect ALL LoRA
        # params (the earlier implementation had an off-by-one break that only
        # picked up the first LoRA parameter per group, collapsing everything
        # into a single bucket).
        name_by_id = {id(p): n for n, p in self.model.named_parameters()}

        lora_pg_idx = None
        base_lr = None
        lora_params_in_pg = []

        for idx, pg in enumerate(optimizer.param_groups):
            pg_has_lora = False
            for p in pg.get("params", []):
                name = name_by_id.get(id(p))
                if name is not None and _is_adapter_param(name):
                    pg_has_lora = True
                    lora_params_in_pg.append((name, p, idx))
            if pg_has_lora and base_lr is None:
                base_lr = pg.get('lr', None)
                lora_pg_idx = idx

        if base_lr is None or lora_pg_idx is None:
            LOGGER.warning("[LoRA-Strategy] No LoRA param group found for layer decay.")
            return 0

        # Group LoRA params by layer index for efficient param_group creation
        from collections import defaultdict
        layer_groups = defaultdict(list)
        
        for name, param, _ in lora_params_in_pg:
            factor = factors.get(name, 1.0)
            # Round factor to reduce number of param groups.
            # Use 1 decimal precision: this reduces group count from ~18 to ~3-5
            # while still preserving meaningful stratification across depths.
            # Previous 3-decimal precision created too many groups (18+), slowing optimizer.
            rounded_factor = round(factor, 1)
            layer_groups[rounded_factor].append(param)
        
        # Remove the original LoRA param group (remove from end to keep indices stable)
        # We need to rebuild param_groups since PyTorch doesn't support deletion
        original_groups = optimizer.param_groups.copy()
        
        # Create new param_groups list
        new_param_groups = []
        for idx, pg in enumerate(original_groups):
            if idx == lora_pg_idx:
                # Replace with multiple layer-specific groups
                for factor, params in sorted(layer_groups.items(), reverse=True):
                    new_lr = base_lr * factor
                    # Start with a copy of the original param_group
                    new_pg = {k: v for k, v in pg.items() if k != "params"}
                    new_pg["params"] = params
                    new_pg["lr"] = new_lr
                    new_pg["initial_lr"] = new_lr  # for warmup scheduler
                    new_param_groups.append(new_pg)
            else:
                new_param_groups.append(pg)
        
        # Replace optimizer's param_groups
        optimizer.param_groups = new_param_groups
        
        # Also rebuild state if necessary (state is keyed by parameter object, so it remains valid)
        # But we need to update the optimizer's internal _param_group map if it exists
        if hasattr(optimizer, '_param_groups'):
            optimizer._param_groups = optimizer.param_groups
        
        avg_factor = sum(factors.values()) / len(factors)
        min_factor = min(factors.values())
        max_factor = max(factors.values())

        # Sanity check: a single LR group means depth stratification failed entirely.
        # This typically happens when the layer index detector returns the same index
        # for every LoRA param (e.g. when all names come from a sub-module without a
        # leading digit), or when decay_rate is so extreme that all factors round to
        # the same bucket.
        if len(layer_groups) == 1:
            LOGGER.warning(
                f"[LoRA-Strategy] ⚠️ Layer decay produced only 1 LR group "
                f"(decay_rate={decay_rate}, factor_range=[{min_factor:.4f}, {max_factor:.4f}]). "
                f"Stratification is effectively disabled. Check module naming or raise decay_rate."
            )

        LOGGER.info(
            f"[LoRA-Strategy] 📐 Layer-wise LR decay applied (rate={decay_rate}): "
            f"{len(layer_groups)} LR groups, "
            f"avg_factor={avg_factor:.3f}, range=[{min_factor:.3f}, {max_factor:.3f}]"
        )
        self._layer_decay_factors = factors
        return len(factors)

    # ── Strategy 2: Alpha Warmup ──
    def prepare_alpha_warmup(self):
        """
        Store original alpha scales and set initial scale to 0.

        PEFT LoRA scaling = alpha / r. This function stores the target alpha value
        for each LoRA layer and temporarily sets effective alpha to 0.

        Handles multiple PEFT internal structures:
          - PEFT >= 0.13: LoraLayer with lora_alpha property (may be property or stored in peft_config dict)
          - PEFT < 0.13: Direct 'scaling' attribute
          - PEFT >= 0.18: lora_alpha and scaling are dicts keyed by adapter name (e.g. {'default': 8})
        """
        self._original_alphas.clear()
        found = False

        # Determine config-level defaults
        cfg_alpha = 32  # default
        cfg_r = 8       # default
        if self.config is not None:
            cfg_alpha = getattr(self.config, 'alpha', 32) or getattr(self.config, 'lora_alpha', 32) or 32
            cfg_r = getattr(self.config, 'r', 8) or getattr(self.config, 'lora_r', 8) or 8

        for module in self.model.modules():
            lora_a = getattr(module, 'lora_A', None)
            # Only process actual LoRA layers
            if lora_a is None:
                continue
            # PEFT >= 0.18 uses nn.ModuleDict for lora_A (e.g. {'default': Conv2d}).
            # Older PEFT stores lora_A as a single Parameter or Module with .weight.
            is_lora_layer = False
            if isinstance(lora_a, nn.ModuleDict):
                # Check that at least one adapter entry has a weight attribute
                is_lora_layer = any(hasattr(child, 'weight') for child in lora_a.values())
            elif hasattr(lora_a, 'weight'):
                is_lora_layer = True
            if not is_lora_layer:
                continue

            # Strategy: detect how to control scaling for this PEFT version
            la_attr = getattr(module, 'lora_alpha', None)
            lr_attr = getattr(module, 'r', None)
            sc_attr = getattr(module, 'scaling', None)

            # ── Path A: PEFT >= 0.18 dict-style lora_alpha / scaling ──
            if isinstance(la_attr, dict) and isinstance(sc_attr, dict):
                # Both are dicts keyed by adapter name (e.g. 'default')
                # scaling = alpha / r, so we control scaling dict directly
                adapter_name = list(la_attr.keys())[0] if la_attr else 'default'
                orig_alpha = float(la_attr.get(adapter_name, cfg_alpha))
                orig_scaling = float(sc_attr.get(adapter_name, orig_alpha / max(cfg_r, 1)))
                self._original_alphas[weakref.ref(module)] = {
                    '_type': 'scaling_dict',
                    'orig_alpha': orig_alpha,
                    'orig_scaling': orig_scaling,
                    'adapter_name': adapter_name,
                    'r': float(lr_attr) if isinstance(lr_attr, (int, float)) else float(cfg_r),
                }
                # Set scaling to 0 to disable LoRA contribution at start
                sc_attr[adapter_name] = 0.0
                found = True
                continue

            # ── Path B: Both lora_alpha and r are directly writable numbers (older PEFT) ──
            if (isinstance(la_attr, (int, float)) and isinstance(lr_attr, (int, float))
                    and lr_attr > 0):
                orig_alpha = float(la_attr)
                self._original_alphas[weakref.ref(module)] = {
                    '_type': 'direct',
                    'orig_alpha': orig_alpha,
                    'r': float(lr_attr),
                }
                # Set alpha to 0 (scaling becomes 0)
                module.lora_alpha = 0.0
                found = True
                continue

            # ── Path C: lora_alpha might be a property in newer PEFT, but we can try to set it ──
            if la_attr is not None:
                try:
                    _orig_alpha = float(la_attr)
                    _r = float(lr_attr) if isinstance(lr_attr, (int, float)) else float(cfg_r)
                    self._original_alphas[weakref.ref(module)] = {
                        '_type': 'property',
                        'orig_alpha': _orig_alpha,
                        'r': _r,
                    }
                    # Attempt to set; we'll verify in step
                    module.lora_alpha = 0.0
                    found = True
                    continue
                except (TypeError, ValueError, AttributeError):
                    pass

            # ── Path D: Has numeric 'scaling' attribute (older PEFT or custom) ──
            if isinstance(sc_attr, (int, float)) and sc_attr > 0:
                self._original_alphas[weakref.ref(module)] = {
                    '_type': 'scaling',
                    'orig_scaling': float(sc_attr),
                }
                module.scaling = 0.0
                found = True
                continue

            # ── Path E: Fallback - try to use peft_config dict if available ──
            peft_config = getattr(module, 'peft_config', None)
            if peft_config is not None:
                try:
                    if isinstance(peft_config, dict) and 'lora_alpha' in peft_config:
                        _orig_alpha = float(peft_config['lora_alpha'])
                        _r = float(peft_config.get('r', cfg_r))
                        self._original_alphas[weakref.ref(module)] = {
                            '_type': 'config_dict',
                            'orig_alpha': _orig_alpha,
                            'r': _r,
                            'module_ref': module,  # store ref to update dict
                        }
                        peft_config['lora_alpha'] = 0.0
                        found = True
                        continue
                except (TypeError, ValueError):
                    pass

        if found:
            self._strategy_active = True
            # Diagnostic: report the distribution of _type paths so users can quickly
            # verify that the PEFT-version-specific fallback is working correctly.
            from collections import Counter
            type_dist = Counter(v.get('_type', 'unknown') for v in self._original_alphas.values())
            type_summary = ", ".join(f"{t}={c}" for t, c in type_dist.most_common())
            LOGGER.info(
                f"[LoRA-Strategy] 🔥 Alpha warmup prepared ({len(self._original_alphas)} layers) "
                f"| path distribution: {type_summary}"
            )
        else:
            # Determine if this architecture critically depends on warmup.
            # YOLO12 Area-Attention (AAttn) requires alpha warmup to prevent NaN.
            has_area_attn = any(
                "AAttn" in m.__class__.__name__ or "A2C2f" in m.__class__.__name__
                for m in self.model.modules()
            )
            if has_area_attn:
                self._warmup_required = True
                LOGGER.error(
                    "[LoRA-Strategy] 🚨 CRITICAL: Alpha warmup FAILED on Area-Attention "
                    "architecture (YOLO12/A2C2f). This model REQUIRES warmup to prevent "
                    "training collapse (loss→0, NaN). No modifiable alpha attributes found "
                    "— likely a PEFT version incompatibility. Training will likely crash. "
                    "Please upgrade peft or set lora_alpha_warmup=0 to proceed at your own risk."
                )
            else:
                LOGGER.warning(
                    "[LoRA-Strategy] ⚠️ No modifiable alpha attributes found for warmup. "
                    "This usually indicates a PEFT version mismatch — alpha warmup will be "
                    "silently disabled but training will continue normally."
                )
        return found

    def step_alpha_warmup(self, epoch, warmup_epochs=5):
        """
        Update alpha scaling based on current epoch (cosine ramp-up).

        Returns current scale factor in [0, 1].
        """
        if not self._original_alphas:
            return 1.0

        progress = min(epoch / max(warmup_epochs, 1), 1.0)
        # Cosine ease-in: starts at 0, ends at 1
        current_scale = 0.5 * (1 - math.cos(math.pi * progress))

        updated = 0
        for module in self.model.modules():
            # Use weakref.ref for key lookup — survives deepcopy/serialization.
            # The weakref key matches if the module is the same object that was
            # registered during prepare_alpha_warmup().
            ref = weakref.ref(module)
            if ref not in self._original_alphas:
                continue

            orig = self._original_alphas[ref]
            _type = orig['_type']

            try:
                # ── Path A: scaling dict (PEFT >= 0.18) ──
                if _type == 'scaling_dict':
                    sc_attr = getattr(module, 'scaling', None)
                    if isinstance(sc_attr, dict):
                        adapter_name = orig.get('adapter_name', 'default')
                        orig_scaling = orig['orig_scaling']
                        sc_attr[adapter_name] = orig_scaling * current_scale
                        updated += 1
                    continue

                if _type == 'direct':
                    target_alpha = orig['orig_alpha'] * current_scale
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = float(target_alpha)
                        updated += 1

                elif _type == 'property':
                    target_alpha = orig['orig_alpha'] * current_scale
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = float(target_alpha)
                        # Verify the write actually stuck
                        actual = getattr(module, 'lora_alpha', None)
                        if actual is not None and abs(float(actual) - target_alpha) < 0.01:
                            updated += 1
                        else:
                            # Property is read-only, try scaling attribute as fallback
                            if hasattr(module, 'scaling'):
                                orig_scaling = orig['orig_alpha'] / orig['r']
                                module.scaling = orig_scaling * current_scale
                                updated += 1
                                # Update type for future steps
                                orig['_type'] = 'scaling_fallback'
                                orig['orig_scaling'] = orig_scaling

                elif _type == 'scaling' or _type == 'scaling_fallback':
                    orig_scaling = orig.get('orig_scaling', orig.get('orig_alpha', 1.0) / orig.get('r', 1.0))
                    if hasattr(module, 'scaling'):
                        module.scaling = orig_scaling * current_scale
                        updated += 1

                elif _type == 'config_dict':
                    target_alpha = orig['orig_alpha'] * current_scale
                    peft_config = getattr(module, 'peft_config', None)
                    if isinstance(peft_config, dict):
                        peft_config['lora_alpha'] = float(target_alpha)
                        updated += 1

            except Exception as e:
                LOGGER.debug(f"[LoRA-Strategy] Alpha warmup step failed for module {type(module).__name__}: {e}")
                continue

        return current_scale

    def finalize_alpha_warmup(self):
        """Restore all alphas to their original values."""
        restored = 0
        for module in self.model.modules():
            ref = weakref.ref(module)
            if ref not in self._original_alphas:
                continue
            orig = self._original_alphas[ref]
            _type = orig['_type']

            try:
                # ── Path A: scaling dict (PEFT >= 0.18) ──
                if _type == 'scaling_dict':
                    sc_attr = getattr(module, 'scaling', None)
                    if isinstance(sc_attr, dict):
                        adapter_name = orig.get('adapter_name', 'default')
                        sc_attr[adapter_name] = float(orig['orig_scaling'])
                        restored += 1
                    continue

                if _type in ('direct', 'property'):
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = float(orig['orig_alpha'])
                        restored += 1

                elif _type in ('scaling', 'scaling_fallback'):
                    if hasattr(module, 'scaling'):
                        module.scaling = float(orig.get('orig_scaling', orig.get('orig_alpha', 1.0) / orig.get('r', 1.0)))
                        restored += 1

                elif _type == 'config_dict':
                    peft_config = getattr(module, 'peft_config', None)
                    if isinstance(peft_config, dict):
                        peft_config['lora_alpha'] = float(orig['orig_alpha'])
                        restored += 1

            except Exception as e:
                LOGGER.debug(f"[LoRA-Strategy] Alpha warmup finalize failed for module {type(module).__name__}: {e}")
                continue

        LOGGER.info(f"[LoRA-Strategy] Alpha warmup finalized — {restored}/{len(self._original_alphas)} alphas restored.")
        self._strategy_active = False

    # ── Strategy 3: Orthogonal Regularization Loss ──
    # Chunk size for batched computation — reduces peak memory by processing
    # N layers at a time instead of accumulating all r×r matrices simultaneously.
    _ORTHO_CHUNK_SIZE = 32

    @staticmethod
    def compute_orthogonal_loss(model, weight=1e-4, chunk_size=None) -> torch.Tensor:
        """
        Compute regularization loss encouraging LoRA A/B matrices to stay orthogonal.

        Prevents rank collapse where A·B degenerates into a low-effective-rank product.
        
        Loss = λ × (Σ||A^T A - I||_F + Σ||B^T B - I||_F) / N_pairs
        
        OPTIMIZED (v2): Uses chunked computation to reduce peak GPU memory.
        Instead of accumulating all r×r intermediate matrices simultaneously,
        processes layers in chunks of ``chunk_size`` (default 32). Each chunk's
        loss is computed and immediately reduced to a scalar before the next
        chunk, keeping peak temporary memory at O(chunk_size × r²) instead of
        O(N_layers × r²).
        
        Args:
            model: LoRA-enabled model
            weight: Scaling factor for the loss
            chunk_size: Number of weight matrices per chunk (default: 32).
                Set to None to use the class default (_ORTHO_CHUNK_SIZE).

        Returns:
            Scalar tensor (orthogonal regularization loss)
        """
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
            
        if chunk_size is None:
            chunk_size = LoraTrainingStrategy._ORTHO_CHUNK_SIZE

        # Collect all (weight, is_A) pairs first, then process in chunks.
        # This avoids re-iterating model.modules() and allows controlled memory.
        weight_pairs = []  # List of (weight_tensor, is_matrix_A)

        def _collect_weights(attr, is_A):
            """Collect weight tensors from either a direct LoRA layer or a ModuleDict."""
            if attr is None:
                return
            if isinstance(attr, nn.ModuleDict):
                for child in attr.values():
                    if hasattr(child, 'weight') and child.weight.numel() > 0:
                        weight_pairs.append((child.weight, is_A))
            elif hasattr(attr, 'weight') and attr.weight.numel() > 0:
                weight_pairs.append((attr.weight, is_A))

        # CRITICAL FIX (P0): Do NOT detach() the weight tensors — that severs the
        # gradient graph and the orthogonal regularization becomes a no-op.
        for name, module in model.named_modules():
            lora_a = getattr(module, 'lora_A', None)
            if lora_a is not None:
                _collect_weights(lora_a, is_A=True)
            lora_b = getattr(module, 'lora_B', None)
            if lora_b is not None:
                _collect_weights(lora_b, is_A=False)

        if not weight_pairs:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        # Process in chunks to bound peak memory.
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        pair_count = 0

        for chunk_start in range(0, len(weight_pairs), chunk_size):
            chunk = weight_pairs[chunk_start:chunk_start + chunk_size]
            chunk_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            for A_w, is_A in chunk:
                # Keep gradient connection (no .detach()).
                W = A_w if A_w.dtype == torch.float32 else A_w.float()
                if is_A:
                    if W.dim() >= 2 and W.shape[0] > 0:
                        if W.dim() > 2:
                            W = W.reshape(W.shape[0], -1)
                        WW_T = torch.matmul(W, W.t())
                        rows = WW_T.shape[0]
                        ident = torch.eye(rows, device=device, dtype=WW_T.dtype)
                        chunk_loss = chunk_loss + torch.norm(WW_T - ident, p='fro')
                        pair_count += 1
                else:
                    if W.dim() >= 2 and W.shape[-1] > 0:
                        if W.dim() > 2:
                            W = W.reshape(W.shape[0], -1)
                        WT_W = torch.matmul(W.t(), W)
                        cols = WT_W.shape[0]
                        ident = torch.eye(cols, device=device, dtype=WT_W.dtype)
                        chunk_loss = chunk_loss + torch.norm(WT_W - ident, p='fro')
                        pair_count += 1

            # Accumulate chunk loss (scalar addition, no retained intermediates).
            total_loss = total_loss + chunk_loss

        if pair_count == 0:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        return weight * (total_loss / pair_count)

    # ── Strategy 4: Dynamic Dropout Scheduling ──
    _DROPOUT_WARNED = False  # class-level flag to emit warning only once
    _last_dropout_value = None  # Cache last applied dropout value to avoid redundant updates

    @staticmethod
    def update_dropout_schedule(model, epoch, epochs_total, 
                                  start_dropout=0.0, end_dropout=0.15,
                                  schedule_start_ratio=0.3) -> int:
        """
        Dynamically increase LoRA dropout rate as training progresses.
        
        In early phases, low dropout preserves gradient signal for learning.
        In later phases, higher dropout acts as regularizer preventing overfitting.

        Args:
            model: LoRA-enabled model
            epoch: Current epoch (0-indexed)
            epochs_total: Total number of training epochs
            start_dropout: Initial dropout rate
            end_dropout: Final dropout rate  
            schedule_start_ratio: When to start increasing (fraction of total)

        Returns:
            Number of dropout layers updated
        """
        # Sanity check: end must be >= start, otherwise the schedule is a no-op / decrease.
        if end_dropout < start_dropout:
            if not LoraTrainingStrategy._DROPOUT_WARNED:
                LOGGER.warning(
                    f"[LoRA-Strategy] ⚠️ lora_dropout_end={end_dropout} < lora_dropout={start_dropout}. "
                    f"Dynamic dropout schedule disabled (dropout would monotonically decrease)."
                )
                LoraTrainingStrategy._DROPOUT_WARNED = True
            return 0
        if not (0.0 <= start_dropout <= 1.0 and 0.0 <= end_dropout <= 1.0):
            if not LoraTrainingStrategy._DROPOUT_WARNED:
                LOGGER.warning(
                    f"[LoRA-Strategy] ⚠️ Invalid dropout range [{start_dropout}, {end_dropout}]. "
                    f"Must be within [0, 1]. Schedule disabled."
                )
                LoraTrainingStrategy._DROPOUT_WARNED = True
            return 0
        if not (0.0 <= schedule_start_ratio <= 1.0):
            if not LoraTrainingStrategy._DROPOUT_WARNED:
                LOGGER.warning(
                    f"[LoRA-Strategy] ⚠️ Invalid schedule_start_ratio={schedule_start_ratio}. "
                    f"Must be within [0, 1]. Schedule disabled."
                )
                LoraTrainingStrategy._DROPOUT_WARNED = True
            return 0

        schedule_start = int(epochs_total * schedule_start_ratio)
        if epoch < schedule_start:
            current_dropout = start_dropout
        else:
            # Linear interpolation after schedule starts
            progress = (epoch - schedule_start) / max(epochs_total - schedule_start, 1)
            current_dropout = start_dropout + (end_dropout - start_dropout) * min(progress, 1.0)

        # OPTIMIZATION: Skip redundant updates if dropout value hasn't changed
        if LoraTrainingStrategy._last_dropout_value is not None and \
           abs(LoraTrainingStrategy._last_dropout_value - current_dropout) < 1e-6:
            return 0  # No change needed
        
        LoraTrainingStrategy._last_dropout_value = current_dropout

        updated = 0
        for module in model.modules():
            # PEFT stores dropout as module.lora_dropout, which may be:
            #   - nn.Dropout directly
            #   - nn.ModuleDict containing a 'default' key → nn.Dropout
            drop_attr = getattr(module, 'lora_dropout', None)
            if drop_attr is None:
                continue

            if isinstance(drop_attr, torch.nn.Dropout):
                drop_attr.p = float(current_dropout)
                updated += 1
            elif hasattr(drop_attr, 'default') and isinstance(drop_attr.default, torch.nn.Dropout):
                drop_attr.default.p = float(current_dropout)
                updated += 1

        return updated


def get_lora_training_stats(model, svd_sample_ratio: float = 0.2, svd_max_layers: int = 20) -> Dict[str, Any]:
    """
    Gather comprehensive LoRA training statistics for monitoring.

    Returns a dict with metrics useful for TensorBoard/W&B logging.

    Args:
        model: LoRA-enabled model
        svd_sample_ratio: Fraction of LoRA layers to run SVD on for effective-rank
            estimation (default 0.2). Full-model SVD is expensive for large models.
        svd_max_layers: Hard cap on number of layers for SVD (default 20).
    """
    s = _compute_param_stats(model)
    stats = {
        'lora_enabled': getattr(model, 'lora_enabled', False),
        'total_params': s.total,
        'trainable_params': s.trainable,
        'lora_params': s.adapter,
        'frozen_params': s.frozen,
        'lora_modules': 0,
        'effective_rank_avg': 0.0,
        'norm_A_frobenius': 0.0,
        'norm_B_frobenius': 0.0,
    }

    # First pass: collect LoRA modules and cheap stats (Frobenius norms).
    # Handles both PEFT <0.18 (direct attr with .weight) and PEFT >=0.18 (ModuleDict).
    def _extract_weights(attr):
        """Return a list of (A_weight_tensor,) from either a direct LoRA layer or ModuleDict."""
        if attr is None:
            return []
        if isinstance(attr, nn.ModuleDict):
            return [child.weight for child in attr.values() if hasattr(child, 'weight')]
        if hasattr(attr, 'weight'):
            return [attr.weight]
        return []

    norm_A_sum = 0.0
    norm_B_sum = 0.0
    lora_module_count = 0
    lora_layers = []

    for module in model.modules():
        a_weights = _extract_weights(getattr(module, 'lora_A', None))
        b_weights = _extract_weights(getattr(module, 'lora_B', None))

        if a_weights:
            for A in a_weights:
                A_det = A.detach()
                norm_A_sum += torch.norm(A_det, p='fro').item()
                if A_det.dim() >= 2:
                    lora_layers.append(A_det)
            lora_module_count += 1

        if b_weights:
            for B in b_weights:
                norm_B_sum += torch.norm(B.detach(), p='fro').item()

    stats['lora_modules'] = lora_module_count
    if lora_module_count > 0:
        stats['norm_A_frobenius'] = norm_A_sum / lora_module_count
        stats['norm_B_frobenius'] = norm_B_sum / lora_module_count

        # Second pass: sampled SVD for effective rank (expensive operation).
        # Evenly sample across depth rather than random-sample so results are reproducible.
        if lora_layers:
            n_sample = min(svd_max_layers, max(1, int(len(lora_layers) * svd_sample_ratio)))
            step = max(1, len(lora_layers) // n_sample)
            sampled = lora_layers[::step][:n_sample]

            rank_values = []
            for A in sampled:
                try:
                    _, S, _ = torch.linalg.svd(A.float(), full_matrices=False)
                    if S.numel() == 0 or S[0].item() == 0:
                        continue
                    effective_rank = (S > 0.01 * S[0]).sum().item()
                    rank_values.append((A.shape[0], A.shape[1], effective_rank))
                except Exception as e:
                    LOGGER.debug(f"[LoRA-Stats] SVD failed on layer shape {tuple(A.shape)}: {e}")
                    continue

            if rank_values:
                avg_eff_rank = sum(r[2] for r in rank_values) / len(rank_values)
                avg_theoretical = sum(min(r[0], r[1]) for r in rank_values) / len(rank_values)
                stats['effective_rank_avg'] = avg_eff_rank / avg_theoretical if avg_theoretical > 0 else 0

    return stats


# Convenience import for math used in strategies
# (math is now imported at the top of the module — keep this comment for clarity)


def suggest_lora_config_for_dataset(
    num_images: Optional[int] = None,
    num_classes: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a LoRA hyperparameter recipe tuned to dataset scale.

    Returns a dict with recommended keys (``lora_r``, ``lora_alpha``,
    ``lora_lr_mult``, ``lora_layer_decay``, ``lora_alpha_warmup``,
    ``lora_ortho_weight``, ``lora_dropout``) plus a human-readable ``notes``
    string explaining the rationale.

    Empirical baseline (from project experiments):
        - VOC (16K+ images, 50ep, batch=128)        : LoRA r=16 beats Full SFT
        - African Wildlife (~1K images, 20ep)       : Full SFT ~= LoRA r=32
        - COCO128 (128 images, <10ep)               : LoRA not recommended

    Args:
        num_images: Training set image count. If None, no sizing advice is given.
        num_classes: Class count; used to estimate per-class sample density.
        epochs: Planned total training epochs.
        batch_size: Planned batch size.

    Returns:
        Dict of recommended hyperparameters + ``notes``.
    """
    rec: Dict[str, Any] = {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_lr_mult": 2.0,
        "lora_layer_decay": 0.9,
        "lora_alpha_warmup": 3,
        "lora_ortho_weight": 0.0,
        "lora_dropout": 0.05,
        "lora_dropout_end": 0.15,
    }
    notes = []

    if num_images is None:
        notes.append("No num_images provided - returning generic medium-dataset defaults.")
        rec["notes"] = " ".join(notes)
        return rec

    per_class = (num_images / num_classes) if num_classes else None

    if num_images < 500 or (per_class is not None and per_class < 5):
        rec.update({
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.0,
            "lora_alpha_warmup": 0,
            "lora_ortho_weight": 0.0,
            "lora_dropout": 0.02,
        })
        notes.append(
            "Small-dataset regime: LoRA often underperforms Full SFT here. "
            "If LoRA is still desired, use rank=32+ and compare against Full SFT baseline (lora_r=0)."
        )
    elif num_images < 5000:
        rec.update({
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.9,
            "lora_alpha_warmup": 3,
            "lora_ortho_weight": 1e-4,
            "lora_dropout": 0.05,
        })
        notes.append("Small/medium regime: rank=32 with orthogonal regularization recommended.")
    elif num_images < 20000:
        rec.update({
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.9,
            "lora_alpha_warmup": 3,
            "lora_ortho_weight": 1e-4,
        })
        notes.append("Medium regime: LoRA typically matches or exceeds Full SFT here.")
    else:
        rec.update({
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.85,
            "lora_alpha_warmup": 5,
            "lora_ortho_weight": 1e-4,
        })
        notes.append("Large regime: LoRA or DoRA recommended; adapter efficiency peaks here.")

    if epochs is not None and epochs < 20:
        notes.append(f"[warn] epochs={epochs} is below recommended 20+ for LoRA convergence.")
    if batch_size is not None and batch_size < 16:
        notes.append(f"[warn] batch={batch_size} is small; gradient noise will hurt LoRA more than Full SFT.")

    rec["notes"] = " ".join(notes)
    return rec



__all__ = ['LoraTrainingStrategy', 'get_lora_training_stats', 'suggest_lora_config_for_dataset']
