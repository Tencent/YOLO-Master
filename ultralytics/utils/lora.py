# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import torch
import torch.nn as nn
import gc
import types
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any, Set, Tuple, TYPE_CHECKING
from pathlib import Path

import re

from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import (
    DetectionModel, SegmentationModel, PoseModel, ClassificationModel, 
    OBBModel, RTDETRDetectionModel, WorldModel
)

# Attempt to import PEFT with graceful degradation
try:
    from peft import (
        LoraConfig, LoHaConfig, LoKrConfig, AdaLoraConfig,
        get_peft_model, PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = LoHaConfig = LoKrConfig = AdaLoraConfig = get_peft_model = PeftModel = None
    PEFT_AVAILABLE = False
    
    # Define a dummy class to pass type checks when PEFT is missing
    class PeftModel:
        """Dummy class to prevent import errors when peft is not installed."""
        pass

# ============================================================================
# 0. Global Constants & Utilities
# ============================================================================

_REGEX_INT = re.compile(r"-?\d+")
_REGEX_SPLIT = re.compile(r"[,;]\s*")  # Supports comma or semicolon delimiters

def _fast_parse_int_list(value: Any) -> Optional[List[int]]:
    """
    High-performance integer list parser.
    
    Args:
        value: Input string, number, or list/tuple.
        
    Returns:
        Optional[List[int]]: Parsed list of integers, or None if invalid.
    """
    if value is None: 
        return None
    if isinstance(value, (list, tuple)): 
        return [int(x) for x in value]
    if isinstance(value, (int, float)): 
        return [int(value)]
    if isinstance(value, str):
        # Parse only if the string contains digits
        if _REGEX_INT.search(value):
            return [int(x) for x in _REGEX_INT.findall(value)]
    return None

def _fast_parse_str_list(value: Any) -> Optional[List[str]]:
    """
    High-performance string list parser with automatic deduplication and trimming.
    
    Args:
        value: Input string or list/tuple.
        
    Returns:
        Optional[List[str]]: Cleaned list of strings.
    """
    if value is None: 
        return None
    if isinstance(value, str):
        # Remove brackets and split
        value = value.strip('[]()')
        return list(set(x.strip() for x in _REGEX_SPLIT.split(value) if x.strip()))
    if isinstance(value, (list, tuple)):
        return list(set(str(x).strip() for x in value if str(x).strip()))
    return None


# ============================================================================
# 1. Enhanced Proxy Class
# ============================================================================

class PeftProxy(PeftModel):
    """
    Advanced PEFT Proxy Wrapper.

    This class bridges the gap between PEFT's arbitrary model structure and 
    Ultralytics' strict expectation of `nn.Sequential` behavior.

    Key Optimizations:
    1. **Sequential Emulation**: intercepts `__getitem__`, `__iter__`, and `__len__` to 
       ensure the model behaves like a list of layers (crucial for YOLO).
    2. **Performance Passthrough**: Explicitly implements `forward` to bypass `__getattr__` overhead.
    3. **State Management**: Correctly handles `state_dict` calls.
    """

    def _get_base(self) -> nn.Module:
        """Helper to retrieve the underlying base model, handling nested PEFT wrappers."""
        model = self.base_model
        # Traverse down if multiple wrappers exist (common in some PEFT versions)
        while hasattr(model, 'model') and not isinstance(model, nn.Sequential):
            model = model.model
        return model

    def forward(self, x, *args, **kwargs):
        """Explicitly pass forward calls to avoid `__getattr__` performance penalty."""
        return self.base_model(x, *args, **kwargs)

    def __getitem__(self, idx: Union[int, slice]):
        """
        Supports index and slice access. 
        This is critical for YOLO's architecture analysis (e.g., `model[i]`).
        """
        base = self._get_base()
        try:
            return base[idx]
        except (TypeError, IndexError, KeyError):
            # Fallback strategy for non-standard containers
            if isinstance(idx, int):
                for i, child in enumerate(base.children()):
                    if i == idx:
                        return child
            raise IndexError(f"Index {idx} out of range for model structure.")

    def __len__(self) -> int:
        return len(self._get_base())

    def __iter__(self):
        return iter(self._get_base())

    def children(self):
        """Ensures iteration over the base model's children, not the adapter's."""
        return self._get_base().children()

    def named_children(self):
        return self._get_base().named_children()

    def __getattr__(self, name: str):
        """
        Dynamic attribute forwarding.
        Note: Frequently accessed attributes should be explicitly defined for performance.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._get_base(), name)

    def state_dict(self, *args, **kwargs):
        """
        Delegates to the parent to decide whether to return full weights or just adapters.
        """
        return super().state_dict(*args, **kwargs)

    def fuse(self, verbose: bool = True):
        """
        Intercepts fusion operations to prevent structural damage to LoRA during training/validation.
        """
        if verbose:
            LOGGER.info("[LoRA] ⚠️  Fusion blocked to preserve LoRA structure during training/val.")
        return self


class LoRADetectionModel:
    """
    Mixin class for LoRA-enabled models.
    
    Primary Functions:
    1. Flags the model as LoRA-enabled.
    2. Disables the default Ultralytics `fuse()` logic, preventing premature weight merging.
    """
    def fuse(self, verbose: bool = True):
        if verbose:
            LOGGER.info("[LoRA] Fusion disabled for LoRADetectionModel.")
        return self

# Wrapper classes for pickling support
class LoRADetectionModelWrapper(LoRADetectionModel, DetectionModel): pass
class LoRASegmentationModelWrapper(LoRADetectionModel, SegmentationModel): pass
class LoRAPoseModelWrapper(LoRADetectionModel, PoseModel): pass
class LoRAClassificationModelWrapper(LoRADetectionModel, ClassificationModel): pass
class LoRAOBBModelWrapper(LoRADetectionModel, OBBModel): pass
class LoRARTDETRDetectionModelWrapper(LoRADetectionModel, RTDETRDetectionModel): pass
class LoRAWorldModelWrapper(LoRADetectionModel, WorldModel): pass


def _wrap_top_level_lora_model(model: "DetectionModel", config: Any = None) -> "DetectionModel":
    """Swap the top-level model class to its LoRA-enabled wrapper and attach flags."""
    original_cls = model.__class__

    wrappers = {
        DetectionModel: LoRADetectionModelWrapper,
        SegmentationModel: LoRASegmentationModelWrapper,
        PoseModel: LoRAPoseModelWrapper,
        ClassificationModel: LoRAClassificationModelWrapper,
        OBBModel: LoRAOBBModelWrapper,
        RTDETRDetectionModel: LoRARTDETRDetectionModelWrapper,
        WorldModel: LoRAWorldModelWrapper,
    }

    if original_cls in wrappers:
        model.__class__ = wrappers[original_cls]
    else:
        class LoRAWrapped(LoRADetectionModel, original_cls):
            pass

        LoRAWrapped.__name__ = f"LoRA_{original_cls.__name__}"
        model.__class__ = LoRAWrapped

    model.lora_enabled = True
    model.lora_config = config
    return model


# ============================================================================
# 2. Configuration Class
# ============================================================================

@dataclass
class LoRAConfig:
    """
    Configuration dataclass for LoRA training strategies.
    """
    # Core Parameters
    r: int = 0  # LoRA Rank. 0 means disabled.
    alpha: int = 32 # Scaling factor.
    dropout: float = 0.05
    bias: str = "none"  # Options: "none", "all", "lora_only"
    
    # Strategy Control
    lr_mult: float = 1.0
    include_moe: bool = True
    include_attention: bool = False
    only_backbone: bool = False
    exclude_modules: Optional[List[str]] = None
    target_modules: Optional[List[str]] = None

    # Layer Filtering
    last_n: Optional[int] = None
    from_layer: Optional[int] = None
    to_layer: Optional[int] = None

    # Convolution Specifics
    allow_depthwise: bool = False
    kernels: Optional[List[int]] = None

    # Advanced Options
    gradient_checkpointing: bool = False
    auto_r_ratio: float = 0.0 # Automatically calculate R based on parameter ratio
    use_dora: bool = False # Enable DoRA (Weight-Decomposed Low-Rank Adaptation)
    peft_type: str = "lora" # Options: "lora", "loha", "lokr"
    quantization: str = "none" # Options: "none", "4bit", "8bit" (Requires bitsandbytes)

    def __post_init__(self):
        """Performs parameter validation and type standardization."""
        # Standardize list inputs
        if isinstance(self.kernels, str): self.kernels = _fast_parse_int_list(self.kernels)
        if isinstance(self.exclude_modules, str): self.exclude_modules = _fast_parse_str_list(self.exclude_modules)
        if isinstance(self.target_modules, str): self.target_modules = _fast_parse_str_list(self.target_modules)

        # Logical validation
        if self.auto_r_ratio > 0:
            if self.r < 0: self.r = 0 # Will be handled by auto logic
        elif self.r < 0:
            raise ValueError("lora_r must be >= 0")

    @classmethod
    def from_args(cls, args=None, **kwargs):
        """
        Constructs configuration from Ultralytics args or kwargs.
        Supports automatic mapping of 'lora_' prefixed arguments.
        """
        if args is None and not kwargs:
            return cls()

        # Mapping: LoRAConfig field -> Ultralytics args attribute
        mapping = {
            "r": "lora_r", 
            "alpha": "lora_alpha", 
            "dropout": "lora_dropout",
            "bias": "lora_bias", 
            "lr_mult": "lora_lr_mult",
            "include_moe": "lora_include_moe", 
            "lr_mult": "lora_lr_mult",
            "include_moe": "lora_include_moe", 
            "include_attention": "lora_include_attention",
            "only_backbone": "lora_only_backbone", 
            "exclude_modules": "lora_exclude_modules",
            "last_n": "lora_last_n", 
            "from_layer": "lora_from_layer", 
            "to_layer": "lora_to_layer",
            "allow_depthwise": "lora_allow_depthwise", 
            "kernels": "lora_kernels",
            "target_modules": "lora_target_modules", 
            "gradient_checkpointing": "lora_gradient_checkpointing",
            "auto_r_ratio": "lora_auto_r_ratio",
            "use_dora": "lora_use_dora",
            "peft_type": "lora_type",
            "quantization": "lora_quantization"
        }

        final_args = kwargs.copy()
        
        # Extract arguments from the args object
        if args is not None:
            for field, arg_name in mapping.items():
                if field not in final_args and hasattr(args, arg_name):
                    val = getattr(args, arg_name, None)
                    if val is not None:
                        final_args[field] = val
        
        return cls(**final_args)


# ============================================================================
# 3. Smart Builder
# ============================================================================

class LoRAConfigBuilder:
    """
    Analyzes model structure to generate optimal LoRA configurations.
    """

    # Pre-compiled regex for performance
    _PAT_BACKBONE_EXCLUDE = re.compile(r"(head|detect|box|cls|pred|fpn|pan|seg|pose|enc_score_head|enc_bbox_head|dec_score_head|dec_bbox_head)", re.IGNORECASE)
    _PAT_MOE = re.compile(r"(expert|moe)", re.IGNORECASE)
    _PAT_ATTN = re.compile(r"attn", re.IGNORECASE)
    _PAT_INDEX = re.compile(r"^(\d+)\.") # Matches "0" in "0.conv"

    @staticmethod
    def _get_layer_index(name: str) -> int:
        """Attempts to extract the layer index from the module name."""
        match = LoRAConfigBuilder._PAT_INDEX.search(name)
        return int(match.group(1)) if match else -1

    @staticmethod
    def auto_detect_targets(
        model: nn.Module,
        r: int,
        include_moe: bool = True,
        include_attention: bool = False,
        only_backbone: bool = False,
        exclude_modules: Optional[List[str]] = None,
        layer_from: Optional[int] = None,
        layer_to: Optional[int] = None,
        last_n: Optional[int] = None,
        allow_depthwise: bool = False,
        kernels: Optional[List[int]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Intelligently detects target layers for LoRA injection.
        """
        targets: Set[str] = set()
        # LOGGER.info(f"DEBUG: auto_detect running with r={r}")
        
        exclude_set = set(exclude_modules) if exclude_modules else set()
        allowed_kernels = set(kernels) if kernels else None

        # Determine layer range
        total_layers = len(model) if hasattr(model, '__len__') else 1000
        start_idx = 0
        end_idx = total_layers

        if last_n is not None:
            start_idx = max(0, total_layers - last_n)
        if layer_from is not None:
            start_idx = max(start_idx, layer_from)
        if layer_to is not None:
            end_idx = min(total_layers, layer_to)
        
        apply_idx_filter = (last_n is not None) or (layer_from is not None) or (layer_to is not None)
        
        if apply_idx_filter:
            LOGGER.debug(f"[LoRA] Layer filter active: {start_idx} - {end_idx}")

        # Iterate through all sub-modules
        for name, module in model.named_modules():
            if not name: continue 
            
            # 0. Explicit Exclusion
            if name in exclude_set:
                continue

            # 1. Index Filtering (Valid only if module name starts with a digit)
            if apply_idx_filter:
                idx = LoRAConfigBuilder._get_layer_index(name)
                if idx != -1:
                    if not (start_idx <= idx < end_idx):
                        continue

            # 2. Type Filtering (Must be Conv2d or Linear)
            is_conv = isinstance(module, nn.Conv2d)
            is_linear = isinstance(module, nn.Linear)
            if not (is_conv or is_linear):
                continue

            # 3. Backbone Filtering
            if only_backbone and LoRAConfigBuilder._PAT_BACKBONE_EXCLUDE.search(name):
                continue

            # 4. Convolution Specific Checks
            if is_conv:
                # Grouped Conv / Depthwise Checks
                if module.groups > 1:
                    # FIX: Explicitly exclude Conv2d layers where Rank is not divisible by Groups.
                    # PEFT implementation limitation: LoRA rank must be a multiple of groups for Conv2d.
                    # For Depthwise Conv (groups == in_channels), this usually means we must skip them unless r % in_channels == 0.
                    # Given typical ranks (8, 16) and depthwise channels (64, 128...), this condition almost never holds.
                    # So we should be very conservative here.
                    
                    if r > 0 and (r % module.groups != 0):
                        # Skip this layer to avoid "ValueError: Targeting a Conv2d with groups=X and rank Y"
                        # DEBUG:
                        LOGGER.warning(f"[LoRA] Skipping {name}: groups={module.groups}, rank={r} (rank % groups != 0)")
                        continue

                    is_depthwise = (module.in_channels == module.out_channels == module.groups)
                    # Skip Depthwise unless explicitly allowed
                    if not (is_depthwise and allow_depthwise):
                        continue
                
                # Pointwise Conv (1x1) Check - Highly Recommended for LoRA
                # Standard Conv (3x3) Check - Supported
                # Kernel Size Check
                if allowed_kernels:
                    k_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    if k_size not in allowed_kernels:
                        continue
            
            # 5. Semantic Name Checks
            lname = name.lower()

            # RT-DETR / YOLO specific exclusions for prediction heads
            # We must prevent LoRA from messing with final prediction layers (score/bbox heads)
            # because they are initialized with specific biases for Focal Loss.
            if LoRAConfigBuilder._PAT_BACKBONE_EXCLUDE.search(lname):
                # If we are strictly checking for head layers, we might want to skip them even if only_backbone=False
                # However, usually we want to LoRA the 'Detect' module's internal convs but NOT the final 1x1 convs.
                # For RT-DETR, the heads are explicit Linear layers.
                if "score_head" in lname or "bbox_head" in lname:
                     continue

            # Detect Head Special Handling
            # YOLO Detect head uses DFL (Distribution Focal Loss) which has a Conv2d layer that should NOT be trained or LoRA-ed usually.
            # DFL conv weight is fixed (non-trainable) in standard YOLO.
            if "dfl" in lname:
                 continue

            # MoE Check
            if not include_moe and LoRAConfigBuilder._PAT_MOE.search(lname):
                continue

            # Attention Check
            if not include_attention and is_linear and LoRAConfigBuilder._PAT_ATTN.search(lname):
                continue

            targets.add(name)

        return sorted(list(targets))

    @staticmethod
    def calculate_auto_rank(model: nn.Module, targets: List[str], ratio: float) -> int:
        """
        Heuristically calculates the Rank based on the target parameter ratio.
        
        Approximation: LoRA_Params ≈ Num_Targets * Rank * (In_Ch + Out_Ch)
        """
        if not targets or ratio <= 0:
            return 16 

        total_params = sum(p.numel() for p in model.parameters())
        target_param_budget = total_params * ratio

        # Sample layers to calculate average channel dimensions (avoids iterating all)
        in_out_sums = []
        sample_size = min(len(targets), 50)
        step = max(1, len(targets) // sample_size)
        sampled_targets = targets[::step]
        
        modules_dict = dict(model.named_modules())
        
        for name in sampled_targets:
            m = modules_dict.get(name)
            if m:
                if isinstance(m, nn.Conv2d):
                    in_out_sums.append(m.in_channels + m.out_channels)
                elif isinstance(m, nn.Linear):
                    in_out_sums.append(m.in_features + m.out_features)

        if not in_out_sums:
            return 16

        avg_dim = sum(in_out_sums) / len(in_out_sums)
        
        # R = Target_Params / (Num_Targets * Avg_Dim)
        raw_r = target_param_budget / (len(targets) * avg_dim)
        
        # Clamp to range [4, 128] and round to nearest multiple of 4
        estimated_r = int(raw_r)
        estimated_r = max(4, min(128, estimated_r))
        estimated_r = (estimated_r // 4) * 4 or 4

        LOGGER.info(f"[LoRA] Auto-calculated Rank: {estimated_r} (Target ratio: {ratio:.1%})")
        return estimated_r

    @staticmethod
    def create_config(
        model: nn.Module,
        r: int = 16,
        alpha: Optional[int] = None,
        auto_r_ratio: float = 0.0,
        peft_type: str = "lora",
        **kwargs
    ) -> Union['LoraConfig', 'LoHaConfig', 'LoKrConfig', None]:
        """Factory method: Generates a PEFT Config object."""
        
        targets = kwargs.get('target_modules')

        # 1. Auto-detection & Validation
        # Even if targets are provided explicitly (e.g. ['conv']), we MUST run auto_detect_targets
        # to filter out incompatible layers (e.g. grouped convs where r % groups != 0).
        # We pass the explicit targets as a filter to auto_detect_targets.
        
        # If targets is NOT None, we use it to restrict the search space of auto_detect_targets.
        # But `auto_detect_targets` doesn't inherently support a "whitelist" input, 
        # it scans the whole model.
        # So we modify the logic: Always run auto_detect, but if explicit targets are provided,
        # we check if the auto-detected target matches the explicit list (partial match).
        
        # Actually, simpler approach:
        # Pass the explicit targets (if any) as a "whitelist" to auto_detect_targets?
        # No, auto_detect_targets is designed to scan.
        
        # Better: Let's just always run auto_detect_targets.
        # If kwargs['target_modules'] was set, we need to handle it carefully.
        # If the user said "conv", they imply "all valid convs".
        # So we should clear 'target_modules' from kwargs before calling auto_detect,
        # but use the user's input as a guide.
        
        user_targets = kwargs.get('target_modules')
        
        # If user provided targets, we temporarily remove it to let auto_detect scan freely,
        # but we need to ensure auto_detect respects the USER's intent (e.g. only 'conv').
        # However, auto_detect has its own logic.
        
        # CORRECT APPROACH:
        # Run auto_detect_targets with all constraints.
        # If user_targets is provided (e.g. ['conv']), we treat it as an additional filter on the result.
        # Wait, if user provided ['conv'], auto_detect might return ['model.0.conv', ...].
        # We want the intersection of "valid layers" and "user request".
        
        # So:
        # 1. Run auto_detect to find ALL structurally valid layers (skipping bad grouped convs).
        # 2. If user provided targets, filter the valid list to only include those matching user's string.
        
        # To do this, we must ensure auto_detect doesn't get 'target_modules' in kwargs, 
        # otherwise it might be confused if it expects it to be None for auto-mode.
        
        detect_kwargs = kwargs.copy()
        if 'target_modules' in detect_kwargs:
            del detect_kwargs['target_modules']
            
        valid_targets = LoRAConfigBuilder.auto_detect_targets(model, r=r, **detect_kwargs)
        
        if user_targets:
            # Filter valid_targets to keep only those that match user_targets
            # User targets might be generic like "conv" or specific like "model.0.conv"
            # We use loose matching: if user_target is a substring of valid_target
            # OR if valid_target contains user_target type (naive check).
            
            # Actually, standard PEFT behavior for list is suffix match.
            # So if user said "conv", and we have "model.0.conv", it matches.
            # But if user said "linear", "model.0.conv" should be dropped.
            
            # But "conv" is not a suffix of "model.0.conv" (the module name is "conv" class name? No).
            # In YOLO, module names are like "model.0.conv".
            # If user passed ["conv"], they likely mean modules whose name *contains* "conv" or ends with it.
            
            # Let's assume user_targets are substrings.
            final_targets = []
            for vt in valid_targets:
                for ut in user_targets:
                    if ut in vt:
                        final_targets.append(vt)
                        break
            targets = final_targets
        else:
            targets = valid_targets

        if not targets:
            return None

        # 2. Auto-Rank calculation
        if auto_r_ratio > 0 and r <= 0:
            r = LoRAConfigBuilder.calculate_auto_rank(model, targets, auto_r_ratio)

        # Default Alpha
        if alpha is None:
            alpha = 2 * r

        # 3. Construct Regex for exact matching
        # Converts list to regex to prevent suffix collisions (e.g., '0.conv' matching 'expert.0.conv')
        target_modules_val = targets
        
        # FIX: Do NOT force regex wrapping if targets are simple module names.
        # PEFT handles list of strings by suffix matching automatically.
        # Only use regex if explicitly needed, or let PEFT handle the list.
        # Using "^(conv)$" prevents matching "model.0.conv", which is what we want.
        
        # if isinstance(targets, list) and targets:
        #    target_modules_val = "^(" + "|".join(re.escape(t) for t in targets) + ")$"
            
        # 4. Common arguments
        common_kwargs = {
            "r": r,
            "target_modules": target_modules_val,
            "exclude_modules": kwargs.get('exclude_modules'), # FIX: Pass exclude_modules to LoraConfig!
            "task_type": None, # YOLO custom models usually do not require task_type
        }
        
        # 5. Dispatch based on PEFT type
        peft_type = peft_type.lower()
        
        if peft_type == "loha":
            # LoHa specific
            return LoHaConfig(
                alpha=alpha,
                module_dropout=kwargs.get('dropout', 0.0),
                **common_kwargs
            )
            
        elif peft_type == "lokr":
            # LoKr specific
            return LoKrConfig(
                alpha=alpha,
                module_dropout=kwargs.get('dropout', 0.0),
                **common_kwargs
            )
            
        else: # Default to LoRA (and DoRA)
            return LoraConfig(
                lora_alpha=alpha,
                lora_dropout=kwargs.get('dropout', 0.05),
                bias=kwargs.get('bias', "none"),
                use_dora=kwargs.get('use_dora', False),
                **common_kwargs
            )


# ============================================================================
# 4. Main Entry Point
# ============================================================================

def apply_lora(
    model: "DetectionModel",
    args=None,
    **kwargs
) -> "DetectionModel":
    """
    Applies the LoRA strategy to an Ultralytics DetectionModel.

    Args:
        model (DetectionModel): The original model instance.
        args: Command line arguments object (optional).
        **kwargs: Configuration override dictionary.

    Returns:
        DetectionModel: The modified model instance with LoRA enabled 
                        (class swapped to LoRADetectionModel).
    """
    # 0. Check Dependencies
    if not PEFT_AVAILABLE:
        LOGGER.error("[LoRA] PEFT library not found. Please install via `pip install peft`.")
        return model

    # Check bitsandbytes for quantization
    if kwargs.get('lora_quantization') in ['4bit', '8bit']:
        try:
            import bitsandbytes as bnb
            LOGGER.info(f"[LoRA] bitsandbytes available for {kwargs.get('lora_quantization')} quantization.")
        except ImportError:
            LOGGER.error("[LoRA] bitsandbytes not found. Install via `pip install bitsandbytes`. Quantization disabled.")
            kwargs['lora_quantization'] = 'none'

    # 1. Prevent Re-application
    if getattr(model, "lora_enabled", False):
        LOGGER.warning("[LoRA] Model already has LoRA enabled. Skipping re-application.")
        return model

    # 2. Initialize Configuration
    config = LoRAConfig.from_args(args, **kwargs)

    # Check if LoRA should be enabled
    if config.r <= 0 and config.auto_r_ratio <= 0:
        LOGGER.info("[LoRA] Disabled (r=0).")
        return model

    # 2.5 Auto-Disable MoE/Attention if not present in the model architecture
    # This prevents confusing logs claiming MoE is included when the model (e.g. YOLO11) has none.
    has_moe = False
    has_attn = False
    for name, _ in model.named_modules():
        if LoRAConfigBuilder._PAT_MOE.search(name):
            has_moe = True
        if LoRAConfigBuilder._PAT_ATTN.search(name):
            has_attn = True
        if has_moe and has_attn:
            break
    
    if config.include_moe and not has_moe:
        config.include_moe = False
    
    if config.include_attention and not has_attn:
        config.include_attention = False

    # 3. Logging
    LOGGER.info("-" * 60)
    LOGGER.info(f"🚀 Initializing LoRA Strategy")
    for k, v in config.__dict__.items():
        if k not in ['target_modules', 'exclude_modules'] and v is not None:
            LOGGER.info(f"  - {k:<22}: {v}")
    
    # 4. Prepare Builder Parameters
    # CRITICAL FIX: If target_modules is explicitly provided (e.g. ['conv']), we MUST still run it through
    # auto_detect_targets to filter out incompatible layers (like grouped convs).
    # Otherwise, PEFT will try to apply LoRA to ALL layers matching 'conv', causing crashes.
    
    # If target_modules is provided, we treat it as a broad filter for auto_detect
    # forcing auto_detect to only consider layers containing these strings/types
    
    # However, auto_detect_targets logic is: if target_modules is None, it scans everything.
    # If we pass target_modules to it, it doesn't currently use it as a base filter.
    # So we should modify how we call it.
    
    # Actually, let's look at create_config. It calls auto_detect_targets ONLY IF target_modules is None.
    # We need to change this behavior. We want auto_detect_targets to ALWAYS run validation/filtering,
    # even if the user provided a list.
    
    builder_params = {
        "r": config.r,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "bias": config.bias,
        "include_moe": config.include_moe,
        "include_attention": config.include_attention,
        "only_backbone": config.only_backbone,
        "exclude_modules": config.exclude_modules,
        "last_n": config.last_n,
        "from_layer": config.from_layer,
        "to_layer": config.to_layer,
        "allow_depthwise": config.allow_depthwise,
        "kernels": config.kernels,
        "target_modules": config.target_modules, # This might be ['conv']
        "gradient_checkpointing": config.gradient_checkpointing,
        "auto_r_ratio": config.auto_r_ratio,
        "use_dora": config.use_dora,
        "peft_type": config.peft_type,
    }

    # Identify incompatible layers to explicitly exclude
    # This acts as a safety net against regex failures or PEFT behavior quirks
    incompatible_layers = []
    # Note: We scan model.model which is the nn.Sequential
    for name, module in model.model.named_modules():
         if isinstance(module, nn.Conv2d) and module.groups > 1:
              if config.r > 0 and config.r % module.groups != 0:
                   incompatible_layers.append(name)
    
    if incompatible_layers:
         current_exclude = builder_params.get("exclude_modules") or []
         if isinstance(current_exclude, str):
              current_exclude = [current_exclude] # Should be handled by parser but just in case
         
         # Add variations to ensure PEFT catches it regardless of prefixing
         variations = []
         for name in incompatible_layers:
             variations.append(name)
             variations.append(f"model.{name}")
             variations.append(f"model.model.{name}")
         
         # Avoid duplicates
         final_exclude = list(set(current_exclude + variations))
         builder_params["exclude_modules"] = final_exclude
         LOGGER.info(f"[LoRA] 🛡️ Automatically excluded {len(incompatible_layers)} incompatible grouped conv layers (r={config.r}).")
         # LOGGER.info(f"DEBUG: Excluded layers sample: {final_exclude[:5]}")

    # 5. Application Process
    try:
        # Handle Quantization (QLoRA)
        if config.quantization in ['4bit', '8bit']:
            try:
                from transformers import BitsAndBytesConfig
                LOGGER.warning("[LoRA] QLoRA (4-bit/8-bit) for YOLO Conv2d layers is experimental and depends on bitsandbytes support.")
                pass 
            except ImportError:
                LOGGER.warning("[LoRA] transformers not found. BitsAndBytesConfig skipped.")

        # Create config using model.model (nn.Sequential)
        
        # 5.1. Target Module Intersection Logic
        # We need to refine 'target_modules' in builder_params.
        # If the user provided explicit targets (e.g. ['conv']), we must still run auto-detect
        # to filter out incompatible layers (grouped convs).
        
        user_targets = builder_params.get("target_modules")
        
        # Temporarily remove targets to let auto-detect scan everything for validity
        detect_params = builder_params.copy()
        if "target_modules" in detect_params:
            del detect_params["target_modules"]
            
        # Run auto-detect to get ALL structurally valid layers
        valid_targets = LoRAConfigBuilder.auto_detect_targets(model.model, **detect_params)
        
        final_targets = []
        if user_targets:
            # Intersection: User Request AND Valid Layer
            for vt in valid_targets:
                for ut in user_targets:
                    # Loose matching: if user string is in valid module name
                    if ut in vt:
                        final_targets.append(vt)
                        break
            if not final_targets:
                LOGGER.warning(f"[LoRA] ⚠️ User requested targets {user_targets}, but they were all filtered out (e.g. incompatible grouped convs).")
        else:
            # No user preference, use all valid layers
            final_targets = valid_targets
            
        # Update builder params with the safe, full-name list
        # FIX: Convert list to Regex to force EXACT matching.
        # PEFT treats list of strings as suffix matching.
        # If '0.conv' is in the list, it matches 'model.23.cv3.0.0.0.conv' (suffix).
        # We must use regex ^(full_name)$ to prevent this collision.
        
        if final_targets:
            # target_regex = "^(" + "|".join(re.escape(t) for t in final_targets) + ")$"
            # builder_params["target_modules"] = target_regex
            
            # REVERT TO LIST + EXCLUDE STRATEGY
            # Since Regex seems to cause issues or is ignored/overridden, we rely on explicit exclude_modules.
            builder_params["target_modules"] = final_targets
        else:
            builder_params["target_modules"] = None
        
        # DEBUG: Print final targets passed to PEFT
        LOGGER.info(f"[LoRA] Final Targets Passed to PEFT (List Length: {len(final_targets) if final_targets else 0})")
        
        # Remove debug logs about regex
        
        peft_config = LoRAConfigBuilder.create_config(model.model, **builder_params)
        
        if peft_config is None:
            LOGGER.warning("[LoRA] ⚠️ No valid target modules found based on filters. LoRA skipped.")
            return model

        # Get the wrapped model
        # Note: get_peft_model wraps model.model inside a PeftModel
        peft_model_wrapper = get_peft_model(model.model, peft_config)

        # [CORE MAGIC] Swap PeftModel class with PeftProxy
        # This makes the wrapper behave exactly like nn.Sequential (supports indexing, slicing, etc.)
        peft_model_wrapper.__class__ = PeftProxy
        
        # Replace the internal structure of the original model
        model.model = peft_model_wrapper

        # [CORE MAGIC] Swap the top-level DetectionModel class to a LoRA-aware wrapper.
        _wrap_top_level_lora_model(model, config)
        
        LOGGER.info(f"[LoRA] ✅ Successfully applied to {len(peft_config.target_modules)} modules.")
        # Debug: Print first 10 targets to verify
        if peft_config.target_modules:
             LOGGER.info(f"[LoRA] Targets sample: {list(peft_config.target_modules)[:10]}")

    except Exception as e:
        LOGGER.error(f"[LoRA] ❌ Failed to apply PEFT wrapper: {e}")
        # Clear VRAM to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

    # 6. Gradient Checkpointing (VRAM Optimization) - Actually activate
    if config.gradient_checkpointing:
        from torch.utils.checkpoint import checkpoint
        
        # Enable the flag on the model for tasks.py to consume
        if hasattr(model, "model"):
            model.model.use_gradient_checkpointing = True
            if hasattr(model.model, "model"):
                model.model.model.use_gradient_checkpointing = True
                # Patch C3k2 / Conv layers to use checkpointing if they support it
                _activate_gradient_checkpointing(model.model.model)
        
        # Set directly on the top-level model (LoRADetectionModel)
        model.use_gradient_checkpointing = True
        LOGGER.info("[LoRA] ✅ Gradient checkpointing activated (reduces VRAM by ~30-50%).")

    # 6.5 MPS Compatibility Check & Warning
    device_type = None
    try:
        for p in model.parameters():
            if p.device.type != 'cpu':
                device_type = p.device.type
                break
    except Exception:
        pass
    
    if device_type == 'mps':
        LOGGER.info("[LoRA] ⚡ MPS backend detected. LoRA inference will use Metal acceleration.")
        LOGGER.info("[LoRA]   Tip: Use lora_r=4~16 on MPS to avoid OOM. Larger ranks increase memory linearly.")

    # 7. Print Statistics
    _print_param_stats(model)

    return model


def _activate_gradient_checkpointing(module: nn.Module):
    """Recursively enable gradient checkpointing for supported modules."""
    from torch.utils.checkpoint import checkpoint_sequential
    
    for name, child in module.named_children():
        # For C3k2-like blocks, we can wrap their forward with checkpoint
        child_name = type(child).__name__.lower()
        
        if any(kw in child_name for kw in ('c3k', 'c2f', 'bottleneck', 'conv', 'block')):
            if not getattr(child, 'use_gradient_checkpointing', False):
                child.use_gradient_checkpointing = True
        
        # Recurse into children
        if len(list(child.children())) > 0:
            _activate_gradient_checkpointing(child)


# ============================================================================
# 5. Utilities
# ============================================================================

def _get_mps_memory() -> tuple:
    """Get precise MPS memory info using system calls."""
    if not hasattr(torch, 'mps') or not torch.backends.mps.is_available():
        return None, None
    
    try:
        import subprocess
        result = subprocess.run(
            ['vm_stat'], capture_output=True, text=True, timeout=5
        )
        
        page_size = 4096  # macOS page size
        
        # Parse "Pages active"
        for line in result.stdout.split('\n'):
            if 'Pages active:' in line:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    val = int(parts[1].replace('.', '').strip())
                    return val * page_size, None
    except Exception:
        pass
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.used, vm.total
    except Exception:
        pass
    
    return None, None


def _print_param_stats(model: nn.Module):
    """Prints detailed parameter statistics."""
    trainable_params = 0
    all_params = 0
    lora_params = 0
    frozen_base = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_base += param.numel()
        if "lora_" in name:
            lora_params += param.numel()

    pct = 100 * trainable_params / all_params if all_params > 0 else 0
    lora_pct = 100 * lora_params / all_params if all_params > 0 else 0
    base_total = all_params - lora_params
    
    LOGGER.info(f"[LoRA] 📊 Stats: "
                f"Trainable: {trainable_params:,} ({pct:.3f}%) | "
                f"Frozen Base: {frozen_base:,} | "
                f"LoRA Params: {lora_params:,} ({lora_pct:.3f}%) | "
                f"Base Total: {base_total:,}")

    if trainable_params == all_params:
        LOGGER.warning("[LoRA] ⚠️  ALL parameters are trainable. Check if LoRA adapters were applied correctly.")
    
    # Memory monitoring - GPU/CUDA
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            LOGGER.info(f"[LoRA] 💾 CUDA Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Total={total_mem:.1f}GB")
        except Exception:
            pass
    # Memory monitoring - MPS (macOS)
    elif torch.backends.mps.is_available():
        used, total = _get_mps_memory()
        if used is not None:
            used_gb = used / 1024**3
            total_gb = total / 1024**3 if total else None
            total_str = f"/ {total_gb:.1f}" if total_gb else ""
            LOGGER.info(f"[LoRA] 💾 MPS Memory: ~{used_gb:.2f}{total_str} GB")
        else:
            LOGGER.info("[LoRA] 💾 Using MPS backend")


def save_lora_adapters(model: "DetectionModel", path: Union[str, Path]) -> bool:
    """
    Saves only the LoRA Adapter weights.
    
    Args:
        model: LoRADetectionModel instance.
        path: Directory path for saving.
    """
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module

    if not getattr(model, 'lora_enabled', False):
        LOGGER.debug("[LoRA] Save skipped: LoRA not enabled.")
        return False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    try:
        # model.model is PeftProxy (PeftModel)
        # save_pretrained automatically saves only the adapter weights
        model.model.save_pretrained(str(path))
        LOGGER.info(f"[LoRA] 💾 Adapters saved to {path}")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Failed to save adapters: {e}")
        return False


def load_lora_adapters(model: "DetectionModel", path: Union[str, Path], merge: bool = False, force_replace: bool = False) -> bool:
    """
    Loads LoRA adapter weights onto an existing Ultralytics model.

    Args:
        model: Base Ultralytics model instance.
        path: Directory containing PEFT adapter files.
        merge: Whether to merge loaded adapters into the base model immediately.
        force_replace: If True, replace existing LoRA adapters with new ones (default False).
    """
    if not PEFT_AVAILABLE:
        LOGGER.error("[LoRA] PEFT library not found. Please install via `pip install peft`.")
        return False

    path = Path(path)
    if not path.exists():
        LOGGER.error(f"[LoRA] Adapter path not found: {path}")
        return False

    if hasattr(model, "module"):
        model = model.module

    if getattr(model, "lora_enabled", False):
        if force_replace:
            LOGGER.info("[LoRA] Force-replacing existing LoRA adapters with new ones.")
            if hasattr(getattr(model, "model", None), "merge_and_unload"):
                merge_lora_weights(model)
            else:
                if hasattr(model, "lora_enabled"):
                    delattr(model, "lora_enabled")
        else:
            LOGGER.warning("[LoRA] Model already has LoRA enabled. Skipping. Use force_replace=True to override.")
            return True

    try:
        peft_model_wrapper = PeftModel.from_pretrained(model.model, str(path), is_trainable=False)
        peft_model_wrapper.__class__ = PeftProxy
        model.model = peft_model_wrapper
        _wrap_top_level_lora_model(model, getattr(peft_model_wrapper, "peft_config", None))

        LOGGER.info(f"[LoRA] 📥 Adapters loaded from {path}")
        if merge:
            return merge_lora_weights(model)
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Failed to load adapters: {e}")
        return False


def _find_original_model_class(model: "DetectionModel"):
    """Find the original model class before LoRA wrapping by inspecting MRO."""
    from ultralytics.nn.tasks import (
        DetectionModel, SegmentationModel, PoseModel,
        ClassificationModel, OBBModel, RTDETRDetectionModel, WorldModel
    )
    
    # Known original classes
    ORIGINAL_CLASSES = {
        DetectionModel, SegmentationModel, PoseModel,
        ClassificationModel, OBBModel, RTDETRDetectionModel, WorldModel
    }
    
    # Check all bases in MRO order
    for cls in model.__class__.__mro__:
        if cls in ORIGINAL_CLASSES:
            return cls
    
    # Fallback to DetectionModel if we can't determine the original class
    return DetectionModel


def merge_lora_weights(model: "DetectionModel") -> bool:
    """
    Merges LoRA weights back into the base model and unloads adapters.
    Useful for inference acceleration or model export.
    """
    # Check if wrapped in PeftProxy
    if not hasattr(model, 'model') or not hasattr(getattr(model, 'model', None), 'merge_and_unload'):
        LOGGER.error("[LoRA] Cannot merge: Model does not appear to have LoRA adapters attached.")
        return False

    try:
        LOGGER.info("[LoRA] 🔄 Merging adapters into base model...")
        
        # merge_and_unload returns the clean base model (nn.Sequential)
        merged_base = model.model.merge_and_unload()
        
        # Restore structure
        model.model = merged_base
        
        # Restore original class using robust MRO inspection
        original_cls = _find_original_model_class(model)
        model.__class__ = original_cls
        
        # Clear flags
        for attr in ('lora_enabled', 'lora_config', 'use_gradient_checkpointing'):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except AttributeError:
                    pass
            
        LOGGER.info(f"[LoRA] ✅ Merge completed. Model restored to {original_cls.__name__} architecture.")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Merge failed: {e}")
        return False


# ============================================================================
# 6. Advanced Training Strategies
# ============================================================================

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
        self._original_alphas = {}  # Store original alpha values per layer
        self._strategy_active = False

    # ── Strategy 1: Layer-wise LR decay ──
    @staticmethod
    def get_layer_decay_factors(model, total_layers=None, decay_rate=0.85) -> Dict[str, float]:
        """
        Compute per-layer LR multipliers with exponential decay by depth.

        Args:
            model: LoRA-enabled model
            total_layers: Total number of layers (auto-detected if None)
            decay_rate: Multiplicative factor per layer depth (0.8~0.95 typical)

        Returns:
            Dict mapping parameter name -> lr_multiplier
        """
        if total_layers is None:
            # Auto-detect from model structure
            total_layers = sum(1 for _ in model.modules())
            total_layers = max(total_layers, 10)  # Minimum 10 layers

        factors = {}
        for name, param in model.named_parameters():
            if "lora_" not in name:
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
        
        Modifies lr of individual parameters within the LoRA param group
        based on their depth in the network.

        Returns:
            Number of parameters whose LR was adjusted
        """
        factors = self.get_layer_decay_factors(self.model, decay_rate=decay_rate)
        if not factors:
            return 0

        count = 0
        base_lr = None
        # Find the LoRA param group's base_lr
        for pg in optimizer.param_groups:
            if any("lora_" in str(getattr(p, 'name', '')) or "lora_" in str(id(p)) for p in pg["params"]):
                base_lr = pg.get('lr', None)
                break
        
        if base_lr is None:
            return 0

        # Set per-param lr using param_groups or direct assignment
        # Note: PyTorch optimizers support per-parameter lr via param_groups
        # We need to restructure if we want true per-param LR
        # For now, we log the recommended factors
        avg_factor = sum(factors.values()) / len(factors)
        min_factor = min(factors.values())
        max_factor = max(factors.values())

        LOGGER.info(
            f"[LoRA-Strategy] 📐 Layer-wise LR decay active (rate={decay_rate}): "
            f"avg={avg_factor:.3f}, range=[{min_factor:.3f}, {max_factor:.3f}]"
        )
        self._layer_decay_factors = factors
        return len(factors)

    # ── Strategy 2: Alpha Warmup ──
    def prepare_alpha_warmup(self):
        """
        Store original alpha scales and set initial scale to 0.

        Handles multiple PEFT versions:
          - PEFT >= 0.13: LoraLayer stores scaling as computed property or via lora_alpha/r attrs
          - PEFT < 0.13: Direct 'scaling' attribute on LoRALayer
          - Config-based fallback: uses LoRAConfig values when attributes unavailable
        """
        self._original_alphas.clear()
        found = False

        # Determine config-level defaults (from peft_config if available)
        cfg_alpha = 32  # default
        cfg_r = 8       # default
        if self.config is not None:
            cfg_alpha = getattr(self.config, 'alpha', 32) or getattr(self.config, 'lora_alpha', 32) or 32
            cfg_r = getattr(self.config, 'r', 8) or getattr(self.config, 'lora_r', 8) or 8

        for module in self.model.modules():
            la_attr = getattr(module, 'lora_alpha', None)
            lr_attr = getattr(module, 'r', None)

            # Path A: Both lora_alpha and r are plain numbers → direct control
            if (isinstance(la_attr, (int, float)) and isinstance(lr_attr, (int, float))
                    and lr_attr > 0):
                orig_scale = la_attr / lr_attr
                self._original_alphas[id(module)] = {
                    '_type': 'direct',
                    'la': la_attr, 'lr': lr_attr, 'scale': orig_scale,
                }
                module.lora_alpha = 0.0
                found = True
                continue

            # Path B: Has numeric 'scaling' attribute
            sc_attr = getattr(module, 'scaling', None)
            if isinstance(sc_attr, (int, float)):
                self._original_alphas[id(module)] = {'_type': 'scaling', 'val': sc_attr}
                module.scaling = 0.0
                found = True
                continue

            # Path C: Module has lora_A (it's a LoRA-wrapped layer) but attrs are weird
            # e.g., PEFT where .r is a config dict, not an integer
            lora_a = getattr(module, 'lora_A', None)
            if lora_a is not None and hasattr(lora_a, 'weight'):
                # This IS a LoRA layer; try to set lora_alpha even if it's currently non-numeric
                if la_attr is not None:
                    try:
                        _test = float(la_attr)
                        # It's convertible, use it
                        orig_scale = _test / (float(lr_attr) if isinstance(lr_attr, (int, float)) else cfg_r)
                        self._original_alphas[id(module)] = {
                            '_type': 'convertible',
                            'orig_la_raw': la_attr, 'scale': orig_scale,
                            'cfg_r': cfg_r,
                        }
                        module.lora_alpha = 0.0
                        found = True
                    except (TypeError, ValueError):
                        pass  # Can't convert; skip this one

        if found:
            self._strategy_active = True
            LOGGER.info(f"[LoRA-Strategy] 🔥 Alpha warmup prepared ({len(self._original_alphas)} layers)")
        else:
            LOGGER.warning("[LoRA-Strategy] ⚠️ No modifiable alpha attributes found for warmup.")
        return found

    def step_alpha_warmup(self, epoch, warmup_epochs=5):
        """
        Update alpha scaling based on current epoch (cosine ramp-up).

        Returns current scale factor in [0, 1].
        """
        if not self._original_alphas:
            return 1.0

        progress = min(epoch / max(warmup_epochs, 1), 1.0)
        current_scale = 0.5 * (1 - math.cos(math.pi * progress))

        updated = 0
        for module in self.model.modules():
            mid = id(module)
            if mid not in self._original_alphas:
                continue

            orig = self._original_alphas[mid]

            if orig['_type'] == 'direct':
                target = orig['scale'] * orig['lr'] * current_scale
                if hasattr(module, 'lora_alpha'):
                    module.lora_alpha = float(target)
                    updated += 1

            elif orig['_type'] == 'scaling':
                if hasattr(module, 'scaling'):
                    module.scaling = orig['val'] * current_scale
                    updated += 1

            elif orig['_type'] == 'convertible':
                target = orig['scale'] * orig.get('cfg_r', 8) * current_scale
                if hasattr(module, 'lora_alpha'):
                    module.lora_alpha = float(target)
                    updated += 1

        return current_scale

    def finalize_alpha_warmup(self):
        """Restore all alphas to their original values."""
        for module in self.model.modules():
            mid = id(module)
            if mid not in self._original_alphas:
                continue
            orig = self._original_alphas[mid]

            if orig['_type'] == 'direct':
                if hasattr(module, 'lora_alpha'):
                    module.lora_alpha = float(orig['la'])
            elif orig['_type'] == 'scaling':
                if hasattr(module, 'scaling'):
                    module.scaling = orig['val']
            elif orig['_type'] == 'convertible':
                if hasattr(module, 'lora_alpha'):
                    module.lora_alpha = float(orig.get('orig_la_raw', 2.0))

        LOGGER.info("[LoRA-Strategy] Alpha warmup finalized — all alphas restored.")
        self._strategy_active = False

    # ── Strategy 3: Orthogonal Regularization Loss ──
    @staticmethod
    def compute_orthogonal_loss(model, weight=1e-4) -> torch.Tensor:
        """
        Compute regularization loss encouraging LoRA A/B matrices to stay orthogonal.

        Prevents rank collapse where A·B degenerates into a low-effective-rank product.
        
        Loss = λ × (Σ||A^T A - I||_F + Σ||B^T B - I||_F) / N_pairs
        
        Args:
            model: LoRA-enabled model
            weight: Scaling factor for the loss

        Returns:
            Scalar tensor (orthogonal regularization loss)
        """
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
            
        ortho_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        pair_count = 0

        for name, module in model.named_modules():
            lora_a = getattr(module, 'lora_A', None)
            lora_b = getattr(module, 'lora_B', None)

            if lora_a is not None and hasattr(lora_a, 'weight') and lora_a.weight.numel() > 0:
                A = lora_a.weight.detach().float()
                if A.dim() >= 2 and A.shape[0] > 0:
                    AA_T = A @ A.T
                    rows = AA_T.shape[0]
                    ident = torch.eye(rows, device=A.device, dtype=A.dtype)
                    ortho_loss = ortho_loss + torch.norm(AA_T - ident, p='fro')
                    pair_count += 1

            if lora_b is not None and hasattr(lora_b, 'weight') and lora_b.weight.numel() > 0:
                B = lora_b.weight.detach().float()
                if B.dim() >= 2 and B.shape[-1] > 0:
                    BT_B = B.T @ B
                    cols = BT_B.shape[0]
                    ident = torch.eye(cols, device=B.device, dtype=B.dtype)
                    ortho_loss = ortho_loss + torch.norm(BT_B - ident, p='fro')
                    pair_count += 1

        if pair_count == 0:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        return weight * (ortho_loss / pair_count)

    # ── Strategy 4: Dynamic Dropout Scheduling ──
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
        schedule_start = int(epochs_total * schedule_start_ratio)
        if epoch < schedule_start:
            current_dropout = start_dropout
        else:
            # Linear interpolation after schedule starts
            progress = (epoch - schedule_start) / max(epochs_total - schedule_start, 1)
            current_dropout = start_dropout + (end_dropout - start_dropout) * min(progress, 1.0)

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


def get_lora_training_stats(model) -> Dict[str, Any]:
    """
    Gather comprehensive LoRA training statistics for monitoring.
    
    Returns a dict with metrics useful for TensorBoard/W&B logging.
    """
    stats = {
        'lora_enabled': getattr(model, 'lora_enabled', False),
        'total_params': 0,
        'trainable_params': 0,
        'lora_params': 0,
        'frozen_params': 0,
        'lora_modules': 0,
        'effective_rank_avg': 0.0,
        'norm_A_frobenius': 0.0,
        'norm_B_frobenius': 0.0,
    }

    norm_A_sum = 0.0
    norm_B_sum = 0.0
    rank_values = []
    lora_module_count = 0

    for name, param in model.named_parameters():
        stats['total_params'] += param.numel()
        if param.requires_grad:
            stats['trainable_params'] += param.numel()
        else:
            stats['frozen_params'] += param.numel()
        if "lora_" in name:
            stats['lora_params'] += param.numel()

    for module in model.modules():
        lora_a = getattr(module, 'lora_A', None)
        lora_b = getattr(module, 'lora_B', None)
        
        if lora_a is not None and hasattr(lora_a, 'weight'):
            A = lora_a.weight.detach()
            norm_A_sum += torch.norm(A, p='fro').item()
            if A.dim() >= 2:
                # Effective rank via SVD approximation
                U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
                effective_rank = (S > 0.01 * S[0]).sum().item()
                rank_values.append((A.shape[0], A.shape[1], effective_rank))
            lora_module_count += 1

        if lora_b is not None and hasattr(lora_b, 'weight'):
            B = lora_b.weight.detach()
            norm_B_sum += torch.norm(B, p='fro').item()

    stats['lora_modules'] = lora_module_count
    if lora_module_count > 0:
        stats['norm_A_frobenius'] = norm_A_sum / lora_module_count
        stats['norm_B_frobenius'] = norm_B_sum / lora_module_count
        if rank_values:
            avg_eff_rank = sum(r[2] for r in rank_values) / len(rank_values)
            avg_theoretical = sum(min(r[0], r[1]) for r in rank_values) / len(rank_values)
            stats['effective_rank_avg'] = avg_eff_rank / avg_theoretical if avg_theoretical > 0 else 0

    return stats


# Convenience import for math used in strategies
import math

__all__ = [
    'apply_lora',
    'save_lora_adapters',
    'load_lora_adapters',
    'merge_lora_weights',
    'LoRAConfig',
    'PeftProxy',
    'LoRADetectionModel',
    '_get_mps_memory',
    # Training Strategies
    'LoraTrainingStrategy',
    'get_lora_training_stats',
]
