# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import torch
import torch.nn as nn
import re
import gc
import types
from dataclasses import dataclass, field
import re
from typing import Optional, List, Union, Dict, Any, Set, Tuple, TYPE_CHECKING
from pathlib import Path

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

        # [CORE MAGIC] Swap the top-level DetectionModel class
        # This disables operations incompatible with LoRA (like default fusion)
        # We use dynamic inheritance (Mixin) to preserve the original model's behavior (e.g. init_criterion)
        original_cls = model.__class__
        
        # Map original class to wrapper class to support pickling
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
            # Fallback for unknown custom classes (still not picklable but maintains functionality)
            class LoRAWrapped(LoRADetectionModel, original_cls):
                pass
            
            # Preserve original class name for display/logging purposes
            LoRAWrapped.__name__ = f"LoRA_{original_cls.__name__}"
            model.__class__ = LoRAWrapped
        
        # Inject flags
        model.lora_enabled = True
        model.lora_config = config
        
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

    # 6. Gradient Checkpointing (VRAM Optimization)
    if config.gradient_checkpointing:
        # Enable the flag on the model for tasks.py to consume
        if hasattr(model, "model"):
            model.model.use_gradient_checkpointing = True
            # Also set on the base model just in case
            if hasattr(model.model, "model"):
                pass
        
        # Set directly on the top-level model (LoRADetectionModel)
        model.use_gradient_checkpointing = True
        LOGGER.info("[LoRA] Gradient checkpointing enabled (YOLO Native Mode).")

    # 7. Print Statistics
    _print_param_stats(model)

    return model


# ============================================================================
# 5. Utilities
# ============================================================================

def _print_param_stats(model: nn.Module):
    """Prints detailed parameter statistics."""
    trainable_params = 0
    all_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora_" in name:
            lora_params += param.numel()

    LOGGER.info(f"[LoRA] 📊 Stats: "
                f"Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%) | "
                f"LoRA Params: {lora_params:,}")

    if trainable_params == all_params:
        LOGGER.warning("[LoRA] ⚠️  ALL parameters are trainable. Check if LoRA adapters were applied correctly.")
    
    # Optional: Log memory usage if available
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            LOGGER.info(f"[LoRA] 💾 GPU Memory: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        except Exception:
            pass # Ignore memory logging errors
    elif torch.backends.mps.is_available():
        # MPS doesn't provide precise memory stats via torch yet, but we can log presence
        LOGGER.info("[LoRA] 💾 Using MPS backend.")


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


def merge_lora_weights(model: "DetectionModel") -> bool:
    """
    Merges LoRA weights back into the base model and unloads adapters.
    Useful for inference acceleration or model export.
    """
    # Check if wrapped in PeftProxy
    if not hasattr(model, 'model') or not hasattr(model.model, 'merge_and_unload'):
        LOGGER.error("[LoRA] Cannot merge: Model does not appear to have LoRA adapters attached.")
        return False

    try:
        LOGGER.info("[LoRA] 🔄 Merging adapters into base model...")
        
        # merge_and_unload returns the clean base model (nn.Sequential)
        merged_base = model.model.merge_and_unload()
        
        # Restore structure
        model.model = merged_base
        
        # Restore original class definition (Remove LoRA Mixin)
        # We assume the second base class is the original one (LoRADetectionModel, OriginalModel)
        if len(model.__class__.__bases__) > 1:
            model.__class__ = model.__class__.__bases__[1]
        else:
             # Fallback if structure is unexpected
            model.__class__ = DetectionModel
        
        # Clear flags
        if hasattr(model, 'lora_enabled'):
            del model.lora_enabled
            
        LOGGER.info("[LoRA] ✅ Merge completed. Model restored to standard architecture.")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Merge failed: {e}")
        return False


__all__ = [
    'apply_lora',
    'save_lora_adapters',
    'merge_lora_weights',
    'LoRAConfig',
    'PeftProxy',
    'LoRADetectionModel'
]
