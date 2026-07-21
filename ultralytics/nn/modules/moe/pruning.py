# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Pruning utilities for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import copy
import argparse
from typing import Dict, List, Optional, Tuple, Any
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
from ultralytics.utils import LOGGER


class MoEPruner:
    """Pruner for Mixture-of-Experts models based on usage statistics"""
    
    def __init__(self, model_path: str, threshold: float = 0.15, dataset: str = 'coco8.yaml',
                 device: Optional[str] = None, importance_mode: str = "usage",
                 keep_top_m: Optional[int] = None):
        """
        Initialize MoE pruner

        Args:
            model_path: Path to the model file
            threshold: Minimum usage percentage to keep an expert (0.0-1.0)
            dataset: Dataset configuration for validation
            device: Device for validation. ``None`` (default) auto-detects CUDA,
                falling back to CPU — previously hard-coded to 'cpu', which was
                needlessly slow on GPU boxes.
            importance_mode: Scoring mode for expert importance — ``"usage"``
                (default, pure hit-rate) or ``"usage_weight"`` (hit-rate ×
                mean routing weight, which distinguishes equally-selected
                experts by how strongly the router favours them).
            keep_top_m: If set, always keep the top-M highest-scoring experts
                regardless of threshold.
        """
        self.model_path = model_path
        self.threshold = threshold
        self.dataset = dataset
        self.device = device if device is not None else self._auto_device()
        self.importance_mode = importance_mode
        self.keep_top_m = keep_top_m
        self.model = None
        self.usage_stats: Dict[str, Dict[int, Any]] = {}
        self.pruning_plan: Dict[str, List[int]] = {}

    def _expert_score(self, stats: Any, total_hits: float) -> float:
        """Score a single expert for pruning decisions.

        Args:
            stats: Object with ``hits`` and optionally ``avg_weight`` attributes.
            total_hits: Sum of hits across all experts in the same layer.

        Returns:
            Importance score in ``[0, 1]``.  ``"usage"`` mode returns the raw
            hit-rate; ``"usage_weight"`` mode returns hit-rate × mean weight.
        """
        hits = float(getattr(stats, "hits", 0.0))
        if total_hits <= 0:
            return 0.0
        base = hits / total_hits
        if self.importance_mode == "usage_weight":
            avg_weight = float(getattr(stats, "avg_weight", 0.0))
            return base * avg_weight
        return base

    @staticmethod
    def _auto_device() -> str:
        """Pick CUDA when available, else MPS, else CPU."""
        import torch
        if torch.cuda.is_available():
            return '0'
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
        
    def _load_model(self) -> None:
        """Load YOLO model from file"""
        from ultralytics import YOLO
        
        try:
            self.model = YOLO(self.model_path)
            LOGGER.info(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _diagnose_usage(self) -> None:
        """Run diagnosis to collect expert usage statistics"""
        LOGGER.info("\n[Phase 1] Diagnosing Expert Usage...")
        
        with ExpertUsageTracker(self.model.model) as tracker:
            try:
                self.model.val(
                    data=self.dataset, 
                    split='val', 
                    batch=1, 
                    verbose=False, 
                    device=self.device
                )
                self.usage_stats = tracker.usage_stats
                LOGGER.info(f"✅ Collected usage stats for {len(self.usage_stats)} layers")
            except Exception as e:
                raise RuntimeError(f"Diagnosis failed: {e}")
    
    def _create_pruning_plan(self) -> None:
        """Create pruning plan based on usage statistics"""
        LOGGER.info("\n[Phase 2] Planning Surgery...")
        
        modules_dict = dict(self.model.model.named_modules())
        
        for layer_name, stats in self.usage_stats.items():
            total_hits = sum(s.hits for s in stats.values())
            if total_hits == 0:
                continue
            
            experts_to_keep = []
            LOGGER.info(f"\n   Layer: {layer_name}")

            scored = [
                (self._expert_score(expert_stats, total_hits), int(expert_id), expert_stats)
                for expert_id, expert_stats in stats.items()
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            threshold_kept = {
                expert_id for score, expert_id, _ in scored if score >= float(self.threshold)
            }
            top_kept = {
                expert_id for _, expert_id, _ in scored[: max(int(self.keep_top_m or 0), 0)]
            }
            keep_set = threshold_kept | top_kept
            for _, expert_id, expert_stats in sorted(scored, key=lambda item: item[1]):
                usage_pct = expert_stats.hits / total_hits
                if expert_id in keep_set:
                    experts_to_keep.append(expert_id)
                    LOGGER.info(f"     ✅ Keep E{expert_id} (Usage: {usage_pct:.1%})")
                else:
                    LOGGER.info(f"     🗑️  Drop E{expert_id} (Usage: {usage_pct:.1%})")
            
            # Safety check: ensure at least one expert remains
            if len(experts_to_keep) == 0:
                LOGGER.info(f"     ❌ Error: All experts would be pruned! Keeping top expert.")
                # Keep the expert with highest usage
                top_expert = max(stats.items(), key=lambda x: x[1].hits)[0]
                experts_to_keep = [top_expert]
            
            # Check against original top_k requirement
            if layer_name in modules_dict:
                module = modules_dict[layer_name]
                original_top_k = getattr(module, 'top_k', 2)
                
                if len(experts_to_keep) < original_top_k:
                    LOGGER.info(f"     ⚠️  Warning: Keeping {len(experts_to_keep)} experts, "
                          f"but original top_k={original_top_k}")
            
            self.pruning_plan[layer_name] = sorted(experts_to_keep)
        
        LOGGER.info(f"\n✅ Pruning plan created for {len(self.pruning_plan)} layers")
    
    def _get_parent_module_name(self, layer_name: str) -> str:
        """
        Extract parent module name from layer name
        
        Args:
            layer_name: Full layer name (e.g., 'model.x.routing')
            
        Returns:
            Parent module name (e.g., 'model.x')
        """
        parts = layer_name.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else ""
    
    def _find_projection_layers(
        self, 
        router: nn.Module, 
        num_experts: int
    ) -> List[Tuple[nn.Module, str]]:
        """
        Find the projection layer in router that outputs to experts
        
        Args:
            router: Router module
            num_experts: Original number of experts
            
        Returns every expert-output projection. Gated routers commonly expose
        more than one branch, and all branches must be sliced before the
        expert list is changed.
        """
        found: List[Tuple[nn.Module, str]] = []
        for path, module in router.named_modules():
            if not path:
                continue
            output_size = getattr(module, "out_channels", None)
            if output_size is None:
                output_size = getattr(module, "out_features", None)
            if not isinstance(module, (nn.Conv2d, nn.Linear)) or output_size != num_experts:
                continue
            parent = router
            if "." in path:
                parent_path, child_name = path.rsplit(".", 1)
                parent = router.get_submodule(parent_path)
            else:
                child_name = path
            if isinstance(parent, nn.Sequential) and child_name.isdigit() and int(child_name) != len(parent) - 1:
                continue
            found.append((module, path))
        return found

    def _find_projection_layer(self, router: nn.Module, num_experts: int):
        """Return the first expert-output projection for legacy callers."""
        layers = self._find_projection_layers(router, num_experts)
        return layers[0] if layers else None
    
    def _prune_experts(
        self, 
        moe_module: nn.Module, 
        keep_indices: List[int]
    ) -> None:
        """
        Prune expert modules
        
        Args:
            moe_module: MoE module containing experts
            keep_indices: Indices of experts to keep
        """
        container, attr_name = self._expert_container(moe_module)
        old_experts = getattr(container, attr_name)
        new_experts = nn.ModuleList([old_experts[i] for i in keep_indices])
        setattr(container, attr_name, new_experts)
        moe_module.num_experts = len(keep_indices)
        if hasattr(container, "num_experts"):
            container.num_experts = len(keep_indices)
        loss_fn = getattr(moe_module, "moe_loss_fn", None)
        if loss_fn is not None and hasattr(loss_fn, "num_experts"):
            loss_fn.num_experts = len(keep_indices)
        
        # Adjust top_k if necessary
        if hasattr(moe_module, 'top_k') and moe_module.top_k > moe_module.num_experts:
            old_top_k = moe_module.top_k
            moe_module.top_k = moe_module.num_experts
            LOGGER.info(f"     📉 Reduced top_k from {old_top_k} to {moe_module.top_k}")
        if hasattr(container, "top_k"):
            container.top_k = min(int(container.top_k), len(keep_indices))

    @staticmethod
    def _expert_container(moe_module: nn.Module) -> Tuple[nn.Module, str]:
        """Resolve conventional and fused expert containers."""
        if hasattr(moe_module, "experts"):
            return moe_module, "experts"
        fused = getattr(moe_module, "fused_experts", None)
        if fused is not None and hasattr(fused, "expert_projections"):
            return fused, "expert_projections"
        raise RuntimeError("MoE module has no supported expert container")
    
    def _prune_router_weights(
        self, 
        router: nn.Module, 
        keep_indices: List[int],
        num_old_experts: int
    ) -> bool:
        """
        Prune router projection layer weights
        
        Args:
            router: Router module
            keep_indices: Indices of experts to keep
            num_old_experts: Original number of experts
            
        Returns:
            True if successful, False otherwise
        """
        results = self._find_projection_layers(router, num_old_experts)
        if not results:
            LOGGER.info(f"     ⚠️  Could not locate router projection layer. "
                  f"Skipping weight pruning.")
            return False
        for proj_layer, layer_path in results:
            LOGGER.info(f"     ✂️  Pruning router projection ({layer_path})")
            if isinstance(proj_layer, nn.Conv2d):
                new_proj = nn.Conv2d(
                    in_channels=proj_layer.in_channels,
                    out_channels=len(keep_indices),
                    kernel_size=proj_layer.kernel_size,
                    stride=proj_layer.stride,
                    padding=proj_layer.padding,
                    dilation=proj_layer.dilation,
                    groups=proj_layer.groups,
                    bias=(proj_layer.bias is not None),
                    padding_mode=proj_layer.padding_mode,
                )
            elif isinstance(proj_layer, nn.Linear):
                new_proj = nn.Linear(
                    in_features=proj_layer.in_features,
                    out_features=len(keep_indices),
                    bias=(proj_layer.bias is not None),
                )
            else:
                return False
            with torch.no_grad():
                new_proj.weight.copy_(proj_layer.weight.data[keep_indices])
                if proj_layer.bias is not None:
                    new_proj.bias.copy_(proj_layer.bias.data[keep_indices])
            if "." in layer_path:
                parent_path, child_name = layer_path.rsplit(".", 1)
                parent = router.get_submodule(parent_path)
            else:
                parent = router
                child_name = layer_path
            if child_name.isdigit():
                parent[int(child_name)] = new_proj
            else:
                setattr(parent, child_name, new_proj)

        prior = getattr(router, "expert_prior", None)
        if isinstance(prior, nn.Parameter) and prior.numel() == num_old_experts:
            router.expert_prior = nn.Parameter(prior.data[keep_indices].clone())
        
        # Update router attributes
        router.num_experts = len(keep_indices)
        if hasattr(router, 'top_k'):
            router.top_k = min(router.top_k, router.num_experts)
        
        return True
    
    def _perform_surgery(self) -> nn.Module:
        """
        Perform actual pruning surgery on the model
        
        Returns:
            Pruned model
        """
        LOGGER.info("\n[Phase 3] Performing Surgery...")
        
        new_model = copy.deepcopy(self.model.model)
        modules_dict = dict(new_model.named_modules())
        
        for layer_name, keep_indices in self.pruning_plan.items():
            # Get parent MoE module
            parent_name = self._get_parent_module_name(layer_name)
            if not parent_name:
                LOGGER.info(f"   ❌ Could not determine parent module for {layer_name}")
                continue
            
            if parent_name not in modules_dict:
                LOGGER.info(f"   ❌ Parent module {parent_name} not found")
                continue
            
            moe_module = modules_dict[parent_name]
            
            # Verify MoE structure
            if not hasattr(moe_module, 'routing'):
                LOGGER.info(f"   ❌ {parent_name} missing 'routing' attribute")
                continue
            try:
                expert_container, expert_attr = self._expert_container(moe_module)
            except RuntimeError:
                raise RuntimeError(f"Cannot prune {parent_name}: missing a supported expert container.")
            num_old_experts = len(getattr(expert_container, expert_attr))
            
            # Skip if no pruning needed
            if len(keep_indices) == num_old_experts:
                LOGGER.info(f"   ⏭️  Skipping {layer_name} (no changes needed)")
                continue
            
            LOGGER.info(f"   🔧 Pruning {layer_name}")
            LOGGER.info(f"     Experts: {num_old_experts} → {len(keep_indices)} "
                  f"(keeping {keep_indices})")
            
            # Prune router weights
            router_updated = self._prune_router_weights(
                moe_module.routing, 
                keep_indices, 
                num_old_experts
            )
            if not router_updated:
                raise RuntimeError(
                    f"Cannot prune {parent_name}: router has no expert-output projection that can be reduced safely."
                )
            # Mutate the expert list only after every router branch is ready.
            self._prune_experts(moe_module, keep_indices)
        
        LOGGER.info("\n✅ Surgery completed")
        return new_model
    
    def _save_model(self, pruned_model: nn.Module, output_path: str) -> None:
        """
        Save pruned model to file
        
        Args:
            pruned_model: Pruned model
            output_path: Output file path
        """
        LOGGER.info(f"\n[Phase 4] Saving Pruned Model...")
        
        # Update YOLO wrapper
        self.model.model = pruned_model
        
        # Save checkpoint
        checkpoint = {
            'model': pruned_model,
            'updates': None,
            'pruning_info': {
                'threshold': self.threshold,
                'pruning_plan': self.pruning_plan
            }
        }
        
        torch.save(checkpoint, output_path)
        import json
        from pathlib import Path

        manifest_path = Path(output_path).with_suffix(Path(output_path).suffix + ".prune.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_model": str(self.model_path),
                    "threshold": self.threshold,
                    "importance_mode": self.importance_mode,
                    "keep_top_m": self.keep_top_m,
                    "pruning_plan": self.pruning_plan,
                    "output_model": str(output_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        LOGGER.info(f"✅ Saved to: {output_path}")
    
    def _verify_model(self, output_path: str) -> bool:
        """
        Verify pruned model can be loaded and validated
        
        Args:
            output_path: Path to pruned model
            
        Returns:
            True if verification successful
        """
        LOGGER.info("\n[Phase 5] Verification...")
        
        try:
            from ultralytics import YOLO
            
            # Load check
            pruned_model = YOLO(output_path)
            LOGGER.info("   ✅ Load check: OK")
            
            # Validation check
            LOGGER.info("   🔄 Running validation on pruned model...")
            pruned_model.val(
                data=self.dataset, 
                split='val', 
                batch=1, 
                verbose=False, 
                device=self.device
            )
            LOGGER.info("   ✅ Validation check: OK")
            
            return True
            
        except Exception as e:
            LOGGER.info(f"   ❌ Verification failed: {e}")
            return False
    
    def prune(self, output_path: str) -> bool:
        """
        Execute complete pruning pipeline
        
        Args:
            output_path: Path to save pruned model
            
        Returns:
            True if pruning successful
        """
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"✂️  MoE MODEL PRUNING PIPELINE".center(80))
        LOGGER.info(f"{'='*80}")
        LOGGER.info(f"\n📋 Configuration:")
        LOGGER.info(f"   • Input Model: {self.model_path}")
        LOGGER.info(f"   • Output Model: {output_path}")
        LOGGER.info(f"   • Usage Threshold: {self.threshold*100:.1f}%")
        LOGGER.info(f"   • Dataset: {self.dataset}")
        
        try:
            # Phase 1: Load model
            self._load_model()
            
            # Phase 2: Diagnose usage
            self._diagnose_usage()
            
            # Phase 3: Create pruning plan
            self._create_pruning_plan()
            
            # Phase 4: Perform surgery
            pruned_model = self._perform_surgery()
            
            # Phase 5: Save model
            self._save_model(pruned_model, output_path)
            
            # Phase 6: Verify
            success = self._verify_model(output_path)
            
            if success:
                LOGGER.info(f"\n{'='*80}")
                LOGGER.info(f"🎉 PRUNING COMPLETED SUCCESSFULLY".center(80))
                LOGGER.info(f"{'='*80}\n")
            
            return success
            
        except Exception as e:
            LOGGER.info(f"\n❌ Pruning failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def prune_moe_model(
    model_path: str,
    output_path: str,
    threshold: float = 0.15,
    dataset: str = 'coco8.yaml',
    device: Optional[str] = None,
    importance_mode: str = "usage",
) -> bool:
    """
    Prune MoE model by removing underutilized experts

    Args:
        model_path: Path to input model file
        output_path: Path to save pruned model
        threshold: Minimum usage percentage to keep expert (0.0-1.0)
        dataset: Dataset configuration for validation
        device: Ultralytics device string
        importance_mode: Expert importance scoring mode

    Returns:
        True if pruning successful
    """
    pruner = MoEPruner(model_path, threshold, dataset, device=device, importance_mode=importance_mode)
    return pruner.prune(output_path)


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Prune underutilized experts from MoE YOLO models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path", 
        help="Path to input model file (.pt)"
    )
    parser.add_argument(
        "--output", 
        default="pruned_model.pt", 
        help="Path to save pruned model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Minimum usage percentage to keep expert (0.0-1.0)"
    )
    parser.add_argument(
        "--dataset",
        default="coco8.yaml",
        help="Dataset configuration for validation"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string, e.g. '0', '0,1,2,3', 'cpu'"
    )
    parser.add_argument(
        "--importance-mode",
        choices=("usage", "usage_weight"),
        default="usage",
        help="Expert importance scoring mode"
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")

    success = prune_moe_model(
        args.model_path,
        args.output,
        args.threshold,
        args.dataset,
        device=args.device,
        importance_mode=args.importance_mode,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
