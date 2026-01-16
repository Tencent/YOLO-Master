# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Pruning utilities for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import copy
import argparse
from typing import Dict, List, Optional, Tuple, Any
from .analysis import ExpertUsageTracker


class MoEPruner:
    """Pruner for Mixture-of-Experts models based on usage statistics"""
    
    def __init__(self, model_path: str, threshold: float = 0.15, dataset: str = 'coco8.yaml'):
        """
        Initialize MoE pruner
        
        Args:
            model_path: Path to the model file
            threshold: Minimum usage percentage to keep an expert (0.0-1.0)
            dataset: Dataset configuration for validation
        """
        self.model_path = model_path
        self.threshold = threshold
        self.dataset = dataset
        self.model = None
        self.usage_stats: Dict[str, Dict[int, Any]] = {}
        self.pruning_plan: Dict[str, List[int]] = {}
        
    def _load_model(self) -> None:
        """Load YOLO model from file"""
        from ultralytics import YOLO
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _diagnose_usage(self) -> None:
        """Run diagnosis to collect expert usage statistics"""
        print("\n[Phase 1] Diagnosing Expert Usage...")
        
        with ExpertUsageTracker(self.model.model) as tracker:
            try:
                self.model.val(
                    data=self.dataset, 
                    split='val', 
                    batch=1, 
                    verbose=False, 
                    device='cpu'
                )
                self.usage_stats = tracker.usage_stats
                print(f"✅ Collected usage stats for {len(self.usage_stats)} layers")
            except Exception as e:
                raise RuntimeError(f"Diagnosis failed: {e}")
    
    def _create_pruning_plan(self) -> None:
        """Create pruning plan based on usage statistics"""
        print("\n[Phase 2] Planning Surgery...")
        
        modules_dict = dict(self.model.model.named_modules())
        
        for layer_name, stats in self.usage_stats.items():
            total_hits = sum(s.hits for s in stats.values())
            if total_hits == 0:
                continue
            
            experts_to_keep = []
            print(f"\n   Layer: {layer_name}")
            
            # Determine which experts to keep based on threshold
            for expert_id, expert_stats in sorted(stats.items()):
                usage_pct = expert_stats.hits / total_hits
                if usage_pct >= self.threshold:
                    experts_to_keep.append(expert_id)
                    print(f"     ✅ Keep E{expert_id} (Usage: {usage_pct:.1%})")
                else:
                    print(f"     🗑️  Drop E{expert_id} (Usage: {usage_pct:.1%})")
            
            # Safety check: ensure at least one expert remains
            if len(experts_to_keep) == 0:
                print(f"     ❌ Error: All experts would be pruned! Keeping top expert.")
                # Keep the expert with highest usage
                top_expert = max(stats.items(), key=lambda x: x[1].hits)[0]
                experts_to_keep = [top_expert]
            
            # Check against original top_k requirement
            if layer_name in modules_dict:
                module = modules_dict[layer_name]
                original_top_k = getattr(module, 'top_k', 2)
                
                if len(experts_to_keep) < original_top_k:
                    print(f"     ⚠️  Warning: Keeping {len(experts_to_keep)} experts, "
                          f"but original top_k={original_top_k}")
            
            self.pruning_plan[layer_name] = sorted(experts_to_keep)
        
        print(f"\n✅ Pruning plan created for {len(self.pruning_plan)} layers")
    
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
    
    def _find_projection_layer(
        self, 
        router: nn.Module, 
        num_experts: int
    ) -> Optional[Tuple[nn.Module, str]]:
        """
        Find the projection layer in router that outputs to experts
        
        Args:
            router: Router module
            num_experts: Original number of experts
            
        Returns:
            Tuple of (projection_layer, layer_path) or None if not found
        """
        # Check common router structures
        candidates = [
            ('router', 'router'),
            ('routing_network', 'routing_network'),
        ]
        
        for attr_name, path_name in candidates:
            if hasattr(router, attr_name):
                sequential = getattr(router, attr_name)
                if isinstance(sequential, nn.Sequential) and len(sequential) > 0:
                    last_layer = sequential[-1]
                    
                    # Check if it's the projection layer
                    if isinstance(last_layer, nn.Conv2d):
                        if last_layer.out_channels == num_experts:
                            return last_layer, f"{path_name}[-1]"
                    elif isinstance(last_layer, nn.Linear):
                        if last_layer.out_features == num_experts:
                            return last_layer, f"{path_name}[-1]"
        
        return None
    
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
        old_experts = moe_module.experts
        new_experts = nn.ModuleList([old_experts[i] for i in keep_indices])
        
        moe_module.experts = new_experts
        moe_module.num_experts = len(keep_indices)
        
        # Adjust top_k if necessary
        if hasattr(moe_module, 'top_k') and moe_module.top_k > moe_module.num_experts:
            old_top_k = moe_module.top_k
            moe_module.top_k = moe_module.num_experts
            print(f"     📉 Reduced top_k from {old_top_k} to {moe_module.top_k}")
    
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
        result = self._find_projection_layer(router, num_old_experts)
        
        if result is None:
            print(f"     ⚠️  Could not locate router projection layer. "
                  f"Skipping weight pruning.")
            return False
        
        proj_layer, layer_path = result
        print(f"     ✂️  Pruning router projection ({layer_path})")
        
        # Create new projection layer with reduced output dimension
        if isinstance(proj_layer, nn.Conv2d):
            new_proj = nn.Conv2d(
                in_channels=proj_layer.in_channels,
                out_channels=len(keep_indices),
                kernel_size=proj_layer.kernel_size,
                stride=proj_layer.stride,
                padding=proj_layer.padding,
                bias=(proj_layer.bias is not None)
            )
        elif isinstance(proj_layer, nn.Linear):
            new_proj = nn.Linear(
                in_features=proj_layer.in_features,
                out_features=len(keep_indices),
                bias=(proj_layer.bias is not None)
            )
        else:
            return False
        
        # Copy weights for kept experts
        with torch.no_grad():
            new_proj.weight.data = proj_layer.weight.data[keep_indices].clone()
            if proj_layer.bias is not None:
                new_proj.bias.data = proj_layer.bias.data[keep_indices].clone()
        
        # Replace the layer in the sequential container
        if 'routing_network' in layer_path:
            router.routing_network[-1] = new_proj
        elif 'router' in layer_path:
            router.router[-1] = new_proj
        
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
        print("\n[Phase 3] Performing Surgery...")
        
        new_model = copy.deepcopy(self.model.model)
        modules_dict = dict(new_model.named_modules())
        
        for layer_name, keep_indices in self.pruning_plan.items():
            # Get parent MoE module
            parent_name = self._get_parent_module_name(layer_name)
            if not parent_name:
                print(f"   ❌ Could not determine parent module for {layer_name}")
                continue
            
            if parent_name not in modules_dict:
                print(f"   ❌ Parent module {parent_name} not found")
                continue
            
            moe_module = modules_dict[parent_name]
            
            # Verify MoE structure
            if not hasattr(moe_module, 'experts') or not hasattr(moe_module, 'routing'):
                print(f"   ❌ {parent_name} missing 'experts' or 'routing' attributes")
                continue
            
            num_old_experts = len(moe_module.experts)
            
            # Skip if no pruning needed
            if len(keep_indices) == num_old_experts:
                print(f"   ⏭️  Skipping {layer_name} (no changes needed)")
                continue
            
            print(f"   🔧 Pruning {layer_name}")
            print(f"     Experts: {num_old_experts} → {len(keep_indices)} "
                  f"(keeping {keep_indices})")
            
            # Prune experts
            self._prune_experts(moe_module, keep_indices)
            
            # Prune router weights
            self._prune_router_weights(
                moe_module.routing, 
                keep_indices, 
                num_old_experts
            )
        
        print("\n✅ Surgery completed")
        return new_model
    
    def _save_model(self, pruned_model: nn.Module, output_path: str) -> None:
        """
        Save pruned model to file
        
        Args:
            pruned_model: Pruned model
            output_path: Output file path
        """
        print(f"\n[Phase 4] Saving Pruned Model...")
        
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
        print(f"✅ Saved to: {output_path}")
    
    def _verify_model(self, output_path: str) -> bool:
        """
        Verify pruned model can be loaded and validated
        
        Args:
            output_path: Path to pruned model
            
        Returns:
            True if verification successful
        """
        print("\n[Phase 5] Verification...")
        
        try:
            from ultralytics import YOLO
            
            # Load check
            pruned_model = YOLO(output_path)
            print("   ✅ Load check: OK")
            
            # Validation check
            print("   🔄 Running validation on pruned model...")
            pruned_model.val(
                data=self.dataset, 
                split='val', 
                batch=1, 
                verbose=False, 
                device='cpu'
            )
            print("   ✅ Validation check: OK")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")
            return False
    
    def prune(self, output_path: str) -> bool:
        """
        Execute complete pruning pipeline
        
        Args:
            output_path: Path to save pruned model
            
        Returns:
            True if pruning successful
        """
        print(f"\n{'='*80}")
        print(f"✂️  MoE MODEL PRUNING PIPELINE".center(80))
        print(f"{'='*80}")
        print(f"\n📋 Configuration:")
        print(f"   • Input Model: {self.model_path}")
        print(f"   • Output Model: {output_path}")
        print(f"   • Usage Threshold: {self.threshold*100:.1f}%")
        print(f"   • Dataset: {self.dataset}")
        
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
                print(f"\n{'='*80}")
                print(f"🎉 PRUNING COMPLETED SUCCESSFULLY".center(80))
                print(f"{'='*80}\n")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Pruning failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def prune_moe_model(
    model_path: str, 
    output_path: str, 
    threshold: float = 0.15, 
    dataset: str = 'coco8.yaml'
) -> bool:
    """
    Prune MoE model by removing underutilized experts
    
    Args:
        model_path: Path to input model file
        output_path: Path to save pruned model
        threshold: Minimum usage percentage to keep expert (0.0-1.0)
        dataset: Dataset configuration for validation
        
    Returns:
        True if pruning successful
    """
    pruner = MoEPruner(model_path, threshold, dataset)
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
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")
    
    success = prune_moe_model(
        args.model_path, 
        args.output, 
        args.threshold,
        args.dataset
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()