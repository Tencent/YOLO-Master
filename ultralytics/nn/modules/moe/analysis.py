# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Analysis utilities for Mixture-of-Experts models"""
import torch
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass


@dataclass
class ExpertStats:
    """Statistics data structure for individual experts"""
    hits: float = 0.0
    weighted_sum: float = 0.0

    @property
    def avg_weight(self) -> float:
        """Calculate average weight per hit"""
        return self.weighted_sum / self.hits if self.hits > 0 else 0.0


class ExpertUsageTracker:
    """Tracker for Mixture-of-Experts (MoE) expert usage patterns"""

    # Module types to skip during hook registration
    SKIP_TYPES = (
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.SiLU,
        torch.nn.Sequential,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.Linear,
        torch.nn.GroupNorm,
        torch.nn.Softmax
    )

    # Keywords to identify router modules
    ROUTER_KEYWORDS = ('routing', 'router')

    def __init__(self, model: torch.nn.Module):
        """
        Initialize the expert usage tracker

        Args:
            model: PyTorch model containing MoE layers
        """
        self.model = model
        self.usage_stats: Dict[str, Dict[int, ExpertStats]] = defaultdict(
            lambda: defaultdict(ExpertStats)
        )
        self.total_tokens = 0
        self.hooks = []
        self._register_hooks()

    def _process_dense_weights(self, name: str, weights: torch.Tensor) -> None:
        """
        Process dense weight tensors with shape [B, E, H, W] or [B, E]

        Args:
            name: Layer name
            weights: Weight tensor from router output
        """
        # Normalize to [N, E] format
        if weights.dim() == 4:
            flat_weights = weights.permute(0, 2, 3, 1).reshape(-1, weights.shape[1])
        elif weights.dim() == 2:
            flat_weights = weights
        else:
            return

        num_tokens, num_experts = flat_weights.shape
        self.total_tokens += num_tokens

        # Vectorized computation of expert statistics
        weights_np = flat_weights.numpy()
        hits_mask = weights_np > 1e-6

        hits_per_expert = hits_mask.sum(axis=0)
        weight_per_expert = weights_np.sum(axis=0)

        # Batch update statistics
        for expert_id in range(num_experts):
            if hits_per_expert[expert_id] > 0:
                stats = self.usage_stats[name][expert_id]
                stats.hits += float(hits_per_expert[expert_id])
                stats.weighted_sum += float(weight_per_expert[expert_id])

    def _process_sparse_topk(
            self,
            name: str,
            topk_vals: torch.Tensor,
            topk_indices: torch.Tensor
    ) -> None:
        """
        Process sparse Top-K router outputs

        Args:
            name: Layer name
            topk_vals: Top-K weight values
            topk_indices: Top-K expert indices
        """
        # Normalize to [N, K] format
        if topk_indices.dim() == 4:
            flat_indices = topk_indices.permute(0, 2, 3, 1).reshape(-1, topk_indices.shape[1])
            flat_vals = topk_vals.permute(0, 2, 3, 1).reshape(-1, topk_vals.shape[1])
        else:
            flat_indices = topk_indices
            flat_vals = topk_vals

        num_tokens = flat_indices.shape[0]
        self.total_tokens += num_tokens

        # Efficient statistics using numpy's bincount
        idx_flat = flat_indices.numpy().flatten().astype(np.int32)
        val_flat = flat_vals.numpy().flatten().astype(np.float32)

        if idx_flat.size == 0:
            return

        max_expert_id = int(idx_flat.max())

        # Compute hits and weights for all experts at once
        hits_counts = np.bincount(idx_flat, minlength=max_expert_id + 1)
        weight_sums = np.bincount(idx_flat, weights=val_flat, minlength=max_expert_id + 1)

        # Batch update
        for expert_id in np.nonzero(hits_counts)[0]:
            stats = self.usage_stats[name][expert_id]
            stats.hits += float(hits_counts[expert_id])
            stats.weighted_sum += float(weight_sums[expert_id])

    def _create_router_hook(self, name: str):
        """
        Create forward hook function for router module

        Args:
            name: Module name

        Returns:
            Hook function
        """

        def hook(module, input, output):
            with torch.no_grad():  # Disable gradient computation
                # Handle dense tensor output
                if isinstance(output, torch.Tensor):
                    weights = output.detach().cpu()
                    self._process_dense_weights(name, weights)

                # Handle sparse Top-K output (values, indices)
                elif isinstance(output, tuple) and len(output) >= 2:
                    topk_vals = output[0].detach().cpu()
                    topk_indices = output[1].detach().cpu()
                    self._process_sparse_topk(name, topk_vals, topk_indices)

        return hook

    def _is_router_module(self, name: str, module: torch.nn.Module) -> bool:
        """
        Determine if a module is a router

        Args:
            name: Module name
            module: PyTorch module

        Returns:
            True if module is a router
        """
        # Check if name contains router keywords
        has_keyword = any(keyword in name.lower() for keyword in self.ROUTER_KEYWORDS)

        # Exclude basic layer types
        is_skip_type = isinstance(module, self.SKIP_TYPES)

        return has_keyword and not is_skip_type

    def _register_hooks(self) -> None:
        """Register forward hooks on all router modules"""
        print(f"{'Module Name':<50} | {'Type':<30} | {'Status'}")
        print("-" * 90)

        hooked_count = 0
        for name, module in self.model.named_modules():
            if self._is_router_module(name, module):
                hook = module.register_forward_hook(self._create_router_hook(name))
                self.hooks.append(hook)
                print(f"{name:<50} | {type(module).__name__:<30} | ✅ Hooked")
                hooked_count += 1

        if hooked_count == 0:
            print("⚠️  WARNING: No router modules found! Check naming conventions.")
        else:
            print(f"\n✅ Successfully hooked {hooked_count} router module(s)")
        print("-" * 90)

    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _calculate_status(self, share_pct: float, ideal_share: float) -> str:
        """
        Calculate expert health status based on usage

        Args:
            share_pct: Actual usage percentage
            ideal_share: Ideal balanced usage percentage

        Returns:
            Status string with emoji indicator
        """
        if share_pct < ideal_share * 0.1:
            return "💀 DEAD"
        elif share_pct < ideal_share * 0.5:
            return "⚠️  LOW"
        elif share_pct > ideal_share * 2.0:
            return "🔥 HOT"
        return "✅ OK"

    def print_report(self) -> None:
        """Print comprehensive diagnostic report"""
        print("\n" + "=" * 80)
        print(" 🔍 EXPERT USAGE DIAGNOSIS REPORT ".center(80))
        print("=" * 80)

        if not self.usage_stats:
            print("\n⚠️  No usage data collected. Did the model run inference?")
            return

        print(f"\n📊 Total Tokens Processed: {self.total_tokens:,}\n")

        # Prepare data for visualization
        layers = []
        all_experts: Set[int] = set()
        data_matrix = []

        for layer_name in sorted(self.usage_stats.keys()):
            expert_stats = self.usage_stats[layer_name]

            if not expert_stats:
                continue

            layers.append(layer_name)

            print(f"\n{'─' * 80}")
            print(f"📍 Layer: {layer_name}")
            print(f"{'─' * 80}")

            # Calculate total hits and ideal distribution
            total_hits = sum(stats.hits for stats in expert_stats.values())
            num_experts = len(expert_stats)
            ideal_share = 100.0 / num_experts if num_experts > 0 else 0

            # Table header
            print(f"{'ID':<6} | {'Usage %':<10} | {'Avg Weight':<12} | {'Hits':<12} | {'Status':<10}")
            print(f"{'-' * 6}|{'-' * 12}|{'-' * 14}|{'-' * 14}|{'-' * 10}")

            # Collect layer data for plotting
            layer_data = {}

            # Output sorted by expert ID
            for expert_id in sorted(expert_stats.keys()):
                stats = expert_stats[expert_id]
                share_pct = (stats.hits / total_hits * 100) if total_hits > 0 else 0
                status = self._calculate_status(share_pct, ideal_share)

                print(f"{expert_id:<6} | {share_pct:>9.2f}% | {stats.avg_weight:>11.4f} | "
                      f"{int(stats.hits):>11,} | {status}")

                layer_data[expert_id] = share_pct
                all_experts.add(expert_id)

            data_matrix.append(layer_data)

            # Statistical summary
            print(f"\n📈 Summary:")
            print(f"   • Total Experts: {num_experts}")
            print(f"   • Ideal Share: {ideal_share:.2f}%")
            print(f"   • Total Hits: {int(total_hits):,}")

            # Calculate load balance metric (standard deviation)
            shares = [stats.hits / total_hits * 100 for stats in expert_stats.values()]
            std_dev = np.std(shares) if shares else 0
            print(f"   • Load Balance (StdDev): {std_dev:.2f}%")

        print("\n" + "=" * 80 + "\n")

        # Generate visualizations
        self._plot_visualizations(layers, all_experts, data_matrix)

    def _plot_visualizations(
            self,
            layers: List[str],
            all_experts: Set[int],
            data_matrix: List[Dict[int, float]]
    ) -> None:
        """
        Generate and save visualization plots

        Args:
            layers: List of layer names
            all_experts: Set of all expert IDs
            data_matrix: Usage data per layer
        """
        if not layers:
            return

        max_expert_id = max(all_experts) if all_experts else 0
        num_experts = max_expert_id + 1

        # Build matrix: [Num_Layers, Num_Experts]
        matrix = np.zeros((len(layers), num_experts))

        for i, layer_data in enumerate(data_matrix):
            for eid, pct in layer_data.items():
                matrix[i, eid] = pct

        # 1. Heatmap
        try:
            plt.figure(figsize=(12, max(4, len(layers) * 0.8 + 2)))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                xticklabels=[f"E{i}" for i in range(num_experts)],
                yticklabels=layers,
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Selection %'}
            )
            plt.title("Expert Usage Heatmap (Selection %)", fontsize=14, fontweight='bold')
            plt.xlabel("Expert ID", fontsize=12)
            plt.ylabel("MoE Layer", fontsize=12)
            plt.tight_layout()
            save_path = "expert_usage_heatmap.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"✅ Heatmap saved to: {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"❌ Heatmap generation failed: {e}")

        # 2. Bar Chart (Aggregated across layers)
        try:
            plt.figure(figsize=(12, 6))
            avg_usage = np.mean(matrix, axis=0)
            bars = plt.bar(
                range(num_experts),
                avg_usage,
                color='skyblue',
                edgecolor='navy',
                alpha=0.7
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            # Add ideal balance reference line
            ideal_line = 100 / num_experts
            plt.axhline(
                y=ideal_line,
                color='r',
                linestyle='--',
                linewidth=2,
                label=f'Ideal Balance ({ideal_line:.1f}%)'
            )

            plt.xlabel("Expert ID", fontsize=12)
            plt.ylabel("Avg Selection %", fontsize=12)
            plt.title(
                "Global Expert Usage Distribution (Averaged across layers)",
                fontsize=14,
                fontweight='bold'
            )
            plt.legend(fontsize=10)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_path_bar = "expert_usage_bar.png"
            plt.savefig(save_path_bar, dpi=150)
            plt.close()
            print(f"✅ Bar chart saved to: {os.path.abspath(save_path_bar)}")
        except Exception as e:
            print(f"❌ Bar chart generation failed: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks"""
        self.remove_hooks()


def diagnose_model(
        model_path: str,
        dataset: str = 'coco8.yaml',
        batch_size: int = 1,
        verbose: bool = False
) -> None:
    """
    Diagnose MoE model expert usage patterns

    Args:
        model_path: Path to model file (.pt)
        dataset: Dataset configuration file
        batch_size: Batch size for validation
        verbose: Enable verbose output during validation
    """
    # Local import to avoid circular dependency
    from ultralytics import YOLO

    print(f"\n🚀 Starting Model Diagnosis")
    print(f"📁 Model: {model_path}")
    print(f"📊 Dataset: {dataset}")

    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Use context manager for automatic hook cleanup
    with ExpertUsageTracker(model.model) as tracker:
        print(f"\n🔄 Running validation (batch_size={batch_size})...")
        try:
            model.val(
                data=dataset,
                split='val',
                batch=batch_size,
                verbose=verbose,
                device='cpu'
            )
            print("✅ Validation completed")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return

        tracker.print_report()


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Analyze MoE Expert Usage in YOLO Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path",
        nargs='?',
        default="/Users/gatilin/Downloads/master-v0.0-yolomoe-v1-small.pt",
        help="Path to .pt model file"
    )
    parser.add_argument(
        "--dataset",
        default="coco8.yaml",
        help="Dataset configuration file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for validation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output during validation"
    )

    args = parser.parse_args()
    diagnose_model(args.model_path, args.dataset, args.batch_size, args.verbose)


if __name__ == "__main__":
    main()