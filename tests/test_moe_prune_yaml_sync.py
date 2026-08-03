# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression tests: MoEPruner must keep ``model.yaml`` consistent with the pruned
ES_MOE blocks so a pruned model survives a YAML-based rebuild during retraining
(the prune -> LoRA / full fine-tune recovery workflow).

Two failure modes are covered:
1. Expert *count* re-inflation - rebuild uses ES_MOE's default expert count and
   ``intersect_dicts`` drops the reduced expert/router weights.
2. Expert *kernel* mismatch - pruning keeps experts with heterogeneous kernels
   (e.g. [5, 9]) but a bare rebuild assigns the defaults [3, 5], so the kept
   experts' depthwise weights are dropped on a shape mismatch and re-initialized.

The fix writes the full ES_MOE arg list (count + per-expert kernel sizes) into
``model.yaml`` so both survive the rebuild.
"""
import yaml

from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.pruning import MoEPruner
from ultralytics.nn.tasks import DetectionModel

CFG = "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml"


def _experts(model):
    return [m.num_experts for _, m in model.named_modules() if isinstance(m, ES_MOE)]


def _kernels(model):
    return [
        [e.conv.depthwise.kernel_size[0] for e in m.experts]
        for _, m in model.named_modules()
        if isinstance(m, ES_MOE)
    ]


def _build_reduced(num_experts):
    """Build a model whose ES_MOE blocks each have ``num_experts`` (stand-in for a pruned model)."""
    d = yaml.safe_load(open(CFG))
    for layer in d["backbone"]:
        if layer[2] == "ES_MOE":
            layer[3] = [layer[3][0], num_experts]
    return DetectionModel(d, ch=3, nc=10, verbose=False)


def _build_with_kernels(num_experts, kernels):
    """Stand-in for a pruned model that kept experts with non-default kernel sizes."""
    d = yaml.safe_load(open(CFG))
    for layer in d["backbone"]:
        if layer[2] == "ES_MOE":
            out_ch = layer[3][0]
            layer[3] = [out_ch, num_experts, 8, num_experts, True, 0.4, 15, list(kernels)]
    return DetectionModel(d, ch=3, nc=10, verbose=False)


def test_reinflation_without_sync():
    """Reproduce the bug: reduced model + un-synced (full) yaml re-inflates on rebuild."""
    reduced = _build_reduced(2)
    assert _experts(reduced) == [2, 2, 2, 2]
    reduced.yaml = yaml.safe_load(open(CFG))  # bug state: yaml still describes default experts
    rebuilt = DetectionModel(reduced.yaml, ch=3, nc=10, verbose=False)
    assert _experts(rebuilt) != [2, 2, 2, 2], "expected re-inflation when yaml is not synced"


def test_sync_preserves_pruned_experts():
    """The fix writes the reduced count into yaml so a rebuild preserves it."""
    reduced = _build_reduced(2)
    reduced.yaml = yaml.safe_load(open(CFG))  # start from the un-synced (buggy) yaml
    MoEPruner._sync_yaml_num_experts(type("D", (), {})(), reduced)  # method only uses self for logging
    es_args = [layer[3] for layer in reduced.yaml["backbone"] if layer[2] == "ES_MOE"]
    assert all(a[1] == 2 for a in es_args), f"num_experts not synced: {es_args}"
    rebuilt = DetectionModel(reduced.yaml, ch=3, nc=10, verbose=False)
    assert _experts(rebuilt) == [2, 2, 2, 2], "pruned expert count must survive the rebuild"


def test_sync_preserves_expert_kernels():
    """The fix also writes per-expert kernel sizes so heterogeneous kept experts survive."""
    reduced = _build_with_kernels(2, [5, 9])
    before = _kernels(reduced)
    assert all(k == [5, 9] for k in before), f"stand-in kernels not built: {before}"
    reduced.yaml = yaml.safe_load(open(CFG))  # un-synced yaml would rebuild default [3, 5]
    MoEPruner._sync_yaml_num_experts(type("D", (), {})(), reduced)
    es_args = [layer[3] for layer in reduced.yaml["backbone"] if layer[2] == "ES_MOE"]
    assert all(a[-1] == [5, 9] for a in es_args), f"kernels not synced: {es_args}"
    rebuilt = DetectionModel(reduced.yaml, ch=3, nc=10, verbose=False)
    assert _kernels(rebuilt) == before, "expert kernel sizes must survive the rebuild"
