"""Executable P0 gates for distributed, adapter, and export lifecycles."""

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.modules.mot import C2fMoT
from ultralytics.nn.modules.multitask.head import MultiTaskHead
from ultralytics.utils.export_capabilities import classify_routed_module, load_export_capability_matrix
from ultralytics.utils.export_preflight import export_preflight
from ultralytics.utils.lora import load_adapters, save_adapters
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.fallback import ManualLoRAConv
from ultralytics.utils.loss import MultiTaskLoss
from ultralytics.utils.dist import ddp_launch_env, ddp_launch_prefix, find_free_network_port
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.engine.extensions.recovery import TrainingRecoveryController
from ultralytics.engine.trainer import BaseTrainer


ROOT = Path(__file__).resolve().parents[1]


def _base_model():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1))


class _TinyMultiTaskMoT(nn.Module):
    """Small real routed graph used for optimizer/checkpoint lifecycle gates."""

    def __init__(self):
        super().__init__()
        self.mot = C2fMoT(
            16,
            16,
            n=1,
            num_heads=2,
            top_k=3,
            window_size=3,
            n_points=2,
            sparse_train=True,
            sparse_train_warmup_steps=1,
            local_attn_window=3,
        )
        self.head = MultiTaskHead(
            nc=3,
            ch=(16, 16, 16),
            tasks=["detect", "segment", "pose"],
            nm=4,
            npr=8,
            kpt_shape=(2, 3),
            reg_max=1,
            end2end=True,
        )
        self.head.stride = torch.tensor([8.0, 16.0, 32.0])
        self.model = nn.Sequential(self.mot, self.head)
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, overlap_mask=True, pose=12.0, kobj=1.0)
        self.end2end = True
        self.task_weights = {"detect": 1.0, "segment": 0.5, "pose": 1.0}

    def forward(self, image):
        feature = self.mot(image)
        return self.head([feature, feature, feature])


def _tiny_multitask_batch():
    return {
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.99, 0.99], [0.5, 0.5, 0.99, 0.99]]),
        "masks": torch.randint(0, 2, (2, 64, 64), dtype=torch.uint8),
        "sem_masks": torch.zeros(2, 64, 64, dtype=torch.long),
        "keypoints": torch.tensor([[[0.5, 0.5, 2.0], [0.6, 0.6, 2.0]], [[0.4, 0.4, 2.0], [0.3, 0.3, 2.0]]]),
    }


def test_cpu_gloo_two_rank_routed_continuous_training():
    command = [
        *ddp_launch_prefix(),
        "--master_addr=127.0.0.1",
        f"--master_port={find_free_network_port()}",
        "--nproc_per_node=2",
        str(ROOT / "tests/ddp_moe_smoke.py"),
    ]
    env = {**ddp_launch_env(), "OMP_NUM_THREADS": "1"}
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "P0 routed DDP gate passed" in completed.stdout


def test_planner_adapter_full_lifecycle(tmp_path):
    torch.manual_seed(11)
    model = apply_lora(
        _base_model(),
        LoRAConfig(r=2, alpha=4, backend="fallback", planner_backend="vpeft", adapter_budget=100_000),
    )
    assert model.lora_placement_plan["status"] == "ACCEPT"
    assert model.lora_target_modules
    assert all(isinstance(model.get_submodule(name), ManualLoRAConv) for name in model.lora_target_modules)

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.SGD(trainable.values(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(2, 3, 8, 8)).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
    assert sum(float(parameter.grad.abs().sum()) for parameter in trainable.values()) > 0.0
    optimizer.step()
    assert any(not torch.equal(before[name], parameter) for name, parameter in trainable.items())

    path = tmp_path / "adapter"
    assert save_adapters(model, path)
    restored = apply_lora(
        _base_model(),
        LoRAConfig(r=2, alpha=4, backend="fallback", planner_backend="vpeft", adapter_budget=100_000),
    )
    assert load_adapters(restored, path)
    source_state = {name: value for name, value in model.state_dict().items() if "lora_" in name}
    restored_state = {name: value for name, value in restored.state_dict().items() if "lora_" in name}
    assert source_state.keys() == restored_state.keys()
    assert all(torch.equal(source_state[name], restored_state[name]) for name in source_state)


def test_routed_export_manifest_and_dense_fallback_are_executable():
    matrix = load_export_capability_matrix()
    modules = [
        OptimizedMOE(8, 8, num_experts=2, top_k=1),
        MoABlock(8, num_heads=3),
        MoTBlock(8, num_heads=2, top_k=1),
    ]
    for module in modules:
        family = classify_routed_module(module)
        assert family in matrix["modules"]
        runtime = getattr(module, "export_capabilities", lambda: {})()
        assert runtime.get("supported", True) is True
        assert isinstance(runtime.get("export_safe_dense_fallback", True), bool)
        report = export_preflight(module, "onnx", strict=False, matrix=matrix)
        decision = report["decisions"][0]
        assert decision["module_family"] == family
        fallback = runtime.get("export_safe_dense_fallback", decision["dense_fallback"])
        assert decision["dense_fallback"] is fallback
        assert report["supported"] is fallback


def test_multitask_mot_optimizer_step_updates_all_supervised_branches():
    torch.manual_seed(17)
    model = _TinyMultiTaskMoT().train()
    criterion = MultiTaskLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.ema = ModelEMA(model)
    trainer._gradient_nonfinite = False
    trainer.optimizer_steps = 0
    trainer.args = SimpleNamespace(lora_few_shot_mode=False)
    trainer.recovery_controller = TrainingRecoveryController(trainer)

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss, items = criterion(model(torch.randn(2, 16, 8, 8)), _tiny_multitask_batch())
    assert torch.isfinite(loss) and loss.item() > 0
    assert torch.isfinite(items).all()
    loss.backward()

    for module in (
        model.head.cv3,
        model.head.cv4_seg,
        model.head.cv4_pose,
        model.mot.m[0].router,
        *model.mot.m[0].experts,
    ):
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0

    assert trainer.optimizer_step() is True
    assert trainer.optimizer_steps == 1
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    assert int(model.mot.m[0]._sparse_train_step.item()) > 0


def test_multitask_checkpoint_resume_preserves_optimizer_and_mot_runtime_state(tmp_path):
    torch.manual_seed(19)
    model = _TinyMultiTaskMoT().train()
    criterion = MultiTaskLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.ema = ModelEMA(model)
    trainer.optimizer_steps = 0
    trainer.start_epoch = 0
    trainer.epoch = 0
    trainer.best_fitness = 0.0
    trainer.fitness = 0.0
    trainer.metrics = {}
    trainer.args = SimpleNamespace(model="tiny.yaml")
    trainer.read_results_csv = lambda: {}
    trainer.adapter_controller = None
    trainer.recovery_controller = TrainingRecoveryController(trainer)

    loss, _ = criterion(model(torch.randn(2, 16, 8, 8)), _tiny_multitask_batch())
    loss.backward()
    assert BaseTrainer.optimizer_step(trainer) is True
    model.mot.m[0].router.temperature.fill_(0.63)
    serialized = trainer.recovery_controller.serialize_checkpoint(include_online_model=True)
    checkpoint = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_steps"] == 1

    restored = _TinyMultiTaskMoT().train()
    restored.load_state_dict(checkpoint["model"].float().state_dict())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_trainer = object.__new__(BaseTrainer)
    restored_trainer.model = restored
    restored_trainer.optimizer = restored_optimizer
    restored_trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_trainer.ema = ModelEMA(restored)
    restored_trainer.optimizer_steps = 0
    restored_trainer.best_fitness = None
    restored_trainer._load_checkpoint_state(checkpoint)

    assert restored_trainer.optimizer_steps == 1
    assert torch.equal(restored.mot.m[0]._sparse_train_step, model.mot.m[0]._sparse_train_step)
    assert torch.equal(restored.mot.m[0].router.temperature, model.mot.m[0].router.temperature)
    assert restored_optimizer.state_dict()["state"]
