"""CPU-only tests for optimizer parameter-group auditing."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.optim import OptimizerGroupAuditError, audit_optimizer_param_groups


class _AuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3))
        self.bias = nn.Parameter(torch.ones(2))
        self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)


class _FakeOptimizer:
    def __init__(self, param_groups, defaults=None):
        self.param_groups = param_groups
        self.defaults = defaults or {}


def _optimizer(*groups, defaults=None):
    return _FakeOptimizer(list(groups), defaults=defaults)


def test_all_trainable_parameters_are_covered_exactly_once():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight, model.bias], "group_name": "trainable"})

    audit = audit_optimizer_param_groups(model, optimizer)

    assert audit["trainable_coverage_complete"]
    assert audit["trainable_coverage_exactly_once"]
    assert not audit["has_missing_trainable"]
    assert not audit["has_duplicates"]
    assert audit["trainable_parameter_count"] == 2
    assert audit["trainable_element_count"] == 8


def test_missing_trainable_is_reported_without_strict_mode():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight], "group_name": "partial"})

    audit = audit_optimizer_param_groups(model, optimizer, strict=False)

    assert audit["has_missing_trainable"]
    assert not audit["trainable_coverage_complete"]
    assert [item["name"] for item in audit["missing_trainable"]] == ["bias"]


def test_missing_trainable_raises_in_strict_mode():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight], "group_name": "partial"})

    with pytest.raises(OptimizerGroupAuditError, match=r"missing_trainable=1.*bias"):
        audit_optimizer_param_groups(model, optimizer, strict=True)


def test_duplicate_across_groups_is_reported_and_strict():
    model = _AuditModel()
    optimizer = _optimizer(
        {"params": [model.weight, model.bias], "group_name": "first"},
        {"params": [model.weight], "group_name": "second"},
    )

    audit = audit_optimizer_param_groups(model, optimizer)

    assert audit["has_duplicates"]
    assert not audit["trainable_coverage_exactly_once"]
    assert audit["duplicated"][0]["name"] == "weight"
    assert [group["name"] for group in audit["duplicated"][0]["groups"]] == ["first", "second"]
    with pytest.raises(OptimizerGroupAuditError, match=r"duplicated=1.*first,second"):
        audit_optimizer_param_groups(model, optimizer, strict=True)


def test_duplicate_within_one_group_is_reported():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight, model.weight, model.bias], "group_name": "repeated"})

    audit = audit_optimizer_param_groups(model, optimizer)

    assert audit["duplicated_count"] == 1
    assert audit["optimizer_parameter_occurrence_count"] == 3
    assert [group["name"] for group in audit["duplicated"][0]["groups"]] == ["repeated", "repeated"]


def test_frozen_parameter_is_reported_but_nonfatal_in_strict_mode():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight, model.bias, model.frozen], "group_name": "mixed"})

    audit = audit_optimizer_param_groups(model, optimizer, strict=True)

    assert audit["has_frozen_in_optimizer"]
    assert audit["frozen_in_optimizer_count"] == 1
    assert audit["frozen_in_optimizer"][0]["name"] == "frozen"
    assert audit["trainable_coverage_exactly_once"]


def test_unknown_optimizer_parameter_is_reported_and_strict():
    model = _AuditModel()
    external = nn.Parameter(torch.ones(5))
    optimizer = _optimizer({"params": [model.weight, model.bias, external], "group_name": "external"})

    audit = audit_optimizer_param_groups(model, optimizer)

    assert audit["has_unknown_parameters"]
    assert audit["unknown_optimizer_parameter_count"] == 1
    assert audit["unknown_optimizer_parameters"][0]["numel"] == 5
    with pytest.raises(OptimizerGroupAuditError, match=r"unknown_optimizer_parameters=1.*external"):
        audit_optimizer_param_groups(model, optimizer, strict=True)


def test_group_tensor_and_element_counts_are_accurate():
    model = _AuditModel()
    optimizer = _optimizer(
        {"params": [model.weight, model.frozen], "group_name": "mixed"},
        {"params": [model.bias], "group_name": "bias"},
    )

    audit = audit_optimizer_param_groups(model, optimizer)

    mixed, bias = audit["groups"]
    assert (mixed["tensor_count"], mixed["total_element_count"]) == (2, 10)
    assert (mixed["trainable_element_count"], mixed["frozen_element_count"]) == (6, 4)
    assert (bias["tensor_count"], bias["total_element_count"]) == (1, 2)
    assert audit["optimizer_unique_parameter_count"] == 3
    assert audit["optimizer_unique_element_count"] == 12


def test_group_hyperparameters_and_lr_scale_are_reported():
    model = _AuditModel()
    optimizer = _optimizer(
        {
            "params": [model.weight, model.bias],
            "group_name": "explicit",
            "lr": 0.02,
            "initial_lr": 0.025,
            "weight_decay": 0.001,
        },
        defaults={"lr": 0.01},
    )

    group = audit_optimizer_param_groups(model, optimizer)["groups"][0]

    assert group["name"] == "explicit"
    assert group["lr"] == pytest.approx(0.02)
    assert group["initial_lr"] == pytest.approx(0.025)
    assert group["weight_decay"] == pytest.approx(0.001)
    assert group["base_lr"] == pytest.approx(0.01)
    assert group["lr_scale"] == pytest.approx(2.0)
    assert group["lr_scale_source"] == "lr/base_lr"


def test_explicit_lr_scale_takes_precedence_over_inferred_scale():
    model = _AuditModel()
    optimizer = _optimizer(
        {"params": [model.weight, model.bias], "lr": 0.02, "lr_scale": 3.0},
        defaults={"lr": 0.01},
    )

    group = audit_optimizer_param_groups(model, optimizer)["groups"][0]

    assert group["lr_scale"] == pytest.approx(3.0)
    assert group["lr_scale_source"] == "group.lr_scale"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"group_name": "named"}, "named"),
        ({"name": "optimizer-name"}, "optimizer-name"),
        ({"role": "role-name"}, "role-name"),
        ({"param_group": "legacy-explicit"}, "legacy-explicit"),
    ],
)
def test_explicit_group_semantics_are_preserved(metadata, expected):
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight, model.bias], **metadata})

    assert audit_optimizer_param_groups(model, optimizer)["groups"][0]["name"] == expected


def test_missing_group_semantics_use_index_fallback():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight]}, {"params": [model.bias]})

    groups = audit_optimizer_param_groups(model, optimizer)["groups"]

    assert [group["name"] for group in groups] == ["group_0", "group_1"]
    assert groups[0]["initial_lr"] is None
    assert groups[0]["lr_scale"] is None


def test_parameter_name_does_not_create_inferred_adapter_role():
    class _OrdinaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter_projection = nn.Parameter(torch.ones(2))

    model = _OrdinaryModel()
    optimizer = _optimizer({"params": [model.adapter_projection]})

    audit = audit_optimizer_param_groups(model, optimizer)

    assert audit["groups"][0]["name"] == "group_0"
    assert audit["groups"][0]["parameter_names"] == ["adapter_projection"]


def test_audit_result_is_json_serializable():
    model = _AuditModel()
    optimizer = _optimizer({"params": [model.weight, model.bias], "group_name": "ordinary"})

    json.dumps(audit_optimizer_param_groups(model, optimizer))


def test_scheduler_update_is_visible_in_a_new_snapshot():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(
        [{"params": list(model.parameters()), "group_name": "ordinary", "lr": 0.1}],
        lr=0.1,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch == 0 else 0.5)
    before = audit_optimizer_param_groups(model, optimizer)

    optimizer.step()
    scheduler.step()
    after = audit_optimizer_param_groups(model, optimizer)

    assert after["groups"][0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    assert after["groups"][0]["lr"] != before["groups"][0]["lr"]
    assert after["groups"][0]["initial_lr"] == pytest.approx(0.1)


def test_trainer_audit_is_read_only_for_peft_named_group():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "param_group": "adapter", "lr": 0.003, "weight_decay": 0.0}]
    )
    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    before_groups = [
        {
            "parameter_ids": [id(parameter) for parameter in group["params"]],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
        }
        for group in optimizer.param_groups
    ]

    audit = trainer._audit_optimizer_groups()

    after_groups = [
        {
            "parameter_ids": [id(parameter) for parameter in group["params"]],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
        }
        for group in optimizer.param_groups
    ]
    assert before_groups == after_groups
    assert trainer.optimizer_group_audit is audit
    assert audit["groups"][0]["name"] == "adapter"


def test_train_pipeline_audits_after_adapter_configuration_and_before_scheduler():
    trainer = object.__new__(BaseTrainer)
    trainer.model = nn.Linear(2, 1)
    trainer.batch_size = 2
    trainer.world_size = 0
    trainer.epochs = 1
    trainer.data = {"train": "train", "val": "val"}
    trainer.args = SimpleNamespace(
        task="detect",
        nbs=2,
        weight_decay=0.001,
        optimizer="AdamW",
        lr0=0.01,
        momentum=0.9,
    )
    loader = SimpleNamespace(dataset=[0, 1])
    trainer.get_dataloader = lambda *args, **kwargs: loader
    events = []

    class _Controller:
        @staticmethod
        def prepare_optimizer(iterations):
            events.append("prepare")

        @staticmethod
        def configure_optimizer(optimizer):
            events.append("configure")

    trainer.adapter_controller = _Controller()

    def _build_optimizer(**kwargs):
        events.append("build")
        return torch.optim.AdamW(trainer.model.parameters(), lr=0.01)

    trainer.build_optimizer = _build_optimizer
    trainer._audit_optimizer_groups = lambda: events.append("audit")
    trainer._save_run_args = lambda: events.append("save")
    trainer._setup_scheduler = lambda: events.append("scheduler")

    trainer._build_train_pipeline()

    assert events == ["prepare", "build", "configure", "audit", "save", "scheduler"]
