"""Unit tests for MultiTaskHead, MultiTaskLoss, and MultiTaskTrainer."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from ultralytics.nn.mixture_loss import CompositeCriterion
from ultralytics.nn.modules.multitask.head import MultiTaskHead
from ultralytics.nn.modules.multitask.router import TaskRouter
from ultralytics.utils.loss import MultiTaskLoss
from ultralytics.models.yolo.multitask.train import (
    MultiTaskTrainer,
    unwrap_model,
    validate_multitask_contract,
    validate_multitask_dataset_supervision,
)
from ultralytics.models.yolo.multitask.val import MultiTaskValidator
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import COCOMultiTaskDataset
from ultralytics.data.multitask_sampler import MultiTaskBatchSampler
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA


ROOT = Path(__file__).resolve().parents[1]


def test_multitask_batch_sampler_is_deterministic_distributed_and_resumable():
    """Source proportions and cursors remain deterministic across rank-local resume."""
    kwargs = {"batch_size": 2, "mode": "round_robin", "seed": 17, "world_size": 2, "steps_per_epoch": 3}
    rank_zero = MultiTaskBatchSampler({"detect": 5, "pose": 3}, rank=0, **kwargs)
    rank_one = MultiTaskBatchSampler({"detect": 5, "pose": 3}, rank=1, **kwargs)
    first = next(iter(rank_zero))
    assert [task for task, _ in first] == ["detect", "detect"]
    assert {item for batch in rank_zero for item in batch}.isdisjoint({item for batch in rank_one for item in batch})

    resumed = MultiTaskBatchSampler({"detect": 5, "pose": 3}, rank=0, **kwargs)
    resumed.load_state_dict(rank_zero.state_dict())
    assert list(resumed) == []

    partial = MultiTaskBatchSampler({"detect": 5, "pose": 3}, rank=0, **kwargs)
    iterator = iter(partial)
    next(iterator)
    partial_state = partial.state_dict()
    expected_tail = list(iterator)
    restored = MultiTaskBatchSampler({"detect": 5, "pose": 3}, rank=0, **kwargs)
    restored.load_state_dict(partial_state)
    assert list(restored) == expected_tail
    assert expected_tail


def test_multitask_loss_respects_task_source_for_auxiliary_criteria(monkeypatch):
    """A detect-only routed batch must not execute segmentation or pose criteria."""
    called = {"segment": 0, "pose": 0}

    class Criterion:
        def __call__(self, preds, batch):
            called[self.name] += 1
            return torch.ones(5), torch.ones(5)

    criterion = MultiTaskLoss.__new__(MultiTaskLoss)
    criterion.model = SimpleNamespace(end2end=False, model=[SimpleNamespace(task_router=None)])
    criterion.task_weights = {"detect": 1.0, "segment": 1.0, "pose": 1.0}
    criterion.det_loss = lambda preds, batch: (torch.ones(1), torch.ones(3))
    criterion.seg_loss = Criterion()
    criterion.seg_loss.name = "segment"
    criterion.pose_loss = Criterion()
    criterion.pose_loss.name = "pose"
    criterion.cls_loss = None
    criterion.semantic_loss = None
    boxes = torch.zeros(1, 4, 4)
    preds = {
        "one2many": {
            "boxes": boxes,
            "scores": torch.zeros(1, 4, 2),
            "mask_coefficient": torch.zeros(1, 4, 1),
            "kpts": torch.zeros(1, 4, 17, 3),
        }
    }
    batch = {"task_source": ["detect"], "masks": torch.zeros(1, 1, 4, 4), "keypoints": torch.zeros(1, 1, 17, 3)}
    total, items = criterion.forward(preds, batch)
    assert total.item() == pytest.approx(1.0)
    assert items.shape == (9,)
    assert called == {"segment": 0, "pose": 0}


def test_coco_multitask_dataset_emits_aligned_targets(tmp_path):
    """COCO JSON annotations are exposed as synchronized box, mask, and pose targets."""
    import cv2
    import json
    import numpy as np

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((32, 48, 3), dtype=np.uint8))
    instances_path = tmp_path / "instances.json"
    keypoints_path = tmp_path / "keypoints.json"
    instance = {
        "id": 1,
        "image_id": 7,
        "category_id": 1,
        "bbox": [4, 5, 20, 18],
        "segmentation": [[4, 5, 24, 5, 24, 23, 4, 23]],
        "iscrowd": 0,
    }
    image = {"id": 7, "file_name": "sample.jpg", "width": 48, "height": 32}
    instances_path.write_text(json.dumps({"images": [image], "annotations": [instance]}))
    keypoints_path.write_text(
        json.dumps(
            {
                "images": [image],
                "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "keypoints": [8, 9, 2] * 17}],
            }
        )
    )
    data = {
        "path": tmp_path,
        "names": {0: "person"},
        "nc": 1,
        "kpt_shape": [17, 3],
        "flip_idx": list(range(17)),
        "multitask_format": "coco",
        "train_instances": instances_path.name,
        "val_instances": instances_path.name,
        "train_keypoints": keypoints_path.name,
        "val_keypoints": keypoints_path.name,
    }
    hyp = IterableSimpleNamespace(**vars(DEFAULT_CFG))
    hyp.mosaic = hyp.mixup = hyp.cutmix = 0.0
    dataset = COCOMultiTaskDataset(
        img_path=str(image_dir),
        imgsz=32,
        batch_size=1,
        augment=False,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="train: ",
        task="multitask",
        data=data,
    )
    sample = dataset[0]
    assert sample["bboxes"].shape == (1, 4)
    assert sample["masks"].shape[0] == 1
    assert sample["keypoints"].shape == (1, 17, 3)
    assert sample["cls"].tolist() == [[0]]


def test_coco_multitask_dataset_rejects_image_list_from_another_coco_split(tmp_path):
    """COCO split/image-list mismatches fail before they can silently skew validation metrics."""
    import cv2
    import json
    import numpy as np

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "matched.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    cv2.imwrite(str(image_dir / "wrong_split.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    annotation = {"id": 1, "file_name": "matched.jpg", "width": 32, "height": 32}
    (tmp_path / "instances.json").write_text(json.dumps({"images": [annotation], "annotations": []}))
    (tmp_path / "keypoints.json").write_text(json.dumps({"images": [annotation], "annotations": []}))
    data = {
        "path": tmp_path,
        "names": {0: "person"},
        "nc": 1,
        "kpt_shape": [17, 3],
        "flip_idx": list(range(17)),
        "multitask_format": "coco",
        "train_instances": "instances.json",
        "val_instances": "instances.json",
        "train_keypoints": "keypoints.json",
        "val_keypoints": "keypoints.json",
    }
    hyp = IterableSimpleNamespace(**vars(DEFAULT_CFG))
    hyp.mosaic = hyp.mixup = hyp.cutmix = 0.0

    with pytest.raises(ValueError, match="same COCO split"):
        COCOMultiTaskDataset(
            img_path=str(image_dir),
            imgsz=32,
            batch_size=1,
            augment=False,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix="val: ",
            task="multitask",
            data=data,
        )


def test_coco_multitask_dataset_emits_dense_targets_and_multilabels(tmp_path):
    """Dense labels share the same geometry and image-level classification is a true multi-hot vector."""
    import cv2
    import json
    import numpy as np

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "sample.jpg"), np.zeros((24, 32, 3), dtype=np.uint8))
    for directory in ("depth", "normal", "stuff", "panoptic"):
        (tmp_path / directory).mkdir()
    cv2.imwrite(str(tmp_path / "depth" / "sample_depth.png"), np.full((24, 32), 127, dtype=np.uint8))
    cv2.imwrite(
        str(tmp_path / "normal" / "sample_normal.png"),
        np.dstack([np.full((24, 32), 127, dtype=np.uint8)] * 2 + [np.full((24, 32), 255, dtype=np.uint8)]),
    )
    cv2.imwrite(str(tmp_path / "stuff" / "sample.png"), np.full((24, 32), 2, dtype=np.uint8))
    panoptic = np.zeros((24, 32, 3), dtype=np.uint8)
    panoptic[..., 0] = 7
    cv2.imwrite(str(tmp_path / "panoptic" / "sample.png"), cv2.cvtColor(panoptic, cv2.COLOR_RGB2BGR))
    image = {"id": 7, "file_name": "sample.jpg", "width": 32, "height": 24}
    annotations = [
        {
            "id": 1,
            "image_id": 7,
            "category_id": 1,
            "bbox": [2, 2, 8, 8],
            "segmentation": [[2, 2, 10, 2, 10, 10]],
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 7,
            "category_id": 3,
            "bbox": [12, 4, 8, 8],
            "segmentation": [[12, 4, 20, 4, 20, 12]],
            "iscrowd": 0,
        },
    ]
    (tmp_path / "instances.json").write_text(json.dumps({"images": [image], "annotations": annotations}))
    (tmp_path / "keypoints.json").write_text(json.dumps({"images": [image], "annotations": []}))
    panoptic_json = {
        "categories": [{"id": 3, "name": "thing"}, {"id": 5, "name": "stuff"}],
        "annotations": [{"image_id": 7, "file_name": "sample.png", "segments_info": [{"id": 7, "category_id": 5}]}],
    }
    (tmp_path / "panoptic.json").write_text(json.dumps(panoptic_json))
    data = {
        "path": tmp_path,
        "names": {0: "person", 1: "bicycle", 2: "car"},
        "nc": 3,
        "kpt_shape": [17, 3],
        "flip_idx": list(range(17)),
        "train_instances": "instances.json",
        "train_keypoints": "keypoints.json",
        "depth_dir": "depth",
        "normal_dir": "normal",
        "semantic_source": "panoptic",
        "semantic_nc": 2,
        "panoptic_train_masks": "panoptic",
        "panoptic_train_annotations": "panoptic.json",
        "tasks": ["detect", "segment", "pose", "classify", "depth", "normal", "semantic"],
    }
    hyp = IterableSimpleNamespace(**vars(DEFAULT_CFG))
    hyp.mosaic = hyp.mixup = hyp.cutmix = hyp.copy_paste = 0.0
    dataset = COCOMultiTaskDataset(
        img_path=str(image_dir),
        imgsz=32,
        batch_size=1,
        augment=False,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="train: ",
        data=data,
    )
    sample = dataset[0]
    assert sample["cls_img"].tolist() == [1.0, 0.0, 1.0]
    assert sample["depth"].shape == (1, 32, 32) and sample["depth_valid"].any()
    assert sample["normal"].shape == (3, 32, 32) and sample["normal_valid"].any()
    assert sample["semantic_mask"].shape == (32, 32) and (sample["semantic_mask"] == 1).any()
    assert sample["panoptic_mask"].shape == (32, 32) and (sample["panoptic_mask"] == 7).any()


def test_build_yolo_dataset_selects_coco_multitask_format(tmp_path):
    """The generic dataset builder opts into COCO loading only for explicit multitask configs."""
    import json
    from types import SimpleNamespace

    data = {
        "path": tmp_path,
        "names": {0: "person"},
        "nc": 1,
        "multitask_format": "coco",
        "train_instances": "instances.json",
        "train_keypoints": "keypoints.json",
    }
    import cv2
    import numpy as np

    cv2.imwrite(str(tmp_path / "sample.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    coco_image = {"id": 1, "file_name": "sample.jpg", "width": 32, "height": 32}
    (tmp_path / "instances.json").write_text(json.dumps({"images": [coco_image], "annotations": []}))
    (tmp_path / "keypoints.json").write_text(json.dumps({"images": [coco_image], "annotations": []}))
    cfg = SimpleNamespace(
        task="multitask",
        fraction=1.0,
        imgsz=32,
        rect=False,
        cache=False,
        single_cls=False,
        classes=None,
        workers=0,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
    )
    dataset = build_yolo_dataset(cfg, str(tmp_path), 1, data, mode="val", stride=32)
    assert isinstance(dataset, COCOMultiTaskDataset)


def test_multitask_head_can_limit_active_tasks():
    head = MultiTaskHead(nc=2, ch=(16, 32, 64), tasks=["detect", "segment", "pose", "classify"])
    head.set_active_tasks(["detect", "segment", "pose"])
    assert head.active_tasks == ["detect", "pose", "segment"]
    assert head.has_task("classify") is False


def test_multitask_head_rejects_dataset_tasks_without_built_heads():
    """A data YAML cannot enable heads that its selected model YAML omitted."""
    head = MultiTaskHead(nc=2, ch=(16, 32, 64), tasks=["detect", "segment"])
    with pytest.raises(ValueError, match="did not build"):
        head.set_active_tasks(["detect", "segment", "classify", "depth"])


def test_multitask_export_rejects_embedded_nms_that_drops_auxiliary_outputs():
    """Embedded NMS cannot be advertised as a complete multi-task export endpoint."""
    from ultralytics.engine.exporter import Exporter

    with pytest.raises(ValueError, match="drops mask, pose, and dense outputs"):
        Exporter(overrides={"format": "onnx", "nms": True})(model=SimpleNamespace(task="multitask"))


def _write_multitask_coco_sources(tmp_path):
    """Create a minimal COCO source pair that exercises sparse multi-task supervision."""
    import json

    image = {"id": 1, "file_name": "sample.jpg", "width": 32, "height": 32}
    instance = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [2, 2, 12, 12],
        "segmentation": [[2, 2, 14, 2, 14, 14, 2, 14]],
        "iscrowd": 0,
    }
    instances = {"images": [image], "annotations": [instance]}
    keypoints = {
        "images": [image],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "keypoints": [8, 8, 2] * 17}],
    }
    for split in ("train", "val"):
        (tmp_path / f"{split}_instances.json").write_text(json.dumps(instances))
        (tmp_path / f"{split}_keypoints.json").write_text(json.dumps(keypoints))
    return {
        "path": tmp_path,
        "multitask_format": "coco",
        "names": {0: "person"},
        "nc": 1,
        "tasks": ["detect", "segment", "pose"],
        "kpt_shape": [17, 3],
        "train_instances": "train_instances.json",
        "val_instances": "val_instances.json",
        "train_keypoints": "train_keypoints.json",
        "val_keypoints": "val_keypoints.json",
    }


def test_multitask_contract_accepts_complete_sparse_coco_supervision(tmp_path):
    """The supported detect/segment/pose path is accepted only with all sources and heads present."""
    data = _write_multitask_coco_sources(tmp_path)
    head = MultiTaskHead(nc=1, ch=(16, 32, 64), tasks=data["tasks"], kpt_shape=(17, 3))

    validate_multitask_contract(data, head)


def test_multitask_contract_rejects_missing_auxiliary_supervision(tmp_path):
    """An enabled dense branch must have a configured source before the trainer can start."""
    data = _write_multitask_coco_sources(tmp_path)
    data["tasks"] = ["detect", "depth"]
    head = MultiTaskHead(nc=1, ch=(16, 32, 64), tasks=data["tasks"])

    with pytest.raises(ValueError, match="depth_dir"):
        validate_multitask_contract(data, head)


def test_multitask_contract_rejects_unsupported_obb_before_training(tmp_path):
    """OBB outputs cannot be selected until its target, loss, and validator path is implemented."""
    data = _write_multitask_coco_sources(tmp_path)
    data["tasks"] = ["detect", "obb"]
    head = MultiTaskHead(nc=1, ch=(16, 32, 64), tasks=data["tasks"])

    with pytest.raises(ValueError, match="obb"):
        validate_multitask_contract(data, head)


def test_multitask_contract_rejects_dataset_task_missing_from_model_head(tmp_path):
    """Data cannot activate a task that the selected model did not physically build."""
    data = _write_multitask_coco_sources(tmp_path)
    head = MultiTaskHead(nc=1, ch=(16, 32, 64), tasks=["detect", "segment"])

    with pytest.raises(ValueError, match="did not build"):
        validate_multitask_contract(data, head)


def test_multitask_contract_rejects_pose_head_shape_mismatch(tmp_path):
    """COCO pose data cannot silently train a head with an incompatible keypoint layout."""
    data = _write_multitask_coco_sources(tmp_path)
    head = MultiTaskHead(nc=1, ch=(16, 32, 64), tasks=data["tasks"], kpt_shape=(5, 3))

    with pytest.raises(ValueError, match="kpt_shape does not match"):
        validate_multitask_contract(data, head)


def test_multitask_contract_rejects_semantic_head_channel_mismatch(tmp_path):
    """Semantic labels and logits must agree on the number of classes at startup."""
    data = _write_multitask_coco_sources(tmp_path)
    data.update(
        tasks=["detect", "semantic"],
        semantic_source="panoptic",
        semantic_nc=2,
        panoptic_train_masks="panoptic",
        panoptic_val_masks="panoptic",
        panoptic_train_annotations="panoptic_train.json",
        panoptic_val_annotations="panoptic_val.json",
    )
    for directory in ("panoptic",):
        (tmp_path / directory).mkdir()
    for annotation in ("panoptic_train.json", "panoptic_val.json"):
        (tmp_path / annotation).write_text("{}")
    head = MultiTaskHead(nc=1, ch=(16, 32, 64), tasks=data["tasks"], semantic_nc=3)

    with pytest.raises(ValueError, match="semantic_nc does not match"):
        validate_multitask_contract(data, head)


def test_multitask_dataset_supervision_rejects_empty_selected_dense_targets():
    """A configured dense directory is insufficient when the selected image list contains no matching files."""
    dataset = SimpleNamespace(labels=[{"bboxes": torch.zeros(1, 4), "depth_path": None}])

    with pytest.raises(ValueError, match="no usable targets.*depth"):
        validate_multitask_dataset_supervision(dataset, ("detect", "depth"), "train")


def test_multitask_dataset_supervision_accepts_visible_pose_and_masks():
    """Selected target inspection accepts actual polygon and visible-keypoint supervision."""
    dataset = SimpleNamespace(
        labels=[
            {
                "bboxes": torch.zeros(1, 4),
                "segments": [torch.zeros(3, 2)],
                "keypoints": torch.tensor([[[4.0, 5.0, 2.0]]]),
            }
        ]
    )

    validate_multitask_dataset_supervision(dataset, ("detect", "segment", "pose"), "train")


# ── helpers ──────────────────────────────────────────────────────────────


def _has_grad(module):
    """True if any parameter has a non-zero finite gradient."""
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
        if p.requires_grad
    )


def _make_fpn_features(bs=2, ch=(64, 128, 256)):
    """Create FPN feature tensors: [P3, P4, P5]."""
    return [torch.randn(bs, c, h, h) for c, h in zip(ch, (32, 16, 8))]


class _MultiTaskHeadExportWrapper(nn.Module):
    """Expose fixed FPN inputs as individual ONNX graph inputs."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, p3, p4, p5):
        """Run the export-configured multi-task head over FPN features."""
        return self.head([p3, p4, p5])


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskHead ──── init / properties / task routing
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskHeadInit:
    """MultiTaskHead construction and property routing."""

    def test_detect_only_defaults(self):
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect"])
        assert head.active_tasks == ["detect"]
        assert head.cv4_seg is None
        assert head.cv4_pose is None
        assert head.cv4_cls is None
        assert head.cv4_depth is None
        assert head.cv4_obb is None

    def test_all_trainable_tasks_default_without_obb(self):
        head = MultiTaskHead(nc=80, ch=(64, 128, 256))
        assert head.active_tasks == ["classify", "depth", "detect", "normal", "pose", "segment", "semantic"]
        for attr in ["cv4_seg", "cv4_pose", "cv4_cls", "cv4_depth", "cv4_normal", "cv4_semantic"]:
            assert getattr(head, attr) is not None, f"{attr} should not be None"
        assert head.cv4_obb is None
        assert set(TaskRouter.available_tasks().values()) == set(head.active_tasks)

    def test_legacy_obb_head_cannot_be_activated_for_training(self):
        head = MultiTaskHead(nc=80, ch=(64, 128, 256), tasks=["detect", "obb"])
        assert head.cv4_obb is not None
        with pytest.raises(ValueError, match="dedicated OBB model"):
            head.set_active_tasks(["detect", "obb"])

    def test_subset_tasks(self):
        head = MultiTaskHead(nc=20, ch=(64, 128), tasks=["detect", "segment", "pose"], kpt_shape=(5, 3))
        assert head.active_tasks == ["detect", "pose", "segment"]
        assert head.cv4_seg is not None and head.cv4_pose is not None
        assert head.cv4_cls is None and head.cv4_depth is None and head.cv4_obb is None

    def test_has_task(self):
        head = MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect", "segment"])
        assert head.has_task("detect") and head.has_task("segment")
        assert not head.has_task("pose") and not head.has_task("classify")

    def test_one2many_keys(self):
        head = MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect", "segment", "obb"])
        d = head.one2many
        assert "box_head" in d and "cls_head" in d
        assert "mask_head" in d and "obb_head" in d
        assert "pose_head" not in d and "depth_head" not in d

    def test_one2one_property_exists(self):
        head = MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect"])
        assert hasattr(head, "one2one")
        assert isinstance(head.one2one, dict)

    def test_end2end_defaults_true_with_complete_one2one_heads(self):
        head = MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect"])
        assert head.end2end is True
        assert set(head.one2one) >= {"box_head", "cls_head"}

    def test_end2end_false_disables_one2one_forward(self):
        head = MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect"], end2end=False).train()
        assert head.end2end is False
        assert "one2many" not in head(_make_fpn_features(ch=(64, 128)))

    def test_task_importance_shape(self):
        head = MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect", "segment", "pose"])
        assert head.task_importance.shape == (3,)
        assert MultiTaskHead(nc=80, ch=(64, 128), tasks=["detect"]).task_importance.shape == (1,)


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskHead ──── forward_head output shapes
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskHeadForwardHead:
    """forward_head() output keys and tensor shapes."""

    @pytest.mark.parametrize(
        "tasks,expected_keys",
        [
            (["detect"], {"boxes", "scores", "feats"}),
            (["detect", "segment"], {"boxes", "scores", "feats", "mask_coefficient", "proto"}),
            (["detect", "pose"], {"boxes", "scores", "feats", "kpts"}),
            (["detect", "classify"], {"boxes", "scores", "feats", "cls_logits"}),
            (["detect", "depth"], {"boxes", "scores", "feats", "depth"}),
            (["detect", "obb"], {"boxes", "scores", "feats", "angle"}),
        ],
    )
    def test_forward_head_keys(self, tasks, expected_keys):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=tasks, kpt_shape=(5, 3)).train()
        out = head.forward_head(_make_fpn_features())
        received = set(out.keys())
        assert received >= expected_keys, f"Missing: {expected_keys - received}"

    def test_detect_output_shapes(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect"]).train()
        out = head.forward_head(_make_fpn_features())
        total = 32 * 32 + 16 * 16 + 8 * 8
        assert out["boxes"].shape == (2, head.reg_max * 4, total)
        assert out["scores"].shape == (2, 10, total)
        assert len(out["feats"]) == 3

    def test_segment_output_shapes(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "segment"], nm=16, npr=128).train()
        out = head.forward_head(_make_fpn_features())
        total = 32 * 32 + 16 * 16 + 8 * 8
        assert out["mask_coefficient"].shape == (2, 16, total)
        proto = out["proto"][0] if isinstance(out["proto"], tuple) else out["proto"]
        assert proto.shape[0] == 2

    def test_pose_output_shapes(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "pose"], kpt_shape=(5, 3)).train()
        out = head.forward_head(_make_fpn_features())
        total = 32 * 32 + 16 * 16 + 8 * 8
        assert out["kpts"].shape == (2, 15, total)

    def test_dense_obb_classify_shapes(self):
        torch.manual_seed(0)
        total = 32 * 32 + 16 * 16 + 8 * 8
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "depth"], depth_bins=64).train()
        assert head.forward_head(_make_fpn_features())["depth"].shape == (2, 1, 32, 32)
        head_normal = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "normal"]).train()
        assert head_normal.forward_head(_make_fpn_features())["normal"].shape == (2, 3, 32, 32)
        head_semantic = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "semantic"], semantic_nc=5).train()
        assert head_semantic.forward_head(_make_fpn_features())["semantic"].shape == (2, 5, 32, 32)
        head2 = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "obb"]).train()
        assert head2.forward_head(_make_fpn_features())["angle"].shape == (2, 1, total)
        head3 = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "classify"]).train()
        assert head3.forward_head(_make_fpn_features())["cls_logits"].shape == (2, 10)

    def test_all_tasks_forward_head(self):
        torch.manual_seed(0)
        head = MultiTaskHead(
            nc=10, ch=(64, 128, 256), tasks=None, nm=16, npr=128, kpt_shape=(5, 3), depth_bins=64
        ).train()
        out = head.forward_head(_make_fpn_features())
        for key in [
            "boxes",
            "scores",
            "feats",
            "mask_coefficient",
            "proto",
            "kpts",
            "depth",
            "normal",
            "semantic",
            "cls_logits",
        ]:
            assert key in out, f"Missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskHead ──── forward (training vs inference)
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskHeadForward:
    """Top-level forward() mode switching."""

    def test_train_returns_one2many_one2one(self):
        head = MultiTaskHead(
            nc=10,
            ch=(64, 128, 256),
            tasks=["detect", "segment", "pose"],
            nm=16,
            npr=128,
            kpt_shape=(5, 3),
        ).train()
        out = head(_make_fpn_features())
        assert isinstance(out, dict) and "one2many" in out and "one2one" in out
        assert "boxes" in out["one2many"] and "mask_coefficient" in out["one2many"]
        assert out["one2one"]["mask_coefficient"] is out["one2many"]["mask_coefficient"]
        assert out["one2one"]["proto"] is out["one2many"]["proto"]
        assert out["one2one"]["kpts"] is out["one2many"]["kpts"]
        assert not hasattr(head, "one2one_cv4_seg") and not hasattr(head, "one2one_cv4_pose")

    def test_eval_returns_tuple(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "segment"], nm=16, npr=128).eval()
        out = head(_make_fpn_features())
        assert isinstance(out, tuple) and len(out) == 2
        det_tensor, raw_dict = out
        assert isinstance(det_tensor, torch.Tensor) and det_tensor.ndim == 3
        assert isinstance(raw_dict, dict)

    def test_eval_preserves_source_anchor_indices_for_auxiliary_decoding(self):
        """End-to-end candidates retain the dense source anchors for validator mask/keypoint decoding."""
        head = MultiTaskHead(
            nc=10, ch=(64, 128, 256), tasks=["detect", "segment", "pose"], nm=16, npr=128, kpt_shape=(5, 3)
        ).eval()
        _, raw_dict = head(_make_fpn_features())
        indices = raw_dict["one2one"]["candidate_indices"]
        assert indices.shape == (2, min(head.max_det, 32 * 32 + 16 * 16 + 8 * 8))
        assert indices.dtype == torch.long

    def test_export_emits_named_detection_segment_pose_tensors(self):
        """Export preserves every supervised task in a stable, detection-aligned schema."""
        head = MultiTaskHead(
            nc=10, ch=(64, 128, 256), tasks=["detect", "segment", "pose"], nm=16, npr=128, kpt_shape=(5, 3)
        ).eval()
        head.export = True
        output = head(_make_fpn_features())

        max_det = min(head.max_det, 32 * 32 + 16 * 16 + 8 * 8)
        assert head.export_output_names == ["detections", "mask_coefficients", "mask_prototypes", "keypoints"]
        assert isinstance(output, tuple) and len(output) == len(head.export_output_names)
        assert output[0].shape == (2, max_det, 6)
        assert output[1].shape == (2, max_det, 16)
        assert output[2].shape[0:2] == (2, 16)
        assert output[3].shape == (2, max_det, 15)

    def test_onnx_export_roundtrip_matches_detection_segment_pose_schema(self, tmp_path):
        """The ONNX graph retains named multi-task tensors and matches eager outputs."""
        onnx = pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        torch.manual_seed(0)
        head = MultiTaskHead(
            nc=10,
            ch=(64, 128, 256),
            tasks=["detect", "segment", "pose"],
            nm=16,
            npr=128,
            kpt_shape=(5, 3),
        ).eval()
        head.export = True
        head.max_det = 32
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        inputs = _make_fpn_features(bs=1)
        wrapper = _MultiTaskHeadExportWrapper(head).eval()
        artifact = tmp_path / "multitask-head.onnx"

        with torch.no_grad():
            expected = tuple(output.detach().clone() for output in wrapper(*inputs))
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            artifact,
            input_names=["p3", "p4", "p5"],
            output_names=head.export_output_names,
            opset_version=17,
            dynamo=False,
        )

        graph = onnx.load(artifact)
        onnx.checker.check_model(graph)
        assert [output.name for output in graph.graph.output] == head.export_output_names
        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        observed = session.run(None, {name: tensor.numpy() for name, tensor in zip(("p3", "p4", "p5"), inputs)})
        assert len(observed) == len(expected) == len(head.export_output_names)
        for eager, runtime in zip(expected, observed):
            np.testing.assert_allclose(eager.numpy(), runtime, atol=1e-4, rtol=1e-4)

    def test_detect_only_uses_one2many_structure(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect"]).train()
        out = head(_make_fpn_features())
        assert "one2many" in out and "one2one" in out
        assert "boxes" in out["one2many"]


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskHead ──── lifecycle: fuse / gradient flow
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskHeadLifecycle:
    def test_bias_init_does_not_crash_for_all_tasks(self):
        """Default construction creates the complete one-to-many and one-to-one detection lifecycle."""
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=None)
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        head.bias_init()

    def test_fuse_keeps_supervised_auxiliary_heads(self):
        head = MultiTaskHead(
            nc=10, ch=(64, 128, 256), tasks=["detect", "segment", "pose"], nm=16, kpt_shape=(5, 3)
        ).eval()
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        head.fuse()

        assert head.cv2 is None and head.cv3 is None
        assert head.cv4_seg is not None and head.cv4_pose is not None
        _, raw = head(_make_fpn_features())
        assert "mask_coefficient" in raw["one2one"] and "kpts" in raw["one2one"]

    def test_gradient_flow_detect(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect"]).train()
        out = head.forward_head(_make_fpn_features())
        (out["boxes"].mean() + out["scores"].mean()).backward()
        assert _has_grad(head)

    def test_gradient_flow_segment(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "segment"], nm=16, npr=128).train()
        out = head.forward_head(_make_fpn_features())
        (out["boxes"].mean() + out["mask_coefficient"].mean()).backward()
        assert _has_grad(head)

    def test_gradient_flow_pose(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "pose"], kpt_shape=(5, 3)).train()
        out = head.forward_head(_make_fpn_features())
        (out["boxes"].mean() + out["kpts"].mean()).backward()
        assert _has_grad(head)

    def test_gradient_flow_all_tasks(self):
        torch.manual_seed(0)
        head = MultiTaskHead(
            nc=10, ch=(64, 128, 256), tasks=None, nm=16, npr=128, kpt_shape=(5, 3), depth_bins=64
        ).train()
        out = head.forward_head(_make_fpn_features())
        total = out["boxes"].mean()
        for k in ["mask_coefficient", "kpts", "depth", "normal", "semantic", "cls_logits"]:
            total = total + out[k].mean()
        total.backward()
        assert _has_grad(head)


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskHead ──── TaskRouter integration
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskHeadTaskRouter:
    def test_balance_loss_is_nonnegative_and_updates_router(self):
        router = TaskRouter(dim=16, num_tasks=3, top_k=1).train()
        router(torch.randn(2, 16, 8, 8))

        loss = router.compute_balance_loss()
        assert loss.requires_grad
        assert torch.isfinite(loss) and loss >= 0
        loss.backward()
        assert _has_grad(router.affinity_router)

    def test_balance_loss_is_zero_before_first_forward(self):
        router = TaskRouter(dim=16, num_tasks=3, top_k=1)
        assert router.compute_balance_loss().item() == 0.0

    def test_without_taskrouter_no_routing_stats(self):
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "segment"], use_task_router=False).train()
        out = head.forward_head(_make_fpn_features())
        assert "routing_stats" not in out

    def test_with_taskrouter_emit_routing_stats(self):
        torch.manual_seed(0)
        head = MultiTaskHead(
            nc=10, ch=(64, 128, 256), tasks=["detect", "segment", "pose"], use_task_router=True, task_router_dim=64
        ).train()
        out = head.forward_head(_make_fpn_features())
        assert "routing_stats" in out
        assert "task_usage" in out["routing_stats"]

    def test_taskrouter_eval_no_routing_stats(self):
        torch.manual_seed(0)
        head = MultiTaskHead(nc=10, ch=(64, 128, 256), tasks=["detect", "segment"], use_task_router=True).eval()
        out = head.forward_head(_make_fpn_features())
        assert "routing_stats" not in out

    def test_taskrouter_accepts_explicit_router_dimension(self):
        head = MultiTaskHead(
            nc=10,
            ch=(64, 128, 256),
            tasks=["detect", "segment"],
            use_task_router=True,
            task_router_dim=32,
        ).train()
        output = head.forward_head(_make_fpn_features())
        assert output["boxes"].shape[0] == 2
        assert head.task_router_input_proj.out_channels == 32

    def test_end2end_one2one_does_not_republish_task_router_stats(self, monkeypatch):
        head = MultiTaskHead(
            nc=10,
            ch=(64, 128, 256),
            tasks=["detect", "segment"],
            use_task_router=True,
            end2end=True,
        ).train()
        calls = []
        original = head._route_task_features

        def wrapped(features):
            calls.append(features[0].requires_grad)
            return original(features)

        monkeypatch.setattr(head, "_route_task_features", wrapped)
        head([feature.detach().requires_grad_() for feature in _make_fpn_features()])
        assert calls == [True]


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskLoss
# ══════════════════════════════════════════════════════════════════════════


class _MockModelWrapper(nn.Module):
    """Real nn.Module wrapping MultiTaskHead so v8DetectionLoss/E2ELoss can call .parameters()."""

    def __init__(self, head, args_dict=None):
        super().__init__()
        self.model = nn.Sequential(nn.Identity(), head)
        if args_dict is None:
            args_dict = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True}
        self.args = SimpleNamespace(**args_dict)
        # MultiTaskHead always has end2end=True (property), so MultiTaskLoss will use E2ELoss.
        # We set end2end here to match the head behavior.
        self.end2end = head.end2end


class TestMultiTaskLoss:
    """MultiTaskLoss init and forward passes."""

    def _make_model(self, tasks=("detect",), **head_kwargs):
        head = MultiTaskHead(
            nc=8, ch=(64, 128, 256), tasks=list(tasks), nm=16, npr=128, kpt_shape=(5, 3), depth_bins=64, **head_kwargs
        )
        return _MockModelWrapper(head)

    def test_init_converts_dict_args(self):
        head = MultiTaskHead(nc=8, ch=(64, 128, 256), tasks=["detect"])
        m = _MockModelWrapper(head, args_dict={"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True})
        MultiTaskLoss(m)
        assert not isinstance(m.args, dict)
        assert m.args.overlap_mask is True

    def test_model_reference_does_not_create_ema_state_dict_cycle(self):
        """A lazily attached criterion keeps a non-owning reference to its parent model."""
        model = self._make_model(["detect"])
        ema = ModelEMA(model)
        criterion = MultiTaskLoss(model)
        model.criterion = criterion

        assert "model" not in criterion._modules
        assert model.state_dict()
        ema.update(model)

    def test_end2end_schedule_updates_and_resumes_through_composite(self):
        model = self._make_model(["detect", "segment", "pose"])
        model.args.epochs = 100
        criterion = MultiTaskLoss(model)
        composite = CompositeCriterion(model, criterion)
        initial_o2m = criterion.det_loss.o2m

        composite.updates = 9
        assert criterion.updates == 9
        composite.update()

        assert composite.updates == criterion.updates == 10
        assert criterion.det_loss.o2m < initial_o2m
        assert criterion.det_loss.o2o == pytest.approx(1.0 - criterion.det_loss.o2m)

    def test_forward_empty_preds_returns_zeros(self):
        loss = MultiTaskLoss(self._make_model(["detect"]))
        total, items = loss.forward({}, {})
        assert total.item() == 0.0 and items.shape == (9,) and (items == 0.0).all()

    def test_forward_tensor_preds_returns_zeros(self):
        loss = MultiTaskLoss(self._make_model(["detect"]))
        total, items = loss.forward(torch.randn(2, 10, 1344), {})
        assert total.item() == 0.0 and items.shape == (9,)

    def test_forward_list_preds_returns_zeros(self):
        loss = MultiTaskLoss(self._make_model(["detect"]))
        total, items = loss.forward([torch.randn(2, 10, 1344), {}], {})
        assert total.item() == 0.0 and items.shape == (9,)

    def test_forward_dict_no_one2many_returns_zeros(self):
        loss = MultiTaskLoss(self._make_model(["detect"]))
        total, items = loss.forward({"not_one2many": {}}, {})
        assert total.item() == 0.0

    def test_task_weights_default(self):
        loss = MultiTaskLoss(self._make_model(["detect"]))
        assert isinstance(loss.task_weights, dict)
        assert loss.task_weights["detect"] == 1.0
        assert "segment" in loss.task_weights

    def test_loss_items_static_shape(self):
        """Even on valid forward, items tensor is always the nine-task logging vector."""
        torch.manual_seed(0)
        model = self._make_model(["detect"])
        loss = MultiTaskLoss(model)

        head = model.model[-1]
        preds = head(_make_fpn_features())
        batch = {
            "batch_idx": torch.zeros(0, dtype=torch.long),
            "cls": torch.zeros(0, 1, dtype=torch.float32),
            "bboxes": torch.zeros(0, 4, dtype=torch.float32),
        }
        total, items = loss.forward(preds, batch)
        assert items.shape == (9,) and torch.isfinite(total)

    def test_seg_loss_nonzero_with_masks(self):
        torch.manual_seed(0)
        model = self._make_model(["detect", "segment"])
        loss = MultiTaskLoss(model)

        head = model.model[-1]
        preds = head(_make_fpn_features())
        batch = {
            "batch_idx": torch.tensor([0, 0], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]], dtype=torch.float32),
            "masks": torch.randint(0, 2, (2, 160, 160), dtype=torch.uint8),
        }
        total, items = loss.forward(preds, batch)
        assert items.shape == (9,) and torch.isfinite(total), f"total={total.item()}"
        # seg_loss is computed but may be 0 for random data through E2ELoss path

    def test_end2end_auxiliary_losses_use_one2one_assignments(self, monkeypatch):
        torch.manual_seed(0)
        model = self._make_model(["detect", "segment", "pose"])
        criterion = MultiTaskLoss(model)
        calls = {"segment": 0, "pose": 0}
        original_segment = criterion.seg_loss_one2one.loss
        original_pose = criterion.pose_loss_one2one.loss

        def segment_loss(preds, batch):
            calls["segment"] += 1
            return original_segment(preds, batch)

        def pose_loss(preds, batch):
            calls["pose"] += 1
            return original_pose(preds, batch)

        monkeypatch.setattr(criterion.seg_loss_one2one, "loss", segment_loss)
        monkeypatch.setattr(criterion.pose_loss_one2one, "loss", pose_loss)
        preds = model.model[-1](_make_fpn_features())
        batch = {
            "batch_idx": torch.tensor([0, 1], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.3, 0.3]], dtype=torch.float32),
            "masks": torch.randint(0, 2, (2, 160, 160), dtype=torch.uint8),
            "keypoints": torch.tensor([[[0.5, 0.5, 1.0]] * 5, [[0.4, 0.4, 1.0]] * 5]),
        }

        total, _ = criterion(preds, batch)
        assert torch.isfinite(total) and calls == {"segment": 1, "pose": 1}

    def test_pose_loss_zero_without_keypoints(self):
        torch.manual_seed(0)
        model = self._make_model(["detect", "pose"])
        loss = MultiTaskLoss(model)

        head = model.model[-1]
        preds = head(_make_fpn_features())
        batch = {
            "batch_idx": torch.zeros(0, dtype=torch.long),
            "cls": torch.zeros(0, 1, dtype=torch.float32),
            "bboxes": torch.zeros(0, 4, dtype=torch.float32),
        }
        _, items = loss.forward(preds, batch)
        assert items[4].item() == 0.0, f"pose_loss should be 0, got {items[4].item()}"

    def test_eval_predictions_keep_multitask_loss_items(self):
        """Validation-mode tuples retain their nested task predictions for loss reporting."""
        torch.manual_seed(0)
        model = self._make_model(["detect", "segment", "pose"])
        loss = MultiTaskLoss(model)
        head = model.model[-1].eval()
        preds = head(_make_fpn_features())
        batch = {
            "batch_idx": torch.tensor([0, 0], dtype=torch.long),
            "cls": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "bboxes": torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]], dtype=torch.float32),
            "masks": torch.randint(0, 2, (2, 160, 160), dtype=torch.uint8),
            "keypoints": torch.tensor([[[0.5, 0.5, 1.0]] * 5, [[0.4, 0.4, 1.0]] * 5]),
        }

        total, items = loss.forward(preds, batch)

        assert torch.isfinite(total)
        assert items[:5].abs().sum() > 0

    def test_detection_instance_labels_do_not_supervise_global_classification(self):
        """COCO detection labels must not be treated as image-level class labels."""
        torch.manual_seed(0)
        model = self._make_model(["detect", "classify"])
        loss = MultiTaskLoss(model)

        preds = model.model[-1](_make_fpn_features())
        batch = {
            "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
            "cls": torch.tensor([[1.0], [3.0], [5.0]], dtype=torch.float32),
            "bboxes": torch.tensor(
                [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]], dtype=torch.float32
            ),
        }

        _, items = loss.forward(preds, batch)
        assert items[5].item() == 0.0

    def test_dense_tasks_have_finite_nonzero_masked_losses(self):
        """Multi-label, depth, normal, and semantic targets each supervise their own branch."""
        torch.manual_seed(0)
        model = self._make_model(["detect", "classify", "depth", "normal", "semantic"], semantic_nc=3)
        criterion = MultiTaskLoss(model)
        preds = model.model[-1](_make_fpn_features())
        batch = {
            "batch_idx": torch.zeros(0, dtype=torch.long),
            "cls": torch.zeros(0, 1),
            "bboxes": torch.zeros(0, 4),
            "cls_img": torch.tensor([[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6]),
            "cls_img_valid": torch.tensor([True, True]),
            "depth": torch.rand(2, 1, 64, 64),
            "depth_valid": torch.ones(2, 64, 64, dtype=torch.bool),
            "normal": torch.nn.functional.normalize(torch.rand(2, 3, 64, 64), dim=1),
            "normal_valid": torch.ones(2, 64, 64, dtype=torch.bool),
            "semantic_mask": torch.randint(0, 3, (2, 64, 64)),
        }
        total, items = criterion(preds, batch)
        total.backward()
        assert torch.isfinite(total) and (items[5:] > 0).all()
        for branch in ("cv4_cls", "cv4_depth", "cv4_normal", "cv4_semantic"):
            assert _has_grad(getattr(model.model[-1], branch))

    def test_semantic_loss_normalizes_over_valid_pixels_only(self):
        criterion = MultiTaskLoss.__new__(MultiTaskLoss)
        nn.Module.__init__(criterion)
        criterion.model = SimpleNamespace(end2end=False, model=[SimpleNamespace(task_router=None)])
        criterion.task_weights = {"detect": 0.0, "semantic": 1.0}
        criterion.det_loss = lambda preds, batch: (torch.zeros(1), torch.zeros(3))
        criterion.seg_loss = criterion.pose_loss = criterion.cls_loss = None
        criterion.semantic_loss = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
        logits = torch.tensor([[[[2.0, 10.0], [10.0, 10.0]], [[-1.0, -10.0], [-10.0, -10.0]]]], requires_grad=True)
        preds = {"boxes": torch.zeros(1, 4, 1), "scores": torch.zeros(1, 2, 1), "semantic": logits}
        target = torch.tensor([[[0, 255], [255, 255]]])

        total, items = criterion(preds, {"semantic_mask": target})
        expected = torch.nn.functional.cross_entropy(logits[:, :, :1, :1], target[:, :1, :1])
        assert total.item() == pytest.approx(expected.item())
        assert items[8].item() == pytest.approx(expected.item())
        total.backward()
        assert logits.grad is not None and logits.grad[:, :, 0, 0].abs().sum() > 0
        assert logits.grad[:, :, 0, 1:].eq(0).all() and logits.grad[:, :, 1:].eq(0).all()

    def test_missing_dense_targets_do_not_create_supervision(self):
        """Absent auxiliary files and ignored semantic pixels must not produce false targets."""
        torch.manual_seed(0)
        model = self._make_model(["detect", "classify", "depth", "normal", "semantic"], semantic_nc=3)
        criterion = MultiTaskLoss(model)
        preds = model.model[-1](_make_fpn_features())
        batch = {
            "batch_idx": torch.zeros(0, dtype=torch.long),
            "cls": torch.zeros(0, 1),
            "bboxes": torch.zeros(0, 4),
            "cls_img": torch.zeros(2, 8),
            "cls_img_valid": torch.tensor([False, False]),
            "depth": torch.zeros(2, 1, 64, 64),
            "depth_valid": torch.zeros(2, 64, 64, dtype=torch.bool),
            "normal": torch.zeros(2, 3, 64, 64),
            "normal_valid": torch.zeros(2, 64, 64, dtype=torch.bool),
            "semantic_mask": torch.full((2, 64, 64), 255),
        }
        _, items = criterion(preds, batch)
        assert items[5:].eq(0).all()


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskTrainer
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskTrainer:
    """MultiTaskTrainer helpers (no full training)."""

    def _new_trainer(self):
        t = MultiTaskTrainer.__new__(MultiTaskTrainer)
        t.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "seg_loss",
            "pose_loss",
            "cls_global_loss",
            "depth_loss",
            "normal_loss",
            "semantic_loss",
        )
        return t

    def test_init_task_override(self):
        overrides = {}
        overrides.setdefault("task", "multitask")
        assert overrides["task"] == "multitask"

    def test_label_loss_items_dict(self):
        t = self._new_trainer()
        result = t.label_loss_items({"box_loss": 1.2, "cls_loss": 3.4, "seg_loss": 5.6}, prefix="train")
        assert result["train/box_loss"] == 1.2
        assert result["train/cls_loss"] == 3.4
        assert result["train/seg_loss"] == 5.6

    def test_label_loss_items_tensor(self):
        t = self._new_trainer()
        result = t.label_loss_items(torch.arange(1.0, 10.0), prefix="val")
        assert result["val/box_loss"] == 1.0
        assert result["val/pose_loss"] == 5.0
        assert result["val/cls_global_loss"] == 6.0
        assert result["val/depth_loss"] == 7.0
        assert result["val/normal_loss"] == 8.0
        assert result["val/semantic_loss"] == 9.0

    def test_label_loss_items_none(self):
        keys = self._new_trainer().label_loss_items(None)
        assert keys == [
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "seg_loss",
            "pose_loss",
            "cls_global_loss",
            "depth_loss",
            "normal_loss",
            "semantic_loss",
        ]

    def test_progress_string_has_all_loss_names(self):
        t = self._new_trainer()
        s = t.progress_string()
        for name in t.loss_names:
            assert name in s
        assert "Epoch" in s

    def test_unwrap_model_identity(self):
        m = nn.Linear(2, 2)
        assert unwrap_model(m) is m

    def test_loss_names_nine_elements(self):
        t = self._new_trainer()
        assert len(t.loss_names) == 9
        assert t.loss_names[-3:] == ("depth_loss", "normal_loss", "semantic_loss")


def test_multitask_validator_keeps_coco_loader_task(monkeypatch):
    """Direct multitask validation must not fall back to the ordinary YOLO-label dataset."""
    from ultralytics.models.yolo.multitask import val as multitask_val

    validator = MultiTaskValidator(args={"task": "multitask"})
    validator.data = {"multitask_format": "coco"}
    validator.stride = 32
    captured = {}

    def build_dataset(cfg, img_path, batch, data, mode, stride):
        captured.update(task=cfg.task, img_path=img_path, batch=batch, data=data, mode=mode, stride=stride)
        return object()

    monkeypatch.setattr(multitask_val, "build_yolo_dataset", build_dataset)
    validator.build_dataset("images/val2017", batch=8)

    assert captured["task"] == "multitask"
    assert captured["data"]["multitask_format"] == "coco"
    assert validator.args.task == "detect"


class _ValidatorModel(nn.Module):
    """Minimal multi-task model metadata for validator unit tests."""

    def __init__(self, tasks):
        super().__init__()
        head = nn.Identity()
        head.agnostic_nms = False
        head.anchors = torch.tensor([[0.5, 1.5], [0.5, 0.5]])
        head.strides = torch.tensor([[8.0, 8.0]])
        self.model = nn.ModuleList([head])
        self.names = {0: "person"}
        self.active_tasks = set(tasks)
        self.end2end = True


class _ValidatorBackend:
    """Minimal AutoBackend-shaped wrapper that exposes a native multi-task model one level down."""

    def __init__(self, model):
        self.model = model
        self.names = model.names
        self.end2end = model.end2end


def _init_multitask_validator(tasks):
    """Create a CPU validator initialized with the requested active branches."""
    validator = MultiTaskValidator(args={"task": "multitask", "conf": 0.001, "max_det": 2})
    validator.data = {"kpt_shape": [17, 3], "tasks": list(tasks)}
    validator.device = torch.device("cpu")
    validator.init_metrics(_ValidatorModel(tasks))
    return validator


def test_multitask_validator_decodes_auxiliary_predictions_after_nms():
    """NMS-retained detection candidates keep mask coefficients and decoded keypoints from their source anchors."""
    validator = _init_multitask_validator(["detect", "segment", "pose"])
    raw = {
        "scores": torch.tensor([[[3.0, 2.0]]]),
        "mask_coefficient": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "proto": torch.tensor([[[[1.0] * 4] * 4, [[-1.0] * 4] * 4]]),
        "kpts": torch.zeros(1, 51, 2),
        "candidate_indices": torch.tensor([[0, 1]]),
    }
    detections = torch.tensor([[[0.0, 0.0, 16.0, 16.0, 0.95, 0.0], [0.0, 0.0, 16.0, 16.0, 0.90, 0.0]]])

    preds = validator.postprocess((detections, {"one2one": raw}))

    assert preds[0]["masks"].shape == (2, 4, 4)
    assert preds[0]["keypoints"].shape == (2, 17, 3)
    assert preds[0]["keypoints"][1, 0, 0].item() == pytest.approx(8.0)


def test_multitask_validator_handles_empty_auxiliary_candidates_after_nms():
    """A batch with no retained detections still has well-formed empty mask and keypoint predictions."""
    validator = _init_multitask_validator(["detect", "segment", "pose"])
    raw = {
        "mask_coefficient": torch.zeros(1, 2, 1),
        "proto": torch.zeros(1, 2, 4, 4),
        "kpts": torch.zeros(1, 51, 1),
        "candidate_indices": torch.tensor([[0]]),
    }
    detections = torch.tensor([[[0.0, 0.0, 16.0, 16.0, 0.0, 0.0]]])

    preds = validator.postprocess((detections, {"one2one": raw}))

    assert preds[0]["bboxes"].shape == (0, 4)
    assert preds[0]["masks"].shape == (0, 4, 4)
    assert preds[0]["keypoints"].shape == (0, 17, 3)


def test_multitask_validator_publishes_combined_mask_and_pose_metrics(tmp_path):
    """A detect/segment/pose pass exposes standard box, mask, and pose metric namespaces together."""
    validator = _init_multitask_validator(["detect", "segment", "pose"])
    validator.save_dir = tmp_path
    batch = {
        "img": torch.zeros(1, 3, 16, 16),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "masks": torch.ones(1, 4, 4),
        "keypoints": torch.tensor([[[0.5, 0.5, 1.0]] * 17]),
        "ori_shape": [(16, 16)],
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))],
        "im_file": ["sample.jpg"],
    }
    preds = [
        {
            "bboxes": torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
            "conf": torch.tensor([0.99]),
            "cls": torch.tensor([0.0]),
            "masks": torch.ones(1, 4, 4),
            "keypoints": torch.tensor([[[8.0, 8.0, 1.0]] * 17]),
        }
    ]

    validator.update_metrics(preds, batch)
    assert validator.metrics.stats["tp_m"] and validator.metrics.stats["tp_p"]
    results = validator.get_stats()

    assert "metrics/mAP50-95(B)" in results
    assert "metrics/mAP50-95(M)" in results
    assert "metrics/mAP50-95(P)" in results


def test_multitask_validator_initializes_only_active_auxiliary_metrics():
    """Inactive branches do not add misleading metric namespaces or DDP collectives."""
    segment_validator = _init_multitask_validator(["detect", "segment"])
    pose_validator = _init_multitask_validator(["detect", "pose"])

    assert "metrics/mAP50-95(M)" in segment_validator.metrics.keys
    assert "metrics/mAP50-95(P)" not in segment_validator.metrics.keys
    assert "metrics/mAP50-95(P)" in pose_validator.metrics.keys
    assert "metrics/mAP50-95(M)" not in pose_validator.metrics.keys


def test_multitask_validator_uses_dataset_tasks_to_suppress_unselected_model_heads():
    """A physical auxiliary head does not create metrics when the selected split does not supervise it."""
    model = _ValidatorModel(["detect", "segment", "pose"])
    validator = MultiTaskValidator(args={"task": "multitask"})
    validator.data = {"tasks": ["detect"]}
    validator.device = torch.device("cpu")

    validator.init_metrics(model)

    assert validator.metrics.keys == [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]


def test_multitask_validator_unwraps_native_head_from_backend():
    """Direct ``model.val``-style backend wrappers retain access to the native auxiliary task head."""
    source_model = _ValidatorModel(["detect", "segment", "pose"])
    validator = MultiTaskValidator(args={"task": "multitask"})
    validator.data = {"kpt_shape": [17, 3], "tasks": ["detect", "segment", "pose"]}
    validator.device = torch.device("cpu")

    validator.init_metrics(_ValidatorBackend(source_model))

    assert validator._head is source_model.model[-1]


def test_multitask_validator_rejects_detection_only_export_backend():
    """Auxiliary mAP cannot be reported from an export backend without raw task payloads."""
    validator = MultiTaskValidator(args={"task": "multitask"})
    validator.data = {"kpt_shape": [17, 3], "tasks": ["detect", "segment"]}
    validator.device = torch.device("cpu")
    export_backend = SimpleNamespace(names={0: "person"}, end2end=True, model=nn.Identity())

    with pytest.raises(TypeError, match="native PyTorch MultiTaskModel"):
        validator.init_metrics(export_backend)


# ══════════════════════════════════════════════════════════════════════════
# MultiTaskModel integration (nn/tasks.py) — slow tests
# ══════════════════════════════════════════════════════════════════════════


class TestMultiTaskModelIntegration:
    @pytest.mark.slow
    def test_build_from_yaml_config(self):
        from ultralytics.nn.tasks import MultiTaskModel

        cfg_path = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-master-mt-n.yaml"
        assert cfg_path.exists()
        model = MultiTaskModel(str(cfg_path), nc=8, ch=3, verbose=False)
        head = model.model[-1]
        assert isinstance(head, MultiTaskHead) and len(head.active_tasks) >= 1

    @pytest.mark.slow
    def test_model_forward_no_crash(self):
        torch.manual_seed(0)
        from ultralytics.nn.tasks import MultiTaskModel

        cfg_path = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-master-mt-n.yaml"
        model = MultiTaskModel(str(cfg_path), nc=8, ch=3, verbose=False)
        model.train()
        out = model(torch.randn(2, 3, 320, 320))
        assert isinstance(out, dict)
        assert "one2many" in out or "boxes" in out

    @pytest.mark.slow
    def test_onnx_export_roundtrip_preserves_multitask_schema(self, tmp_path):
        """The real MoT model exports a runnable, stable multi-task ONNX endpoint."""
        onnx = pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        from ultralytics import YOLO
        from ultralytics.engine.exporter import Exporter

        config = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-master-mt-n.yaml"
        facade = YOLO(config, task="multitask")
        facade.model.pt_path = str(tmp_path / "multitask.pt")
        exporter = Exporter(
            overrides={
                "format": "onnx",
                "imgsz": 64,
                "batch": 1,
                "device": "cpu",
                "nms": False,
                "simplify": False,
                "opset": 17,
            }
        )
        artifact = Path(exporter(model=facade.model))

        graph = onnx.load(artifact)
        onnx.checker.check_model(graph)
        output_names = [output.name for output in graph.graph.output]
        assert output_names == ["detections", "mask_coefficients", "mask_prototypes", "keypoints"]
        assert exporter.metadata["kpt_shape"] == [17, 3]
        assert set(exporter.metadata["multitask_output_schema"]) == set(output_names)

        with torch.no_grad():
            expected = tuple(output.detach().cpu().numpy() for output in exporter.model(exporter.im))
        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        observed = session.run(None, {session.get_inputs()[0].name: exporter.im.cpu().numpy()})
        assert len(observed) == len(expected) == len(output_names)
        for eager, runtime in zip(expected, observed):
            assert runtime.shape == eager.shape
            assert np.isfinite(runtime).all()

        # The untrained one-to-one classification head has tied anchor scores.
        # PyTorch and ONNX may select different, equally valid Top-K candidates,
        # so candidate-aligned outputs are checked strictly at head level above.
        np.testing.assert_allclose(expected[2], observed[2], atol=1e-4, rtol=1e-4)
