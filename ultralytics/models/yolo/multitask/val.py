# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation support for unified multi-task YOLO models."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import DetMetrics, Metric, OKS_SIGMA, ap_per_class, kpt_iou, mask_iou


class MultiTaskMetrics(DetMetrics):
    """Accumulate shared box statistics plus only the enabled mask and pose task metrics."""

    def __init__(self, names: dict[int, str] | None = None, tasks: frozenset[str] = frozenset({"detect"})) -> None:
        """Initialize box metrics and optional instance-mask and pose metric state."""
        super().__init__(names)
        self.tasks = tasks
        if "segment" in tasks:
            self.seg = Metric()
            self.stats["tp_m"] = []
        if "pose" in tasks:
            self.pose = Metric()
            self.stats["tp_p"] = []

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Append shared and enabled task evidence from one image."""
        for key in self.stats:
            self.stats[key].append(stat[key])
        self.box.update_image_metrics(stat["tp"], stat["target_cls"], stat["pred_cls"], stat["im_name"])
        if "segment" in self.tasks:
            self.seg.update_image_metrics(stat["tp_m"], stat["target_cls"], stat["pred_cls"], stat["im_name"])
        if "pose" in self.tasks:
            self.pose.update_image_metrics(stat["tp_p"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def clear_image_metrics(self) -> None:
        """Clear per-image evidence for the shared and enabled task metrics."""
        super().clear_image_metrics()
        if "segment" in self.tasks:
            self.seg.clear_image_metrics()
        if "pose" in self.tasks:
            self.pose.clear_image_metrics()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Compute box AP and each enabled auxiliary task AP from the same candidate statistics."""
        stats = super().process(save_dir, plot, on_plot=on_plot)
        for task, metric, stat_key, prefix in (
            ("segment", getattr(self, "seg", None), "tp_m", "Mask"),
            ("pose", getattr(self, "pose", None), "tp_p", "Pose"),
        ):
            if task not in self.tasks:
                continue
            results = ap_per_class(
                stats[stat_key],
                stats["conf"],
                stats["pred_cls"],
                stats["target_cls"],
                plot=plot,
                save_dir=save_dir,
                names=self.names,
                on_plot=on_plot,
                prefix=prefix,
            )[2:]
            metric.nc = len(self.names)
            metric.update(results)
        return stats

    @property
    def keys(self) -> list[str]:
        """Return only metric namespaces backed by active task supervision."""
        keys = [*DetMetrics.keys.fget(self)]
        if "segment" in self.tasks:
            keys.extend(["metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"])
        if "pose" in self.tasks:
            keys.extend(["metrics/precision(P)", "metrics/recall(P)", "metrics/mAP50(P)", "metrics/mAP50-95(P)"])
        return keys

    def mean_results(self) -> list[float]:
        """Return box results followed by each active auxiliary task result vector."""
        results = DetMetrics.mean_results(self)
        if "segment" in self.tasks:
            results.extend(self.seg.mean_results())
        if "pose" in self.tasks:
            results.extend(self.pose.mean_results())
        return results

    def class_result(self, i: int) -> list[float]:
        """Return the class result vector for every enabled metric namespace."""
        results = list(DetMetrics.class_result(self, i))
        if "segment" in self.tasks:
            results.extend(self.seg.class_result(i))
        if "pose" in self.tasks:
            results.extend(self.pose.class_result(i))
        return results

    @property
    def fitness(self) -> float:
        """Average the comparable mAP fitness values of the tasks evaluated in this run."""
        values = [self.box.fitness()]
        if "segment" in self.tasks:
            values.append(self.seg.fitness())
        if "pose" in self.tasks:
            values.append(self.pose.fitness())
        return float(np.mean(values))


class MultiTaskValidator(DetectionValidator):
    """Validate shared detection plus active instance-mask and pose branches against aligned COCO targets."""

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move multi-task targets to the validation device and normalize instance-mask dtype when needed."""
        batch = super().preprocess(batch)
        if "segment" in self.active_tasks:
            batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize box metrics and only the auxiliary task metrics backed by the validated model."""
        super().init_metrics(model)
        source_model, self._head = self._get_multitask_source(model)
        model_tasks = set(getattr(source_model, "active_tasks", ()))
        declared_tasks = set(self.data.get("tasks", ()))
        metric_tasks = {"detect", "segment", "pose"}
        if declared_tasks:
            requested_tasks = declared_tasks & metric_tasks
            unavailable_tasks = requested_tasks.difference(model_tasks)
            if unavailable_tasks and self._head is not None:
                raise ValueError(
                    "The multi-task dataset requests metric branches not enabled by the selected model: "
                    f"{sorted(unavailable_tasks)}. Model branches: {sorted(model_tasks)}."
                )
            enabled_tasks = requested_tasks.intersection(model_tasks)
        else:
            requested_tasks = model_tasks & metric_tasks
            enabled_tasks = requested_tasks
        self.active_tasks = frozenset({"detect", *enabled_tasks})
        if requested_tasks.difference({"detect"}) and self._head is None:
            raise TypeError(
                "Multi-task mask and pose metrics require the native PyTorch MultiTaskModel head. "
                "This validation backend exposes detection outputs only, so it cannot produce trustworthy auxiliary metrics."
            )
        self.metrics = MultiTaskMetrics(model.names, self.active_tasks)
        self.metrics.clear_stats()
        self.metrics.clear_image_metrics()
        self.mask_process = None
        self.kpt_shape = None
        self.sigma = None

        if "segment" in self.active_tasks:
            self.mask_process = (
                ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
            )
        if "pose" in self.active_tasks:
            self.kpt_shape = self.data["kpt_shape"]
            nkpt = self.kpt_shape[0]
            if sigmas := self.data.get("kpt_oks_sigmas"):
                self.sigma = np.array(sigmas, dtype=np.float32).flatten()
                if len(self.sigma) != nkpt or not np.all(self.sigma > 0):
                    raise ValueError(f"'kpt_oks_sigmas' must be {nkpt} positive values, got {sigmas}")
            else:
                self.sigma = OKS_SIGMA if self.kpt_shape == [17, 3] else np.ones(nkpt) / nkpt

    @staticmethod
    def _get_multitask_source(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module | None]:
        """Return the native model and head from direct validation or the ``AutoBackend`` PyTorch wrapper."""
        source_model = model
        layers = getattr(source_model, "model", None)
        if isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
            return source_model, layers[-1]

        # AutoBackend delegates ``model`` to its backend, which in turn holds the native PyTorch model.
        source_model = layers
        layers = getattr(source_model, "model", None)
        if isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
            return source_model, layers[-1]
        return model, None

    def get_desc(self) -> str:
        """Return a validation table header that matches enabled metric namespaces."""
        active_tasks = getattr(
            self,
            "active_tasks",
            frozenset(({"detect"} | set(getattr(self, "data", {}).get("tasks", ())) & {"detect", "segment", "pose"})),
        )
        labels = ["Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)"]
        if "segment" in active_tasks:
            labels.extend(["Mask(P", "R", "mAP50", "mAP50-95)"])
        if "pose" in active_tasks:
            labels.extend(["Pose(P", "R", "mAP50", "mAP50-95)"])
        return ("%22s" + "%11s" * len(labels)) % ("Class", *labels)

    def postprocess(self, preds: tuple[torch.Tensor, dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
        """Apply detection filtering and decode enabled auxiliary outputs for the retained source anchors."""
        if not isinstance(preds, (tuple, list)) or len(preds) != 2 or not isinstance(preds[1], dict):
            return super().postprocess(preds)

        detections, raw_predictions = preds
        task_predictions = raw_predictions.get("one2one", raw_predictions)
        candidate_indices = task_predictions.get("candidate_indices")
        if candidate_indices is None:
            if self.active_tasks.difference({"detect"}):
                raise RuntimeError(
                    "Multi-task validation did not receive native auxiliary anchor metadata. "
                    "Use the native PyTorch MultiTaskModel rather than an exported detection-only backend."
                )
            return super().postprocess(preds)

        outputs, kept_indices = nms.non_max_suppression(
            detections,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            return_idxs=True,
        )
        processed = []
        for image_index, (output, kept) in enumerate(zip(outputs, kept_indices)):
            prediction = {"bboxes": output[:, :4], "conf": output[:, 4], "cls": output[:, 5], "extra": output[:, 6:]}
            source_indices = candidate_indices[image_index, kept.reshape(-1).long()]
            if "segment" in self.active_tasks:
                proto = task_predictions["proto"]
                proto = proto[0] if isinstance(proto, tuple) else proto
                coefficients = task_predictions["mask_coefficient"][image_index, :, source_indices].transpose(0, 1)
                prediction["masks"] = self.mask_process(
                    proto[image_index], coefficients, prediction["bboxes"], shape=[4 * size for size in proto.shape[2:]]
                )
            if "pose" in self.active_tasks:
                prediction["keypoints"] = self._decode_keypoints(
                    task_predictions["kpts"][image_index, :, source_indices].transpose(0, 1), source_indices
                )
            processed.append(prediction)
        return processed

    def _decode_keypoints(self, keypoints: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
        """Decode raw keypoint offsets for the dense anchors retained by end-to-end candidate selection."""
        keypoints = keypoints.reshape(-1, *self.kpt_shape).clone()
        anchors = self._head.anchors[:, source_indices]
        strides = self._head.strides[:, source_indices]
        if self.kpt_shape[1] == 3:
            keypoints[..., 2] = keypoints[..., 2].sigmoid()
        keypoints[..., 0] = (keypoints[..., 0] * 2.0 + (anchors[0].unsqueeze(1) - 0.5)) * strides[0].unsqueeze(1)
        keypoints[..., 1] = (keypoints[..., 1] * 2.0 + (anchors[1].unsqueeze(1) - 0.5)) * strides[0].unsqueeze(1)
        return keypoints

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare aligned detection labels plus only the enabled mask and pose targets for one image."""
        prepared_batch = super()._prepare_batch(si, batch)
        label_count = prepared_batch["cls"].shape[0]
        if "segment" in self.active_tasks:
            if self.args.overlap_mask:
                masks = batch["masks"][si]
                indices = torch.arange(1, label_count + 1, device=masks.device).view(label_count, 1, 1)
                masks = (masks == indices).float()
            else:
                masks = batch["masks"][batch["batch_idx"] == si]
            if label_count:
                mask_size = [
                    size if self.mask_process is ops.process_mask_native else size // 4
                    for size in prepared_batch["imgsz"]
                ]
                if masks.shape[1:] != mask_size:
                    masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0].gt_(0.5)
            prepared_batch["masks"] = masks
        if "pose" in self.active_tasks:
            keypoints = batch["keypoints"][batch["batch_idx"] == si].clone()
            height, width = prepared_batch["imgsz"]
            keypoints[..., 0] *= width
            keypoints[..., 1] *= height
            prepared_batch["keypoints"] = keypoints
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Match retained detections against box, mask, and pose targets using established task definitions."""
        result = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        no_targets_or_predictions = gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0
        if "segment" in self.active_tasks:
            if no_targets_or_predictions:
                result["tp_m"] = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
            else:
                iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())
                result["tp_m"] = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        if "pose" in self.active_tasks:
            if no_targets_or_predictions:
                result["tp_p"] = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
            else:
                area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
                iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
                result["tp_p"] = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        return result

    def gather_stats(self) -> None:
        """Gather shared detection statistics and every enabled auxiliary image metric in DDP-safe order."""
        super().gather_stats()
        if "segment" in self.active_tasks:
            self._gather_image_metrics(self.metrics.seg)
        if "pose" in self.active_tasks:
            self._gather_image_metrics(self.metrics.pose)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the configured multi-task dataset without changing detection post-processing semantics."""
        dataset_args = copy(self.args)
        dataset_args.task = "multitask"
        return build_yolo_dataset(dataset_args, img_path, batch, self.data, mode=mode, stride=self.stride)
