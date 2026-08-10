"""Unified Multi-Task Detection Head.

Extends the YOLO Detect head to support detection, instance segmentation,
pose estimation, image multi-label classification, dense depth, surface
normals, semantic segmentation, and oriented bounding box detection.

MOT-inspired design:
- Two-stage task assignment: primary (high-conf) + secondary (ambiguous)
- Cross-task feature sharing through a shared bottleneck
- Task-aware dynamic weighting via learned task importance scores
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.block import Proto26
from ultralytics.nn.modules.conv import Conv


DEFAULT_MULTITASK_TASKS = ("detect", "segment", "pose", "classify", "depth", "normal", "semantic")
COMPATIBILITY_ONLY_TASKS = frozenset({"obb"})


class MultiTaskHead(Detect):
    """Unified multi-task head with sparse and dense prediction branches.

    Each task branch is independently configurable. The head can be used with
    or without a TaskRouter — without one it defaults to parallel prediction
    for all configured tasks.

    Args:
        nc: Number of classes shared by detection and related branches.
        ch: Tuple of input channel sizes from FPN (e.g. (256, 512, 1024)).
        tasks: List of built task names. Trainable branches are "detect", "segment", "pose", "classify", "depth",
            "normal", and "semantic". "obb" remains construction-compatible for legacy checkpoints but is not a
            supported unified multi-task training branch.
        nm: Number of mask coefficients (segment).
        npr: Number of mask prototypes (segment).
        kpt_shape: (num_keypoints, dims) for pose estimation.
        depth_bins: Number of depth discretization bins.
        reg_max: DFL channels.
        end2end: Use one-to-one + one-to-many heads.
        use_task_router: Integrate TaskRouter for adaptive task routing.
        task_router_dim: TaskRouter hidden dimension.
    """

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (),
        tasks: Optional[list[str]] = None,
        nm: int = 32,
        npr: int = 256,
        kpt_shape: tuple = (17, 3),
        depth_bins: int = 80,
        semantic_nc: int = 0,
        reg_max: int = 16,
        end2end: bool = True,
        use_task_router: bool = False,
        task_router_dim: Optional[int] = None,
    ):
        # Default to branches with a complete unified training contract.
        if tasks is None:
            tasks = list(DEFAULT_MULTITASK_TASKS)
        self._built_tasks = set(tasks)
        self._built_tasks.add("detect")
        self._active_tasks = set(self._built_tasks)
        self._use_task_router = use_task_router

        # Initialize Detect base (detection always present) and keep its branch lifecycle explicit.
        super().__init__(nc, reg_max, end2end, ch)
        self.end2end = end2end

        # ── Task-specific heads ──────────────────────────────────────────
        self._build_segment_head(ch, nm, npr)
        self._build_pose_head(ch, kpt_shape)
        self._build_classify_head(ch)
        self._build_dense_heads(ch, depth_bins, semantic_nc)
        self._build_obb_head(ch)

        # ── TaskRouter (optional) ────────────────────────────────────────
        if use_task_router:
            from .router import TaskRouter

            router_dim = task_router_dim or ch[0]
            self.task_router = TaskRouter(
                dim=router_dim,
                num_tasks=len(self._active_tasks),
                top_k=2,
                shared_expert_ratio=0.2,
            )
            self._task_router_names = sorted(self._built_tasks)
            self.task_router_input_proj = (
                nn.Identity() if int(ch[0]) == int(router_dim) else nn.Conv2d(ch[0], router_dim, 1, bias=False)
            )
            # Per-scale projectors: shared_feats → each FPN scale channel
            shared_c = int(router_dim * 0.2)
            self.task_router_proj = nn.ModuleList([nn.Conv2d(shared_c, c, 1, bias=False) for c in ch])
            self.task_router_task_proj = nn.ModuleList(
                nn.ModuleList(nn.Conv2d(router_dim, c, 1, bias=False) for c in ch) for _ in self._task_router_names
            )
        else:
            self.task_router = None
            self.task_router_input_proj = None
            self.task_router_proj = None
            self.task_router_task_proj = None

        # ── Task importance scores (learnable) ───────────────────────────
        num_active = len(self._active_tasks)
        self.task_importance = nn.Parameter(torch.zeros(num_active))

    # ── Task builders ────────────────────────────────────────────────────
    def _build_segment_head(self, ch, nm, npr):
        if "segment" not in self._active_tasks:
            self.nm = 0
            self.proto = None
            self.cv4_seg = None
            return
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, npr, nm, self.nc)
        c4 = max(ch[0] // 4, nm)
        self.cv4_seg = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, nm, 1)) for x in ch)

    def _build_pose_head(self, ch, kpt_shape):
        if "pose" not in self._active_tasks:
            self.kpt_shape = (0, 0)
            self.nk = 0
            self.cv4_pose = None
            return
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4_pose = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch
        )

    def _build_classify_head(self, ch):
        if "classify" not in self._active_tasks:
            self.cv4_cls = None
            self.global_pool = None
            return
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        cls_in = ch[-1]  # deepest feature
        self.cv4_cls = nn.Sequential(
            Conv(cls_in, cls_in // 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cls_in // 2, self.nc),
        )
        # Classification head doesn't use end2end

    def _build_dense_heads(self, ch, depth_bins, semantic_nc):
        """Build P3-aligned dense depth, normal, and semantic prediction branches."""
        self.depth_bins = depth_bins
        dense_channels = max(ch[0] // 2, 32)
        self.cv4_depth = (
            nn.Sequential(
                Conv(ch[0], dense_channels, 3), Conv(dense_channels, dense_channels, 3), nn.Conv2d(dense_channels, 1, 1)
            )
            if "depth" in self._active_tasks
            else None
        )
        self.cv4_normal = (
            nn.Sequential(
                Conv(ch[0], dense_channels, 3), Conv(dense_channels, dense_channels, 3), nn.Conv2d(dense_channels, 3, 1)
            )
            if "normal" in self._active_tasks
            else None
        )
        self.semantic_nc = semantic_nc or self.nc
        self.cv4_semantic = (
            nn.Sequential(
                Conv(ch[0], dense_channels, 3),
                Conv(dense_channels, dense_channels, 3),
                nn.Conv2d(dense_channels, self.semantic_nc, 1),
            )
            if "semantic" in self._active_tasks
            else None
        )

    def _build_obb_head(self, ch):
        if "obb" not in self._active_tasks:
            self.ne = 0
            self.cv4_obb = None
            return
        self.ne = 1  # angle
        c4 = max(ch[0] // 4, self.ne)
        self.cv4_obb = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch
        )

    # ── Property helpers ─────────────────────────────────────────────────
    @property
    def active_tasks(self) -> list[str]:
        return sorted(self._active_tasks)

    def has_task(self, task: str) -> bool:
        return task in self._active_tasks

    @property
    def export_output_names(self) -> list[str]:
        """Return the stable ordered names emitted by the multi-task export path."""
        names = ["detections"]
        if self.has_task("segment"):
            names.extend(("mask_coefficients", "mask_prototypes"))
        if self.has_task("pose"):
            names.append("keypoints")
        if self.has_task("classify"):
            names.append("class_logits")
        if self.has_task("depth"):
            names.append("depth")
        if self.has_task("normal"):
            names.append("normal")
        if self.has_task("semantic"):
            names.append("semantic")
        if self.has_task("obb"):
            names.append("angles")
        return names

    @property
    def export_output_schema(self) -> dict[str, str]:
        """Describe the tensor layout of every ordered multi-task export output."""
        schema = {"detections": "[batch, max_det, 6] (xyxy, confidence, class)"}
        if self.has_task("segment"):
            schema.update(
                {
                    "mask_coefficients": "[batch, max_det, nm], aligned with detections",
                    "mask_prototypes": "[batch, nm, mask_height, mask_width]",
                }
            )
        if self.has_task("pose"):
            schema["keypoints"] = "[batch, max_det, num_keypoints * dimensions], aligned with detections"
        if self.has_task("classify"):
            schema["class_logits"] = "[batch, nc]"
        if self.has_task("depth"):
            schema["depth"] = "[batch, 1, height, width]"
        if self.has_task("normal"):
            schema["normal"] = "[batch, 3, height, width]"
        if self.has_task("semantic"):
            schema["semantic"] = "[batch, semantic_nc, height, width]"
        if self.has_task("obb"):
            schema["angles"] = "[batch, max_det, ne], aligned with detections"
        return schema

    def _route_task_features(self, x: list[torch.Tensor]) -> tuple[dict[str, list[torch.Tensor]], dict | None]:
        """Return per-task routed features and diagnostics."""

        if self.task_router is None or self.task_router_task_proj is None:
            return {task: x for task in self._active_tasks}, None
        router_input = self.task_router_input_proj(x[0])
        task_feats, shared_feats, routing_stats = self.task_router(router_input)
        task_map = {name: index for index, name in enumerate(self._task_router_names)}
        routed = {}
        for task in self._active_tasks:
            index = task_map.get(task)
            if index is None or index >= task_feats.shape[1]:
                raise RuntimeError(f"TaskRouter has no expert feature for active task {task!r}")
            importance = F.softplus(self.task_importance[index])
            importance = importance / F.softplus(self.task_importance.new_zeros(()))
            features = []
            for i, xi in enumerate(x):
                routed_task = self.task_router_task_proj[index][i](
                    F.interpolate(task_feats[:, index], size=xi.shape[2:], mode="bilinear", align_corners=False)
                )
                if shared_feats is not None:
                    routed_task = routed_task + self.task_router_proj[i](
                        F.interpolate(shared_feats, size=xi.shape[2:], mode="bilinear", align_corners=False)
                    )
                features.append(xi + importance.to(dtype=xi.dtype) * routed_task)
            routed[task] = features
        return routed, routing_stats

    def set_active_tasks(self, tasks: list[str] | set[str] | tuple[str, ...]) -> None:
        """Select dataset-supervised branches without creating missing heads at runtime.

        A model YAML defines the physical branches and a dataset YAML only selects
        a subset of those branches. Allowing the latter to activate a branch that
        was not built silently trains a different task set than the user asked for.
        """
        tasks = set(tasks)
        if "detect" not in tasks:
            tasks.add("detect")
        compatibility_only = tasks.intersection(COMPATIBILITY_ONLY_TASKS)
        if compatibility_only:
            raise ValueError(
                "Multi-task 'obb' is not trainable: use a dedicated OBB model or remove 'obb' from data.tasks. "
                "The compatibility-only angle head is retained solely for loading legacy checkpoints."
            )
        unknown = tasks.difference(DEFAULT_MULTITASK_TASKS)
        if unknown:
            raise ValueError(f"Unsupported multi-task branches: {sorted(unknown)}")
        unavailable = tasks.difference(self._built_tasks)
        if unavailable:
            raise ValueError(
                "Dataset requests multi-task branches that this model YAML did not build: "
                f"{sorted(unavailable)}. Model branches: {sorted(self._built_tasks)}. "
                "Use a model YAML with matching 'tasks'."
            )
        self._active_tasks = tasks

    @property
    def one2many(self):
        d = dict(box_head=self.cv2, cls_head=self.cv3)
        if self.has_task("segment"):
            d["mask_head"] = self.cv4_seg
        if self.has_task("pose"):
            d["pose_head"] = self.cv4_pose
        if self.has_task("depth"):
            d["depth_head"] = self.cv4_depth
        if self.has_task("normal"):
            d["normal_head"] = self.cv4_normal
        if self.has_task("semantic"):
            d["semantic_head"] = self.cv4_semantic
        if self.has_task("obb"):
            d["obb_head"] = self.cv4_obb
        return d

    @property
    def one2one(self):
        """Return the independently trained one-to-one detection heads."""
        if not hasattr(self, "one2one_cv2"):
            return {}
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    # ── Core forward ─────────────────────────────────────────────────────
    def forward_head(self, x: list[torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Forward through task-specific heads.

        Args:
            x: List of feature maps from FPN [P3, P4, P5].
            **kwargs: head component overrides (box_head, cls_head, mask_head, etc.)

        Returns:
            dict with keys: boxes, scores, feats, and optionally:
                mask_coefficient, proto, kpts, depth, normal, semantic, angle, cls_logits
        """
        box_head = kwargs.get("box_head", self.cv2)
        cls_head = kwargs.get("cls_head", self.cv3)
        skip_task_router = bool(kwargs.pop("skip_task_router", False))
        skip_auxiliary_tasks = bool(kwargs.pop("skip_auxiliary_tasks", False))

        routed_features, routing_stats = (
            ({task: x for task in self._active_tasks}, None) if skip_task_router else self._route_task_features(x)
        )
        # Base detection forward uses its own routed feature branch.
        preds = super().forward_head(routed_features.get("detect", x), box_head=box_head, cls_head=cls_head)
        bs = x[0].shape[0]

        # ── TaskRouter integration ───────────────────────────────────────
        if self.task_router is not None and self.training:
            preds["routing_stats"] = routing_stats

        # ── Segmentation ─────────────────────────────────────────────────
        if self.has_task("segment") and not skip_auxiliary_tasks:
            x_mod = routed_features.get("segment", x)
            mask_head = kwargs.get("mask_head", self.cv4_seg)
            if mask_head is not None:
                preds["mask_coefficient"] = torch.cat(
                    [mask_head[i](x_mod[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
                )
                preds["proto"] = self.proto(x_mod) if self.proto is not None else None

        # ── Pose ─────────────────────────────────────────────────────────
        if self.has_task("pose") and not skip_auxiliary_tasks:
            x_mod = routed_features.get("pose", x)
            pose_head = kwargs.get("pose_head", self.cv4_pose)
            if pose_head is not None:
                preds["kpts"] = torch.cat([pose_head[i](x_mod[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)

        # ── Depth ────────────────────────────────────────────────────────
        if self.has_task("depth") and not skip_auxiliary_tasks:
            x_mod = routed_features.get("depth", x)
            depth_head = kwargs.get("depth_head", self.cv4_depth)
            if depth_head is not None:
                preds["depth"] = F.softplus(depth_head(x_mod[0]))

        if self.has_task("normal") and not skip_auxiliary_tasks:
            x_mod = routed_features.get("normal", x)
            normal_head = kwargs.get("normal_head", self.cv4_normal)
            if normal_head is not None:
                preds["normal"] = F.normalize(normal_head(x_mod[0]), dim=1, eps=1e-6)

        if self.has_task("semantic") and not skip_auxiliary_tasks:
            x_mod = routed_features.get("semantic", x)
            semantic_head = kwargs.get("semantic_head", self.cv4_semantic)
            if semantic_head is not None:
                preds["semantic"] = semantic_head(x_mod[0])

        # ── OBB compatibility output ────────────────────────────────────
        if self.has_task("obb") and not skip_auxiliary_tasks:
            x_mod = routed_features.get("obb", x)
            obb_head = kwargs.get("obb_head", self.cv4_obb)
            if obb_head is not None:
                preds["angle"] = torch.cat([obb_head[i](x_mod[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)

        # ── Classification ───────────────────────────────────────────────
        if self.has_task("classify") and self.cv4_cls is not None and not skip_auxiliary_tasks:
            x_mod = routed_features.get("classify", x)
            preds["cls_logits"] = self.cv4_cls(x_mod[-1])  # global image-level classification

        return preds

    def forward(self, x: list[torch.Tensor]):
        """Unified multi-task forward pass.

        Training: returns dict with one2many + one2one predictions.
        Inference: returns (detections, task_outputs) tuple.
        """
        # Detection forward (base)
        preds = self.forward_head(x, **self.one2many)

        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one_kwargs = {k: v for k, v in self.one2one.items() if v is not None}
            # The auxiliary one-to-one branch is detached from the backbone;
            # routing it again would nevertheless update TaskRouter weights
            # and replace the primary branch diagnostics.
            one2one_kwargs["skip_task_router"] = True
            one2one_kwargs["skip_auxiliary_tasks"] = True
            one2one = self.forward_head(x_detach, **one2one_kwargs)
            # Auxiliary heads are supervised on the routed one-to-many features. Reuse those trained dense predictions
            # for candidate-aligned validation/export instead of maintaining unsupervised one-to-one copies.
            for key in ("mask_coefficient", "proto", "kpts", "angle"):
                if key in preds:
                    one2one[key] = preds[key]
            preds = {"one2many": preds, "one2one": one2one}

        if self.training:
            return preds

        # Inference: decode detections (standard format)
        inference_predictions = preds["one2one"] if self.end2end else preds
        y = self._inference(inference_predictions)
        if self.end2end:
            if self.export:
                y, candidate_indices = self._postprocess_with_indices(y.permute(0, 2, 1))
                return self._export_outputs(y, inference_predictions, preds.get("one2many"), candidate_indices)
            elif self.has_task("segment") or self.has_task("pose"):
                y, candidate_indices = self._postprocess_with_indices(y.permute(0, 2, 1))
                # Auxiliary predictions remain dense-anchor aligned. Preserve every end-to-end candidate's anchor index
                # so validation can recover mask coefficients and keypoint offsets after confidence filtering.
                preds["one2one"]["candidate_indices"] = candidate_indices
            else:
                y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _export_outputs(
        self,
        detections: torch.Tensor,
        anchor_predictions: dict[str, torch.Tensor],
        dense_predictions: dict[str, torch.Tensor] | None,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Build ordered export tensors while preserving auxiliary alignment with retained detections."""
        outputs = [detections]
        if self.has_task("segment"):
            coefficients = self._gather_anchor_predictions(anchor_predictions["mask_coefficient"], candidate_indices)
            outputs.extend((coefficients, anchor_predictions["proto"]))
        if self.has_task("pose"):
            raw_keypoints = self._gather_anchor_predictions(anchor_predictions["kpts"], candidate_indices)
            outputs.append(self._decode_export_keypoints(raw_keypoints, candidate_indices))

        dense_predictions = dense_predictions or {}
        for task, key in (
            ("classify", "cls_logits"),
            ("depth", "depth"),
            ("normal", "normal"),
            ("semantic", "semantic"),
        ):
            if self.has_task(task):
                outputs.append(dense_predictions[key])
        if self.has_task("obb"):
            outputs.append(self._gather_anchor_predictions(anchor_predictions["angle"], candidate_indices))
        # Keep the established detection-only tensor endpoint. Multi-task
        # exports intentionally return a named tuple of tensors so ONNX and
        # TorchScript consumers can bind stable output names.
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @staticmethod
    def _gather_anchor_predictions(predictions: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        """Gather [B, C, anchors] predictions into detection-aligned [B, max_det, C] tensors."""
        gather_indices = candidate_indices.unsqueeze(1).expand(-1, predictions.shape[1], -1)
        return predictions.gather(2, gather_indices).transpose(1, 2)

    def _decode_export_keypoints(self, keypoints: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        """Decode selected keypoints using the same anchor geometry used for exported detections."""
        batch_size, max_det, _ = keypoints.shape
        decoded = keypoints.reshape(batch_size, max_det, *self.kpt_shape).clone()
        anchors = self.anchors.transpose(0, 1)[candidate_indices].unsqueeze(2)
        strides = self.strides.transpose(0, 1)[candidate_indices].unsqueeze(2)
        if self.kpt_shape[1] == 3:
            decoded[..., 2] = decoded[..., 2].sigmoid()
        decoded[..., :2] = (decoded[..., :2] * 2.0 + (anchors - 0.5)) * strides
        return decoded.reshape(batch_size, max_det, self.nk)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode boxes + scores. In eval mode, return standard Detection format (no extra dims)."""
        preds = super()._inference(x)

        # Training-only: append task-specific predictions
        if not self.training:
            return preds

        extra_parts = []
        if "mask_coefficient" in x:
            extra_parts.append(x["mask_coefficient"])
        if "kpts" in x:
            extra_parts.append(x["kpts"])
        if "angle" in x:
            extra_parts.append(x["angle"])

        if extra_parts:
            return torch.cat([preds] + extra_parts, dim=1)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Standard end2end postprocess: delegate to Detect."""
        return super().postprocess(preds)

    def _postprocess_with_indices(self, preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return end-to-end detections and their source dense-anchor indices."""
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, indices = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=indices.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1), indices.squeeze(-1)

    def bias_init(self):
        """Initialize biases for all head branches."""
        super().bias_init()
        # Additional init for task branches can be added here

    def fuse(self) -> None:
        """Remove one-to-many detection heads while retaining every supervised auxiliary branch."""
        self.cv2 = self.cv3 = None
