# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import torch
from torch import nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        topk2 (int): Secondary topk value for additional filtering.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        stride (list): List of stride values for different feature levels.
        stride_val (int): The stride value used for select_candidates_in_gts.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2=None,
        stal_enabled: bool = False,
        stal_stats: bool = False,
        stal_candidate_mode: str = "fixed",
        stal_area_threshold: float = 0.01,
        stal_small_topk: int | None = None,
        stal_small_topk_min: int | None = None,
        stal_min_base_candidates: int = 0,
        stal_max_extra_candidates: int = 0,
        stal_min_candidate_guarantee: bool = False,
        stal_crowding_mode: str = "none",
        stal_crowded_relaxation: float = 0.0,
        stal_relaxation: float = 8.0,
        stal_relaxation_scale_mode: str = "constant",
        stal_relaxation_min: float = 0.0,
        stal_min_size_stride_ratio: float = 0.0,
        stal_warmup_epochs: float = 0.0,
        stal_zero_positive_rescue: bool = False,
        stal_rescue_score_floor: float = 0.0,
        stal_rescue_floor_decay_epochs: float = 0.0,
        stal_nwd_weight: float = 0.0,
        stal_nwd_constant: float = 12.8,
        stal_nwd_target_mode: str = "match",
        stal_nwd_target_weight: float = 0.0,
        stal_nwd_zero_score_floor: float = 0.0,
        stal_candidate_iou_floor: float = 0.0,
        stal_expanded_quality_ratio: float = 0.0,
        stal_expanded_score_floor: float = 1.0,
        stal_simd_weight: float = 0.0,
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            stride (list, optional): List of stride values for different feature levels.
            eps (float, optional): A small value to prevent division by zero.
            topk2 (int, optional): Secondary topk value for additional filtering.
            stal_enabled (bool, optional): Whether to enable small-target area-aware label assignment.
            stal_stats (bool, optional): Whether to collect assignment diagnostics during training.
            stal_candidate_mode (str, optional): Candidate policy: pure, fixed, or adaptive.
            stal_area_threshold (float, optional): Maximum target-to-image area ratio that activates STAL.
            stal_small_topk (int, optional): Adaptive-mode top-k for small targets; ``None`` follows ``topk``.
            stal_small_topk_min (int, optional): Minimum top-k for near-zero-area targets. Values below the small-target
                maximum enable square-root area interpolation.
            stal_min_base_candidates (int, optional): Expand only small GTs with fewer base candidates. Zero disables.
            stal_max_extra_candidates (int, optional): Preserve base TAL and add at most this many adaptive-only
                candidates per small GT. Zero preserves the existing joint top-k behavior.
            stal_min_candidate_guarantee (bool, optional): Give an otherwise candidate-free small GT its nearest
                feature point while preserving normal TAL ranking and scoring.
            stal_crowding_mode (str, optional): Reduce relaxation for small GTs whose full candidate masks overlap.
            stal_crowded_relaxation (float, optional): Total width/height increase for crowded small GTs.
            stal_relaxation (float, optional): Maximum total width and height added to the candidate region in pixels.
            stal_relaxation_scale_mode (str, optional): Use a constant increase or interpolate it by square-root area.
            stal_relaxation_min (float, optional): Total increase near the area threshold in square-root-area mode.
            stal_min_size_stride_ratio (float, optional): Minimum candidate width and height as a multiple of the
                model's minimum stride. Zero disables this guarantee.
            stal_warmup_epochs (float, optional): Number of epochs used to linearly ramp STAL relaxation.
            stal_zero_positive_rescue (bool, optional): Whether to rescue uncovered small GTs with their best legal
                positive-quality candidate.
            stal_rescue_score_floor (float, optional): Minimum supervision weight for a zero-quality bootstrap rescue.
            stal_rescue_floor_decay_epochs (float, optional): Epochs used to linearly decay the bootstrap floor.
            stal_nwd_weight (float, optional): NWD share in small-target assignment quality; zero preserves CIoU TAL.
            stal_nwd_constant (float, optional): Positive normalization constant for NWD similarity.
            stal_nwd_target_mode (str, optional): Use match quality or CIoU for final target-score normalization.
            stal_nwd_target_weight (float, optional): NWD share for target scores in weighted mode.
            stal_nwd_zero_score_floor (float, optional): Score floor for selected small anchors with zero target score.
            stal_candidate_iou_floor (float, optional): Geometric IoU capacity floor for expanded small-GT candidates.
            stal_expanded_quality_ratio (float, optional): Minimum alignment of an adaptive-only candidate relative to
                the best candidate available before adaptive relaxation. Zero disables this gate.
            stal_expanded_score_floor (float, optional): Minimum continuous target-score multiplier for adaptive-only
                candidates relative to the best pre-relaxation candidate. One disables attenuation.
            stal_simd_weight (float, optional): CIoU-to-SimD blend weight for small-target matching quality.
        """
        super().__init__()
        numeric_stal_parameters = {
            "stal_area_threshold": stal_area_threshold,
            "stal_small_topk": stal_small_topk,
            "stal_small_topk_min": stal_small_topk_min,
            "stal_min_base_candidates": stal_min_base_candidates,
            "stal_max_extra_candidates": stal_max_extra_candidates,
            "stal_crowded_relaxation": stal_crowded_relaxation,
            "stal_relaxation": stal_relaxation,
            "stal_relaxation_min": stal_relaxation_min,
            "stal_min_size_stride_ratio": stal_min_size_stride_ratio,
            "stal_warmup_epochs": stal_warmup_epochs,
            "stal_rescue_score_floor": stal_rescue_score_floor,
            "stal_rescue_floor_decay_epochs": stal_rescue_floor_decay_epochs,
            "stal_nwd_weight": stal_nwd_weight,
            "stal_nwd_constant": stal_nwd_constant,
            "stal_nwd_target_weight": stal_nwd_target_weight,
            "stal_nwd_zero_score_floor": stal_nwd_zero_score_floor,
            "stal_candidate_iou_floor": stal_candidate_iou_floor,
            "stal_expanded_quality_ratio": stal_expanded_quality_ratio,
            "stal_expanded_score_floor": stal_expanded_score_floor,
            "stal_simd_weight": stal_simd_weight,
        }
        for name, value in numeric_stal_parameters.items():
            if isinstance(value, bool):
                raise TypeError(f"{name} must be numeric, not boolean")
        for name, value in {
            "stal_enabled": stal_enabled,
            "stal_stats": stal_stats,
            "stal_min_candidate_guarantee": stal_min_candidate_guarantee,
            "stal_zero_positive_rescue": stal_zero_positive_rescue,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps
        if stal_min_candidate_guarantee and stal_candidate_iou_floor > 0:
            raise ValueError("stal_min_candidate_guarantee cannot be combined with a positive stal_candidate_iou_floor")
        for name, value in {
            "stal_relaxation": stal_relaxation,
            "stal_min_size_stride_ratio": stal_min_size_stride_ratio,
            "stal_warmup_epochs": stal_warmup_epochs,
            "stal_rescue_floor_decay_epochs": stal_rescue_floor_decay_epochs,
            "stal_nwd_constant": stal_nwd_constant,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= stal_candidate_iou_floor <= 1.0:
            raise ValueError("stal_candidate_iou_floor must be in [0, 1]")
        if not 0.0 <= stal_expanded_quality_ratio <= 1.0:
            raise ValueError("stal_expanded_quality_ratio must be in [0, 1]")
        if not 0.0 <= stal_expanded_score_floor <= 1.0:
            raise ValueError("stal_expanded_score_floor must be in [0, 1]")
        if not 0.0 <= stal_simd_weight <= 1.0:
            raise ValueError("stal_simd_weight must be in [0, 1]")
        self.stal_candidate_iou_floor = float(stal_candidate_iou_floor)
        self.stal_expanded_quality_ratio = float(stal_expanded_quality_ratio)
        self.stal_expanded_score_floor = float(stal_expanded_score_floor)
        self.stal_simd_weight = float(stal_simd_weight)
        if not 0.0 < stal_area_threshold <= 1.0:
            raise ValueError(f"stal_area_threshold must be in (0, 1], got {stal_area_threshold}")
        if stal_small_topk is not None and (
            not isinstance(stal_small_topk, int) or isinstance(stal_small_topk, bool) or stal_small_topk < 1
        ):
            raise ValueError(f"stal_small_topk must be a positive integer, got {stal_small_topk}")
        effective_small_topk = topk if stal_small_topk is None else stal_small_topk
        if stal_small_topk_min is not None and (
            not isinstance(stal_small_topk_min, int)
            or isinstance(stal_small_topk_min, bool)
            or not 1 <= stal_small_topk_min <= effective_small_topk
        ):
            raise ValueError(f"stal_small_topk_min must be in [1, stal_small_topk], got {stal_small_topk_min}")
        if (
            not isinstance(stal_min_base_candidates, int)
            or isinstance(stal_min_base_candidates, bool)
            or stal_min_base_candidates < 0
        ):
            raise ValueError(f"stal_min_base_candidates must be a non-negative integer, got {stal_min_base_candidates}")
        if (
            not isinstance(stal_max_extra_candidates, int)
            or isinstance(stal_max_extra_candidates, bool)
            or stal_max_extra_candidates < 0
        ):
            raise ValueError(
                f"stal_max_extra_candidates must be a non-negative integer, got {stal_max_extra_candidates}"
            )
        if stal_crowding_mode not in {"none", "candidate_overlap"}:
            raise ValueError("stal_crowding_mode must be none or candidate_overlap")
        if not 0.0 <= stal_crowded_relaxation <= stal_relaxation:
            raise ValueError("stal_crowded_relaxation must be in [0, stal_relaxation]")
        if stal_relaxation < 0.0:
            raise ValueError(f"stal_relaxation must be non-negative, got {stal_relaxation}")
        if stal_relaxation_scale_mode not in {"constant", "sqrt_area"}:
            raise ValueError("stal_relaxation_scale_mode must be constant or sqrt_area")
        if not 0.0 <= stal_relaxation_min <= stal_relaxation:
            raise ValueError("stal_relaxation_min must be in [0, stal_relaxation]")
        if stal_min_size_stride_ratio < 0.0:
            raise ValueError("stal_min_size_stride_ratio must be non-negative")
        if stal_min_size_stride_ratio > 0.0 and stal_relaxation > 0.0:
            raise ValueError("stal_min_size_stride_ratio>0 requires stal_relaxation=0 for an isolated strategy")
        if stal_min_size_stride_ratio > 0.0 and stal_relaxation_scale_mode != "constant":
            raise ValueError("stal_min_size_stride_ratio>0 requires stal_relaxation_scale_mode=constant")
        if stal_warmup_epochs < 0.0:
            raise ValueError(f"stal_warmup_epochs must be non-negative, got {stal_warmup_epochs}")
        if not 0.0 <= stal_rescue_score_floor <= 1.0:
            raise ValueError(f"stal_rescue_score_floor must be in [0, 1], got {stal_rescue_score_floor}")
        if stal_rescue_floor_decay_epochs < 0.0:
            raise ValueError(
                f"stal_rescue_floor_decay_epochs must be non-negative, got {stal_rescue_floor_decay_epochs}"
            )
        if not 0.0 <= stal_nwd_weight <= 1.0:
            raise ValueError(f"stal_nwd_weight must be in [0, 1], got {stal_nwd_weight}")
        if stal_nwd_constant <= 0.0:
            raise ValueError(f"stal_nwd_constant must be positive, got {stal_nwd_constant}")
        if stal_nwd_target_mode not in {"match", "ciou", "weighted"}:
            raise ValueError(f"stal_nwd_target_mode must be match, ciou, or weighted, got {stal_nwd_target_mode}")
        if not 0.0 <= stal_nwd_target_weight <= stal_nwd_weight:
            raise ValueError(f"stal_nwd_target_weight must be in [0, stal_nwd_weight], got {stal_nwd_target_weight}")
        if not 0.0 <= stal_nwd_zero_score_floor <= 1.0:
            raise ValueError(f"stal_nwd_zero_score_floor must be in [0, 1], got {stal_nwd_zero_score_floor}")
        if stal_candidate_mode not in {"pure", "fixed", "adaptive"}:
            raise ValueError(f"stal_candidate_mode must be pure, fixed, or adaptive, got {stal_candidate_mode}")
        if stal_enabled and stal_candidate_mode == "pure":
            raise ValueError("stal_enabled=True conflicts with stal_candidate_mode=pure")
        self.stal_enabled = stal_enabled
        self.stal_stats = stal_stats
        self.stal_candidate_mode = "adaptive" if stal_enabled else stal_candidate_mode
        if stal_rescue_score_floor > 0.0 and not stal_zero_positive_rescue:
            raise ValueError("stal_rescue_score_floor>0 requires stal_zero_positive_rescue=True")
        self.stal_area_threshold = float(stal_area_threshold)
        self.stal_small_topk = effective_small_topk
        self.stal_small_topk_min = effective_small_topk if stal_small_topk_min is None else stal_small_topk_min
        self.stal_min_base_candidates = stal_min_base_candidates
        self.stal_max_extra_candidates = stal_max_extra_candidates
        self.stal_min_candidate_guarantee = stal_min_candidate_guarantee
        self.stal_crowding_mode = stal_crowding_mode
        self.stal_crowded_relaxation = float(stal_crowded_relaxation)
        self.stal_relaxation = float(stal_relaxation)
        self.stal_relaxation_scale_mode = stal_relaxation_scale_mode
        self.stal_relaxation_min = float(stal_relaxation_min)
        self.stal_min_size_stride_ratio = float(stal_min_size_stride_ratio)
        self.stal_warmup_epochs = float(stal_warmup_epochs)
        self.stal_zero_positive_rescue = stal_zero_positive_rescue
        self.stal_rescue_score_floor = float(stal_rescue_score_floor)
        self.stal_rescue_floor_decay_epochs = float(stal_rescue_floor_decay_epochs)
        self.stal_nwd_weight = float(stal_nwd_weight)
        self.stal_nwd_constant = float(stal_nwd_constant)
        self.stal_nwd_target_mode = stal_nwd_target_mode
        self.stal_nwd_target_weight = float(stal_nwd_target_weight)
        self.stal_nwd_zero_score_floor = float(stal_nwd_zero_score_floor)
        self._last_rescue_stats = torch.zeros(9, dtype=torch.long)
        self._last_stage_stats = torch.zeros(9, dtype=torch.long)
        self._last_rescue_targets = None
        self._last_bootstrap_targets = None
        self._last_bootstrap_mask = None
        self._last_expanded_score_weights = None

    def rescue_statistics(self) -> torch.Tensor:
        """Return diagnostics from the most recent zero-positive rescue attempt."""
        return self._last_rescue_stats.clone()

    def assignment_stage_statistics(self) -> torch.Tensor:
        """Return small-target counts at candidate, alignment, top-k, and conflict stages."""
        return self._last_stage_stats.clone()

    def record_assignment_stages(
        self, gt_bboxes, mask_gt, mask_in_gts, align_metric, pre_conflict_mask, post_conflict_mask, image_size
    ) -> None:
        """Record a behavior-neutral small-target assignment funnel for the latest batch."""
        image_size = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        valid = mask_gt.squeeze(-1).bool()
        small = self.small_target_mask(gt_bboxes, image_size) & valid
        has_legal = mask_in_gts.bool().any(-1)
        has_nonzero_alignment = (align_metric > 0).any(-1)
        has_positive_quality = (align_metric > self.eps).any(-1)
        pre_selected = pre_conflict_mask.bool().any(-1)
        post_selected = post_conflict_mask.bool().any(-1)
        self._last_stage_stats = torch.stack(
            (
                small.sum(),
                (small & ~has_legal).sum(),
                (small & has_legal & ~has_nonzero_alignment).sum(),
                (small & has_nonzero_alignment & ~has_positive_quality).sum(),
                (small & has_positive_quality).sum(),
                (small & has_nonzero_alignment & ~pre_selected).sum(),
                (small & pre_selected).sum(),
                (small & pre_selected & ~post_selected).sum(),
                (small & ~post_selected).sum(),
            )
        )

    def stal_rescue_score_floor_at_epoch(self, epoch: float) -> float:
        """Return the bootstrap supervision floor after optional linear decay."""
        if self.stal_rescue_score_floor == 0.0 or self.stal_rescue_floor_decay_epochs == 0.0:
            return self.stal_rescue_score_floor
        remaining = 1.0 - min(max(float(epoch), 0.0) / self.stal_rescue_floor_decay_epochs, 1.0)
        return self.stal_rescue_score_floor * remaining

    def stal_relaxation_at_epoch(self, epoch: float) -> float:
        """Return the symmetric STAL size increase after applying linear warmup."""
        if self.stal_candidate_mode != "adaptive" or self.stal_relaxation == 0.0:
            return 0.0
        if self.stal_warmup_epochs == 0.0:
            return self.stal_relaxation
        progress = min(max(float(epoch), 0.0) / self.stal_warmup_epochs, 1.0)
        return self.stal_relaxation * progress

    def stal_min_candidate_size_at_epoch(self, epoch: float) -> float:
        """Return the warmup-scaled minimum candidate width and height."""
        if self.stal_candidate_mode != "adaptive" or self.stal_min_size_stride_ratio == 0.0:
            return 0.0
        progress = 1.0 if self.stal_warmup_epochs == 0.0 else min(max(float(epoch), 0.0) / self.stal_warmup_epochs, 1.0)
        return self.stal_min_size_stride_ratio * min(self.stride) * progress

    def stal_relaxation_for_area(
        self, relative_area: torch.Tensor, epoch: float, maximum: float | None = None
    ) -> torch.Tensor:
        """Return the warmup-scaled total size increase for each target area."""
        maximum = self.stal_relaxation_at_epoch(epoch) if maximum is None else float(maximum)
        if self.stal_relaxation_scale_mode == "constant":
            return torch.full_like(relative_area, maximum)
        progress = maximum / self.stal_relaxation if self.stal_relaxation > 0.0 else 0.0
        minimum = self.stal_relaxation_min * progress
        area_ratio = (relative_area / self.stal_area_threshold).clamp(0, 1).sqrt()
        return minimum + (maximum - minimum) * (1 - area_ratio)

    def assignment_statistics(self, gt_bboxes, mask_gt, fg_mask, target_gt_idx, image_size):
        """Return GT, positive, and zero-positive counts for STAL/COCO-size/all groups."""
        if gt_bboxes.shape[1] == 0:
            return torch.zeros(15, dtype=torch.long, device=gt_bboxes.device)
        image_size = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        if image_size.numel() != 2 or (image_size <= 0).any():
            raise ValueError(f"image_size must contain positive height and width, got {image_size.tolist()}")
        gt_wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min_(0)
        valid_gt = mask_gt.squeeze(-1).bool()
        area = gt_wh.prod(-1, dtype=torch.float32)
        stal_gt = (area / image_size.prod(dtype=torch.float32) < self.stal_area_threshold) & valid_gt
        foreground = fg_mask.bool()
        positives_per_gt = torch.zeros_like(valid_gt, dtype=torch.long)
        positives_per_gt.scatter_add_(1, target_gt_idx.long(), foreground.long())

        def group_counts(group):
            return torch.stack((group.sum(), positives_per_gt[group].sum(), (group & (positives_per_gt == 0)).sum()))

        small = (area < 32**2) & valid_gt
        medium = (area >= 32**2) & (area < 96**2) & valid_gt
        large = (area >= 96**2) & valid_gt
        return torch.cat(
            (
                group_counts(stal_gt),
                group_counts(small),
                group_counts(medium),
                group_counts(large),
                group_counts(valid_gt),
            )
        )

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, image_size=None, epoch=0):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            image_size (tuple | torch.Tensor, optional): Current input image height and width.
            epoch (int, optional): Zero-based training epoch used by STAL warmup.

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        self._last_rescue_stats = torch.zeros(9, dtype=torch.long, device=device)
        self._last_stage_stats = torch.zeros(9, dtype=torch.long, device=device)
        self._last_rescue_targets = None
        self._last_bootstrap_targets = None
        self._last_bootstrap_mask = None

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes, dtype=torch.long),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
            )

        try:
            return self._forward(
                pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, image_size=image_size, epoch=epoch
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        # Recover outside the except block: exiting it drops e.__traceback__, releasing the failed attempt's GPU
        # intermediates back to the allocator so the copy-back below can succeed
        LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
        cpu_image_size = image_size.cpu() if isinstance(image_size, torch.Tensor) else image_size
        result = self._forward(
            *(t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)),
            image_size=cpu_image_size,
            epoch=epoch,
        )
        return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, image_size=None, epoch=0):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps, mask_in_gts, target_overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, image_size=image_size, epoch=epoch
        )

        pre_conflict_mask = mask_pos
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        if self.stal_stats:
            self.record_assignment_stages(
                gt_bboxes, mask_gt, mask_in_gts, align_metric, pre_conflict_mask, mask_pos, image_size
            )
        if self.stal_zero_positive_rescue:
            mask_pos = self.rescue_zero_positive_small_targets(
                mask_pos,
                align_metric,
                mask_in_gts,
                gt_bboxes,
                mask_gt,
                image_size,
                anc_points,
                epoch,
                available_anchors=~fg_mask.bool(),
            )
            target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
                mask_pos, overlaps, self.n_max_boxes, align_metric
            )
            rescued = (self._last_rescue_targets & (mask_pos.sum(-1) > 0)).sum()
            self._last_rescue_stats[5] = rescued
            self._last_rescue_stats[6] = self._last_rescue_stats[4] - rescued
            bootstrap_rescued = (self._last_bootstrap_targets & (mask_pos.sum(-1) > 0)).sum()
            self._last_rescue_stats[8] = bootstrap_rescued
            self._last_bootstrap_mask &= mask_pos.bool()

        expanded_score_weights = None
        if self._last_expanded_score_weights is not None:
            expanded_score_weights = (self._last_expanded_score_weights * mask_pos).amax(-2).unsqueeze(-1)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (target_overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        if self.stal_nwd_zero_score_floor > 0.0:
            assigned_boxes = gt_bboxes.gather(1, target_gt_idx.unsqueeze(-1).expand(-1, -1, 4))
            image_size_tensor = torch.as_tensor(image_size, dtype=assigned_boxes.dtype, device=assigned_boxes.device)
            assigned_small = self.small_target_mask(assigned_boxes, image_size_tensor)
            zero_score_small = fg_mask.bool() & assigned_small & (norm_align_metric.squeeze(-1) <= self.eps)
            norm_align_metric = torch.where(
                zero_score_small.unsqueeze(-1),
                norm_align_metric.new_tensor(self.stal_nwd_zero_score_floor),
                norm_align_metric,
            )
        if self._last_bootstrap_mask is not None:
            bootstrap_anchor_mask = self._last_bootstrap_mask.any(dim=1).unsqueeze(-1)
            floor = self.stal_rescue_score_floor_at_epoch(epoch)
            norm_align_metric = torch.where(
                bootstrap_anchor_mask, norm_align_metric.clamp_min(floor), norm_align_metric
            )
        if expanded_score_weights is not None:
            norm_align_metric = norm_align_metric * expanded_score_weights
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, image_size=None, epoch=0):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
            mask_in_gts (torch.Tensor): Legal candidate-region mask with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt, image_size=image_size, epoch=epoch)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, target_overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, image_size=image_size
        )
        self._last_expanded_score_weights = None
        if self.stal_candidate_mode == "adaptive" and self.stal_expanded_score_floor < 1.0:
            if image_size is None:
                raise ValueError("image_size is required when adaptive expanded-candidate score weighting is active")
            base_candidates = self.select_candidates_in_gts(
                anc_points, gt_bboxes, mask_gt, image_size=image_size, epoch=epoch, relaxation_override=0.0
            )
            valid_gt = mask_gt.squeeze(-1).bool()
            image_size_tensor = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            small_gt = self.small_target_mask(gt_bboxes, image_size_tensor) & valid_gt
            has_base_candidate = base_candidates.any(-1)
            best_base_metric = align_metric.masked_fill(~base_candidates, -torch.inf).amax(-1)
            quality_ratio = (align_metric / best_base_metric.clamp_min(self.eps).unsqueeze(-1)).clamp(0, 1)
            soft_weight = self.stal_expanded_score_floor + (1.0 - self.stal_expanded_score_floor) * quality_ratio
            attenuated = small_gt.unsqueeze(-1) & has_base_candidate.unsqueeze(-1) & ~base_candidates
            self._last_expanded_score_weights = torch.where(attenuated, soft_weight, torch.ones_like(soft_weight))
        if self.stal_candidate_mode == "adaptive" and self.stal_expanded_quality_ratio > 0.0:
            if image_size is None:
                raise ValueError("image_size is required when adaptive expanded-candidate quality gating is active")
            base_candidates = self.select_candidates_in_gts(
                anc_points, gt_bboxes, mask_gt, image_size=image_size, epoch=epoch, relaxation_override=0.0
            )
            valid_gt = mask_gt.squeeze(-1).bool()
            image_size_tensor = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            small_gt = self.small_target_mask(gt_bboxes, image_size_tensor) & valid_gt
            has_base_candidate = base_candidates.any(-1)
            best_base_metric = align_metric.masked_fill(~base_candidates, -torch.inf).amax(-1)
            # A relative threshold is undefined when every base candidate has zero alignment. Without the
            # positive-base guard, ``0 >= 0 * ratio`` admits every zero-quality expanded candidate.
            has_positive_base = best_base_metric > self.eps
            competitive = has_positive_base.unsqueeze(-1) & (
                align_metric >= best_base_metric.unsqueeze(-1) * self.stal_expanded_quality_ratio
            )
            gated_gt = (small_gt & has_base_candidate).unsqueeze(-1)
            quality_candidates = base_candidates | competitive
            mask_in_gts = torch.where(gated_gt, mask_in_gts & quality_candidates, mask_in_gts)
            align_metric = align_metric * mask_in_gts
        # Keep standard TAL top-k for non-small GTs; optionally interpolate small-target top-k from relative area.
        valid_gt = mask_gt.squeeze(-1).bool()
        base_topk_mask = valid_gt.unsqueeze(-1).expand(-1, -1, self.topk)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=base_topk_mask)
        if self.stal_candidate_mode == "adaptive" and self.stal_max_extra_candidates > 0:
            if image_size is None:
                raise ValueError("image_size is required when adaptive extra-candidate limiting is active")
            base_candidates = self.select_candidates_in_gts(
                anc_points, gt_bboxes, mask_gt, image_size=image_size, epoch=epoch, relaxation_override=0.0
            )
            image_size_tensor = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            small_gt = self.small_target_mask(gt_bboxes, image_size_tensor) & valid_gt
            base_metrics = align_metric.masked_fill(~base_candidates, -torch.inf)
            base_selected = self.select_topk_candidates(base_metrics, topk_mask=base_topk_mask) * base_candidates
            extra_candidates = mask_in_gts.bool() & ~base_candidates
            extra_slots = torch.ones(
                (*valid_gt.shape, self.stal_max_extra_candidates), dtype=torch.bool, device=valid_gt.device
            )
            extra_slots &= valid_gt.unsqueeze(-1)
            extra_metrics = align_metric.masked_fill(~extra_candidates, -torch.inf)
            extra_selected = self.select_topk_candidates(extra_metrics, topk_mask=extra_slots) * extra_candidates
            limited_selection = torch.maximum(base_selected, extra_selected)
            mask_topk = torch.where(small_gt.unsqueeze(-1), limited_selection, mask_topk)
        adaptive_topk_active = self.stal_small_topk != self.topk or self.stal_small_topk_min != self.topk
        if self.stal_candidate_mode == "adaptive" and adaptive_topk_active:
            if image_size is None:
                raise ValueError("image_size is required when adaptive small-target top-k is active")
            image_size_tensor = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            small_gt = self.small_target_mask(gt_bboxes, image_size_tensor) & valid_gt
            gt_area = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min(0).prod(-1, dtype=torch.float32)
            area_fraction = (
                (gt_area / image_size_tensor.prod(dtype=torch.float32) / self.stal_area_threshold).clamp(0, 1).sqrt()
            )
            dynamic_topk = (
                (self.stal_small_topk_min + (self.stal_small_topk - self.stal_small_topk_min) * area_fraction)
                .round()
                .long()
            )
            topk_slots = torch.arange(self.stal_small_topk, device=gt_bboxes.device)
            small_topk_mask = valid_gt.unsqueeze(-1) & (topk_slots < dynamic_topk.unsqueeze(-1))
            adaptive_topk = self.select_topk_candidates(align_metric, topk_mask=small_topk_mask)
            if self.stal_max_extra_candidates > 0:
                # Both policies constrain the result; adaptive top-k cannot undo the extra-candidate cap.
                adaptive_topk = adaptive_topk * limited_selection
            mask_topk = torch.where(small_gt.unsqueeze(-1), adaptive_topk, mask_topk)
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps, mask_in_gts, target_overlaps

    def rescue_zero_positive_small_targets(
        self,
        mask_pos,
        align_metric,
        mask_in_gts,
        gt_bboxes,
        mask_gt,
        image_size,
        anc_points=None,
        epoch=0,
        available_anchors=None,
    ):
        """Propose a quality-ranked or center-prior bootstrap candidate for each uncovered small GT."""
        if not self.stal_zero_positive_rescue:
            return mask_pos
        if image_size is None:
            raise ValueError("image_size is required when stal_zero_positive_rescue=True")

        image_size = torch.as_tensor(image_size, dtype=gt_bboxes.dtype, device=gt_bboxes.device)
        if image_size.numel() != 2 or (image_size <= 0).any():
            raise ValueError(f"image_size must contain positive height and width, got {image_size.tolist()}")

        small_gt = self.small_target_mask(gt_bboxes, image_size) & mask_gt.squeeze(-1).bool()
        uncovered_small_gt = small_gt & (mask_pos.sum(-1) == 0)
        legal_candidates = mask_in_gts.bool() & mask_gt.bool()
        has_legal = legal_candidates.any(-1)
        if available_anchors is not None:
            legal_candidates &= available_anchors.bool().unsqueeze(1)
        has_free_legal = legal_candidates.any(-1)
        legal_metrics = align_metric.masked_fill(~legal_candidates, -torch.inf)
        best_metric, quality_best_idx = legal_metrics.max(-1, keepdim=True)
        positive_quality = torch.isfinite(best_metric.squeeze(-1)) & (best_metric.squeeze(-1) > self.eps)
        floor = self.stal_rescue_score_floor_at_epoch(epoch)
        bootstrap_gt = uncovered_small_gt & has_free_legal & ~positive_quality & (floor > 0.0)
        if bootstrap_gt.any():
            if anc_points is None:
                raise ValueError("anc_points is required when stal_rescue_score_floor>0")
            gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
            gt_wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min(0)
            scale = gt_wh.clamp_min(1.0).unsqueeze(-2)
            distance = ((anc_points.view(1, 1, -1, 2) - gt_centers.unsqueeze(-2)) / scale).square().sum(-1)
            bootstrap_best_idx = distance.masked_fill(~legal_candidates, torch.inf).argmin(-1, keepdim=True)
        else:
            bootstrap_best_idx = quality_best_idx
        best_idx = torch.where(positive_quality.unsqueeze(-1), quality_best_idx, bootstrap_best_idx)
        eligible_gt = uncovered_small_gt & (positive_quality | bootstrap_gt)
        eligible = eligible_gt.unsqueeze(-1)

        self._last_rescue_targets = uncovered_small_gt
        self._last_bootstrap_targets = bootstrap_gt
        self._last_rescue_stats = torch.stack(
            (
                uncovered_small_gt.sum(),
                (uncovered_small_gt & has_legal).sum(),
                (uncovered_small_gt & has_free_legal).sum(),
                (uncovered_small_gt & has_free_legal & positive_quality).sum(),
                eligible_gt.sum(),
                torch.zeros((), dtype=torch.long, device=mask_pos.device),
                torch.zeros((), dtype=torch.long, device=mask_pos.device),
                bootstrap_gt.sum(),
                torch.zeros((), dtype=torch.long, device=mask_pos.device),
            )
        )

        rescue = torch.zeros_like(mask_pos)
        rescue.scatter_(-1, best_idx, eligible.to(mask_pos.dtype))
        bootstrap_rescue = torch.zeros_like(mask_pos, dtype=torch.bool)
        bootstrap_rescue.scatter_(-1, best_idx, bootstrap_gt.unsqueeze(-1))
        self._last_bootstrap_mask = bootstrap_rescue
        return torch.maximum(mask_pos, rescue)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, image_size=None):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): Matching quality, using CIoU or a size-gated CIoU/NWD blend.
            target_overlaps (torch.Tensor): Quality used to normalize target scores.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        target_overlaps = torch.zeros_like(overlaps)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        # Do not boolean-index expanded views here. On MPS, the backend can return different numbers of
        # selected elements for equivalent expanded views, which makes the predicted and ground-truth box
        # lists misaligned for IoU calculation on high-object-count batches. A single nonzero index list keeps
        # all three tensors aligned and avoids materializing the full (batch, gt, anchor, 4) expanded views.
        batch_idx, gt_idx, anchor_idx = mask_gt.nonzero(as_tuple=True)
        if batch_idx.numel():
            labels = gt_labels[batch_idx, gt_idx, 0].long()
            bbox_scores[batch_idx, gt_idx, anchor_idx] = pd_scores[batch_idx, anchor_idx, labels]
            pd_boxes = pd_bboxes[batch_idx, anchor_idx]
            gt_boxes = gt_bboxes[batch_idx, gt_idx]
            ciou = self.iou_calculation(gt_boxes, pd_boxes)
            quality = ciou
            small_gt = None
            if self.stal_nwd_weight > 0.0:
                if image_size is None:
                    raise ValueError("image_size is required when stal_nwd_weight>0")
                image_size_tensor = torch.as_tensor(image_size, dtype=gt_boxes.dtype, device=gt_boxes.device)
                if image_size_tensor.numel() != 2 or (image_size_tensor <= 0).any():
                    raise ValueError(
                        f"image_size must contain positive height and width, got {image_size_tensor.tolist()}"
                    )
                small_gt = self.small_target_mask(gt_boxes, image_size_tensor)
                nwd = self.nwd_similarity(gt_boxes, pd_boxes)
                blended = quality.lerp(nwd, self.stal_nwd_weight)
                quality = torch.where(small_gt, blended, quality)
            if self.stal_simd_weight > 0.0:
                if image_size is None:
                    raise ValueError("image_size is required when stal_simd_weight>0")
                if small_gt is None:
                    image_size_tensor = torch.as_tensor(image_size, dtype=gt_boxes.dtype, device=gt_boxes.device)
                    if image_size_tensor.numel() != 2 or (image_size_tensor <= 0).any():
                        raise ValueError("image_size must contain positive height and width")
                    small_gt = self.small_target_mask(gt_boxes, image_size_tensor)
                simd = self.simd_similarity(gt_boxes, pd_boxes)
                quality = torch.where(small_gt, quality.lerp(simd, self.stal_simd_weight), quality)
            overlaps[batch_idx, gt_idx, anchor_idx] = quality
            if self.stal_nwd_target_mode == "match":
                target_quality = quality
            elif self.stal_nwd_target_mode == "weighted" and self.stal_nwd_target_weight > 0.0:
                target_blend = ciou.lerp(nwd, self.stal_nwd_target_weight)
                target_quality = torch.where(small_gt, target_blend, ciou)
            else:
                target_quality = ciou
            target_overlaps[batch_idx, gt_idx, anchor_idx] = target_quality

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps, target_overlaps

    def nwd_similarity(self, gt_bboxes, pd_bboxes):
        """Return normalized Gaussian Wasserstein similarity for paired horizontal xyxy boxes."""
        gt_xywh = xyxy2xywh(gt_bboxes)
        pd_xywh = xyxy2xywh(pd_bboxes)
        wasserstein_2 = (gt_xywh[..., :2] - pd_xywh[..., :2]).square().sum(-1)
        wasserstein_2 += ((gt_xywh[..., 2:] - pd_xywh[..., 2:]) / 2).square().sum(-1)
        return torch.exp(-torch.sqrt(wasserstein_2.clamp_min(0)) / self.stal_nwd_constant)

    @staticmethod
    def simd_similarity(gt_bboxes, pd_bboxes):
        """Return paired SimD using the official VisDrone implementation's x=6.13 and y=4.59 constants."""
        gt = xyxy2xywh(gt_bboxes)
        pred = xyxy2xywh(pd_bboxes)
        sums = (gt[..., 2:] + pred[..., 2:]).clamp_min(1e-9)
        location_scale = sums / sums.new_tensor((6.13, 4.59))
        sim_location = ((gt[..., :2] - pred[..., :2]) / location_scale).square().sum(-1).sqrt()
        sim_shape = ((gt[..., 2:] - pred[..., 2:]) / location_scale).square().sum(-1).sqrt()
        return torch.exp(-(sim_location + sim_shape))

    def small_target_mask(self, gt_bboxes, image_size):
        """Return the existing strict relative-area STAL gate for paired horizontal xyxy boxes."""
        gt_wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min(0)
        # Area products are evaluated in FP32 because typical 800x800 images overflow FP16 before division.
        return gt_wh.prod(-1, dtype=torch.float32) / image_size.prod(dtype=torch.float32) < self.stal_area_threshold

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        candidate_topk = self.topk if topk_mask is None else topk_mask.shape[-1]
        candidate_topk = min(candidate_topk, metrics.shape[-1])
        if topk_mask is not None:
            topk_mask = topk_mask[..., :candidate_topk]
        topk_metrics, topk_idxs = torch.topk(metrics, candidate_topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        for k in range(candidate_topk):
            # Disabled variable-top-k slots add zero instead of aliasing a real candidate at anchor index zero.
            slot_enabled = topk_mask[:, :, k : k + 1].to(torch.int8)
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], slot_enabled)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        target_scores = target_scores * (fg_mask[:, :, None] > 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(
        self, xy_centers, gt_bboxes, mask_gt, eps=1e-9, image_size=None, epoch=0, relaxation_override=None
    ):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.
            image_size (tuple | torch.Tensor, optional): Current input image height and width.
            epoch (int, optional): Zero-based training epoch used by STAL warmup.
            relaxation_override (float, optional): Internal override used to recover the pre-relaxation candidate set.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        relaxation = self.stal_relaxation_at_epoch(epoch) if relaxation_override is None else relaxation_override
        min_candidate_size = self.stal_min_candidate_size_at_epoch(epoch) if relaxation_override is None else 0.0
        stal_mask = None
        if self.stal_candidate_mode == "adaptive" and (relaxation > 0.0 or min_candidate_size > 0.0):
            if image_size is None:
                raise ValueError("image_size is required when adaptive STAL candidate geometry is active")
            image_size = torch.as_tensor(image_size, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device)
            if image_size.numel() != 2 or (image_size <= 0).any():
                raise ValueError(f"image_size must contain positive height and width, got {image_size.tolist()}")
            relative_area = gt_bboxes_xywh[..., 2:].prod(-1, keepdim=True, dtype=torch.float32) / image_size.prod(
                dtype=torch.float32
            )
            stal_mask = (relative_area < self.stal_area_threshold) & mask_gt.bool()

        if self.stal_candidate_mode != "pure":
            wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
            gt_bboxes_xywh[..., 2:] = torch.where(
                (wh_mask * mask_gt).bool(),
                torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
                gt_bboxes_xywh[..., 2:],
            )
        if stal_mask is not None and self.stal_min_base_candidates > 0:
            base_boxes = xywh2xyxy(gt_bboxes_xywh)
            base_lt, base_rb = base_boxes.unsqueeze(2).chunk(2, 3)
            base_candidates = ((xy_centers - base_lt > eps) & (base_rb - xy_centers > eps)).all(3)
            undercovered = base_candidates.sum(-1, keepdim=True) < self.stal_min_base_candidates
            stal_mask &= undercovered
        if stal_mask is not None:
            if min_candidate_size > 0.0:
                min_wh = torch.full_like(gt_bboxes_xywh[..., 2:], min_candidate_size)
                gt_bboxes_xywh[..., 2:] = torch.where(
                    stal_mask.expand_as(gt_bboxes_xywh[..., 2:]),
                    torch.maximum(gt_bboxes_xywh[..., 2:], min_wh),
                    gt_bboxes_xywh[..., 2:],
                )
            relaxation_per_gt = self.stal_relaxation_for_area(relative_area, epoch, maximum=relaxation)
            if relaxation > 0.0 and self.stal_crowding_mode == "candidate_overlap":
                full_wh = torch.where(
                    stal_mask.expand_as(gt_bboxes_xywh[..., 2:]),
                    gt_bboxes_xywh[..., 2:] + relaxation,
                    gt_bboxes_xywh[..., 2:],
                )
                full_boxes = xywh2xyxy(torch.cat((gt_bboxes_xywh[..., :2], full_wh), dim=-1))
                full_lt, full_rb = full_boxes.unsqueeze(2).chunk(2, 3)
                full_candidates = ((xy_centers - full_lt > eps) & (full_rb - xy_centers > eps)).all(3)
                full_candidates &= mask_gt.bool()
                shared = full_candidates & (full_candidates.sum(1, keepdim=True) > 1)
                crowded = shared.any(-1, keepdim=True) & stal_mask
                crowded_relaxation = self.stal_crowded_relaxation * relaxation / self.stal_relaxation
                relaxation_per_gt = torch.where(
                    crowded, relaxation_per_gt.clamp(max=crowded_relaxation), relaxation_per_gt
                )
            if relaxation > 0.0:
                gt_bboxes_xywh[..., 2:] = torch.where(
                    stal_mask.expand_as(gt_bboxes_xywh[..., 2:]),
                    gt_bboxes_xywh[..., 2:] + relaxation_per_gt,
                    gt_bboxes_xywh[..., 2:],
                )
        expanded_boxes = xywh2xyxy(gt_bboxes_xywh)

        lt, rb = expanded_boxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2) left-top, right-bottom
        candidates = ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)
        if self.stal_candidate_iou_floor > 0.0 and self.stal_candidate_mode != "pure":
            if image_size is None:
                raise ValueError("image_size is required when stal_candidate_iou_floor>0")
            image_size = torch.as_tensor(image_size, device=gt_bboxes.device, dtype=gt_bboxes.dtype)
            if image_size.numel() != 2 or (image_size <= 0).any():
                raise ValueError("image_size must contain positive height and width")
            small = self.small_target_mask(gt_bboxes, image_size) & mask_gt.squeeze(-1).bool()
            # Nonnegative DFL distances force a predicted box to contain its feature point. The smallest
            # rectangle enclosing both that point and the original GT gives an optimistic IoU ceiling.
            lower = torch.minimum(gt_bboxes[..., None, :2], xy_centers)
            upper = torch.maximum(gt_bboxes[..., None, 2:], xy_centers)
            gt_area = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min(0).prod(-1, dtype=torch.float32)
            enclosing_area = (upper - lower).prod(-1, dtype=torch.float32).clamp_min(self.eps)
            capacity = gt_area.unsqueeze(-1) / enclosing_area
            candidates &= ~small.unsqueeze(-1) | (capacity >= self.stal_candidate_iou_floor)
        if self.stal_min_candidate_guarantee and self.stal_candidate_mode == "adaptive":
            if image_size is None:
                raise ValueError("image_size is required when stal_min_candidate_guarantee=True")
            image_size = torch.as_tensor(image_size, device=gt_bboxes.device, dtype=gt_bboxes.dtype)
            small = self.small_target_mask(gt_bboxes, image_size) & mask_gt.squeeze(-1).bool()
            candidate_free = small & ~candidates.any(-1)
            if candidate_free.any():
                gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
                distance = (xy_centers.view(1, 1, -1, 2) - gt_centers.unsqueeze(-2)).square().sum(-1)
                nearest = distance.argmin(-1, keepdim=True)
                guarantee = torch.zeros_like(candidates)
                guarantee.scatter_(-1, nearest, candidate_free.unsqueeze(-1))
                candidates |= guarantee
        return candidates

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.
            align_metric (torch.Tensor): Alignment metric for selecting best matches.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            # Resolve conflicts only among GTs that actually nominated the anchor. Taking argmax over every GT can
            # reassign an anchor to a non-candidate GT whose IoU happens to be larger, breaking the top-k contract.
            candidate_overlaps = overlaps.masked_fill(~mask_pos.bool(), -torch.inf)
            max_overlaps_idx = candidate_overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk or self.topk2 == 1:
            align_metric = align_metric.masked_fill(~mask_pos.bool(), -torch.inf)
            # (b, n_max_boxes, topk2)
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, image_size=None, epoch=0):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        gt_bboxes_clone = gt_bboxes.clone()
        wh_mask = gt_bboxes_clone[..., 2:4] < self.stride[0]
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
            gt_bboxes_clone[..., 2:4],
        )

        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes_clone)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """Transform rotated bounding box (xywh) to distance (ltrb). This is the inverse of dist2rbox.

    Args:
        target_bboxes (torch.Tensor): Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        target_angle (torch.Tensor): Target angle with shape (bs, h*w, 1).
        dim (int, optional): Dimension along which to split.
        reg_max (int, optional): Maximum regression value for clamping.

    Returns:
        (torch.Tensor): Rotated distance with shape (bs, h*w, 4), format [l, t, r, b].
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist
