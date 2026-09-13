# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

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
        small_area_threshold: float = 1024.0,
        dynamic_topk_small: bool = False,
        dynamic_topk_lambda: float = 0.8,
        dynamic_topk_cap: bool = False,
        dynamic_topk_min: int = 0,
        dynamic_topk_max: int = 0,
        candidate_expand_0_8: float = 16.0,
        candidate_expand_8_16: float = -1.0,
        candidate_expand_linear_decay: bool = False,
        candidate_expand_full_epochs: int = 60,
        candidate_expand_decay_epochs: int = 60,
        candidate_expand_coverage_triggered: bool = False,
        candidate_expand_coverage_tiered: bool = False,
        candidate_expand_coverage_min: int = 3,
        candidate_expand_coverage_target: float = 20.0,
        candidate_expand_coverage_long_side: float = 32.0,
        collect_coverage_stats: bool = False,
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
            small_area_threshold (float, optional): GT area threshold for P2 anchor gating (default 1024 = 32x32).
            dynamic_topk_small (bool, optional): Apply dynamic TopK only to small GTs.
            dynamic_topk_lambda (float, optional): Small-GT TopK ratio in K=ceil(lambda*x), where x is candidate count.
            dynamic_topk_cap (bool, optional): Cap small-GT dynamic K at this assigner's fixed top-k.
            dynamic_topk_min (int, optional): K_min for bounded small-GT dynamic TopK.
            dynamic_topk_max (int, optional): K_max for bounded small-GT dynamic TopK; 0 disables it.
            candidate_expand_0_8 (float, optional): Per-side candidate-box target for original sides in [0, 8).
                Set to -1 to disable this interval.
            candidate_expand_8_16 (float, optional): Per-side candidate-box target for original sides in [8, 16).
                Set to -1 to disable this interval.
            candidate_expand_linear_decay (bool, optional): Keep full expansion first, then linearly contract to the
                original candidate box.
            candidate_expand_full_epochs (int, optional): Number of fully expanded epochs before contraction.
            candidate_expand_decay_epochs (int, optional): Number of epochs over which expansion strength reaches 0.
            candidate_expand_coverage_triggered (bool, optional): Expand only short-side [8, 16) GTs whose
                baseline candidate count is below ``candidate_expand_coverage_min``.
            candidate_expand_coverage_tiered (bool, optional): Use the C2 deficit-tiered budget: empty ordinary GTs
                target ``candidate_expand_coverage_min`` candidates, non-empty ordinary GTs target one fewer, and
                elongated GTs target the full minimum.
            candidate_expand_coverage_min (int, optional): Minimum candidate count targeted by coverage-triggered mode.
            candidate_expand_coverage_target (float, optional): Per-side target used by coverage-triggered expansion.
            candidate_expand_coverage_long_side (float, optional): Minimum long side for full-budget C2 expansion.
            collect_coverage_stats (bool, optional): Retain detached per-batch coverage counters for diagnostics.
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps
        self.small_area_threshold = small_area_threshold
        self.dynamic_topk_small = dynamic_topk_small
        self.dynamic_topk_lambda = dynamic_topk_lambda
        self.dynamic_topk_cap = dynamic_topk_cap
        self.dynamic_topk_min = dynamic_topk_min
        self.dynamic_topk_max = dynamic_topk_max
        self.candidate_expand_0_8 = candidate_expand_0_8
        self.candidate_expand_8_16 = candidate_expand_8_16
        self.candidate_expand_linear_decay = candidate_expand_linear_decay
        self.candidate_expand_full_epochs = candidate_expand_full_epochs
        self.candidate_expand_decay_epochs = candidate_expand_decay_epochs
        self.candidate_expand_coverage_triggered = candidate_expand_coverage_triggered
        self.candidate_expand_coverage_tiered = candidate_expand_coverage_tiered
        self.candidate_expand_coverage_min = candidate_expand_coverage_min
        self.candidate_expand_coverage_target = candidate_expand_coverage_target
        self.candidate_expand_coverage_long_side = candidate_expand_coverage_long_side
        self.collect_coverage_stats = collect_coverage_stats
        self.candidate_expand_strength = 1.0
        self._coverage_stats = {}
        self._coverage_added_mask = None
        self._assignment_stage_counts = {}
        if self.dynamic_topk_small and not 0.0 <= self.dynamic_topk_lambda <= 1.0:
            raise ValueError("dynamic_topk_lambda must be within [0, 1]")
        if self.dynamic_topk_min < 0:
            raise ValueError("dynamic_topk_min must be >= 0")
        if self.dynamic_topk_max < 0:
            raise ValueError("dynamic_topk_max must be >= 0")
        if self.dynamic_topk_max and self.dynamic_topk_min > self.dynamic_topk_max:
            raise ValueError("dynamic_topk_min must be <= dynamic_topk_max when the maximum is enabled")
        self._validate_candidate_expand_target("candidate_expand_0_8", self.candidate_expand_0_8, 8.0)
        self._validate_candidate_expand_target("candidate_expand_8_16", self.candidate_expand_8_16, 16.0)
        if self.candidate_expand_full_epochs < 0:
            raise ValueError("candidate_expand_full_epochs must be >= 0")
        if self.candidate_expand_decay_epochs < 0:
            raise ValueError("candidate_expand_decay_epochs must be >= 0")
        if self.candidate_expand_coverage_min < 1:
            raise ValueError("candidate_expand_coverage_min must be >= 1")
        if self.candidate_expand_coverage_triggered and self.candidate_expand_coverage_tiered:
            raise ValueError("coverage-triggered C1 and coverage-tiered C2 modes are mutually exclusive")
        self._validate_candidate_expand_target(
            "candidate_expand_coverage_target", self.candidate_expand_coverage_target, 16.0
        )
        if self.candidate_expand_coverage_long_side < 16.0:
            raise ValueError("candidate_expand_coverage_long_side must be >= 16")
        if self.candidate_expand_linear_decay and self.candidate_expand_decay_epochs < 1:
            raise ValueError("candidate_expand_decay_epochs must be >= 1 when linear decay is enabled")
        self._stride_tensor = None

    @staticmethod
    def _validate_candidate_expand_target(name: str, target: float, interval_upper: float) -> None:
        """Validate one optional per-side candidate-box expansion target."""
        if target != -1 and target < interval_upper:
            raise ValueError(f"{name} must be -1 (disabled) or >= {interval_upper:g}")

    def set_epoch(self, epoch: int) -> float:
        """Set the one-based training epoch and return the effective candidate-expansion strength."""
        if epoch < 1:
            raise ValueError("epoch must be one-based and >= 1")
        if not self.candidate_expand_linear_decay or epoch <= self.candidate_expand_full_epochs:
            strength = 1.0
        else:
            decay_step = epoch - self.candidate_expand_full_epochs
            strength = max(1.0 - decay_step / self.candidate_expand_decay_epochs, 0.0)
        self.candidate_expand_strength = strength
        return strength

    def coverage_stats(self) -> dict[str, torch.Tensor]:
        """Return detached coverage counters from the most recent assignment batch."""
        return {name: value.detach().clone() for name, value in self._coverage_stats.items()}

    def assignment_stage_counts(self) -> dict[str, torch.Tensor]:
        """Return per-GT counts before TopK, before conflict resolution, and after conflict resolution."""
        return {name: value.detach().clone() for name, value in self._assignment_stage_counts.items()}

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride_tensor=None):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            stride_tensor (torch.Tensor, optional): Stride value for each anchor point, shape (num_total_anchors, 1).

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
        self._stride_tensor = stride_tensor
        self._assignment_stage_counts = {}
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        # Recover outside the except block: exiting it drops e.__traceback__, releasing the failed attempt's GPU
        # intermediates back to the allocator so the copy-back below can succeed
        LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
        result = self._forward(*(t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)))
        return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
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
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        if self.collect_coverage_stats:
            self._assignment_stage_counts["final"] = mask_pos.sum(-1).to(torch.long)
        if self.collect_coverage_stats and self._coverage_added_mask is not None:
            self._coverage_stats["coverage_supplement_final"] = (
                mask_pos.bool() & self._coverage_added_mask
            ).sum()

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
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
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        if self.collect_coverage_stats:
            self._assignment_stage_counts["candidate"] = mask_in_gts.sum(-1).to(torch.long)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(
            align_metric,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
            candidate_mask=mask_in_gts,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
        )
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        if self.collect_coverage_stats:
            self._assignment_stage_counts["preconflict"] = mask_pos.sum(-1).to(torch.long)
        if self.collect_coverage_stats and self._coverage_added_mask is not None:
            added_selected = mask_pos.bool() & self._coverage_added_mask
            multi_gt_anchor = mask_pos.sum(-2) > 1
            self._coverage_stats["coverage_supplement_topk"] = added_selected.sum()
            self._coverage_stats["coverage_supplement_conflict"] = (
                added_selected & multi_gt_anchor.unsqueeze(-2)
            ).sum()

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
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
            overlaps[batch_idx, gt_idx, anchor_idx] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None, candidate_mask=None, gt_bboxes=None, mask_gt=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.
            candidate_mask (torch.Tensor, optional): Expanded-box candidate mask, used only for dynamic small-GT TopK.
            gt_bboxes (torch.Tensor, optional): Original GT boxes in xyxy format.
            mask_gt (torch.Tensor, optional): Valid-GT mask.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        mask_topk = count_tensor.to(metrics.dtype)
        if not self.dynamic_topk_small:
            return mask_topk

        if candidate_mask is None or gt_bboxes is None or mask_gt is None:
            raise ValueError("dynamic small-GT TopK requires candidate_mask, gt_bboxes, and mask_gt")

        valid_gt = mask_gt.squeeze(-1).bool()
        gt_wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., :2]).clamp_min(0)
        small_gt = valid_gt & (gt_wh.prod(-1) < self.small_area_threshold)
        candidate_count = candidate_mask.sum(-1)
        dynamic_k = torch.ceil(candidate_count * self.dynamic_topk_lambda).to(torch.long)
        if self.dynamic_topk_cap or self.dynamic_topk_max > 0:
            # Preserve each branch's scope: O2M commonly uses topk=10 while O2O may use topk=1.
            configured_max = self.dynamic_topk_max or self.topk
            effective_max = min(configured_max, self.topk)
            floor_k = min(self.dynamic_topk_min, effective_max)
            dynamic_k = torch.maximum(dynamic_k, torch.full_like(dynamic_k, floor_k))
            dynamic_k = torch.minimum(dynamic_k, candidate_count.to(torch.long)).clamp_max(effective_max)
        dynamic_k = torch.where(small_gt, dynamic_k, torch.zeros_like(dynamic_k))
        max_dynamic_k = int(dynamic_k.max().item())
        if max_dynamic_k == 0:
            return mask_topk

        candidate_metrics = metrics.masked_fill(~candidate_mask.bool(), float("-inf"))
        _, dynamic_idxs = torch.topk(candidate_metrics, max_dynamic_k, dim=-1, largest=True)
        ranks = torch.arange(max_dynamic_k, device=metrics.device).view(1, 1, -1)
        selected = ranks < dynamic_k.unsqueeze(-1)
        dynamic_mask = torch.zeros_like(metrics)
        dynamic_mask.scatter_(-1, dynamic_idxs, selected.to(metrics.dtype))
        return torch.where(small_gt.unsqueeze(-1), dynamic_mask, mask_topk)

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

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        original_wh = gt_bboxes_xywh[..., 2:]
        valid_sides = mask_gt.bool().expand_as(original_wh)
        expanded_wh = original_wh.clone()

        if self.candidate_expand_0_8 != -1:
            interval_0_8 = valid_sides & (original_wh < 8)
            target_0_8 = original_wh + self.candidate_expand_strength * (self.candidate_expand_0_8 - original_wh)
            expanded_wh = torch.where(
                interval_0_8, target_0_8, expanded_wh
            )
        if self.candidate_expand_8_16 != -1:
            interval_8_16 = valid_sides & (original_wh >= 8) & (original_wh < 16)
            target_8_16 = original_wh + self.candidate_expand_strength * (self.candidate_expand_8_16 - original_wh)
            expanded_wh = torch.where(
                interval_8_16, target_8_16, expanded_wh
            )

        gt_bboxes_xywh[..., 2:] = expanded_wh
        baseline_boxes = xywh2xyxy(gt_bboxes_xywh)

        def centers_in_boxes(boxes):
            lt, rb = boxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2) left-top, right-bottom
            return ((xy_centers - lt > eps) & (rb - xy_centers > eps)).all(3)

        mask = centers_in_boxes(baseline_boxes)  # (b, n_boxes, h*w)

        valid_gt = mask_gt.squeeze(-1).bool()
        base_count = mask.sum(-1)
        short_side_8_16 = (original_wh >= 8.0) & (original_wh < 16.0)
        coverage_group = valid_gt & short_side_8_16.any(-1)
        elongated = original_wh.max(-1).values >= self.candidate_expand_coverage_long_side
        self._coverage_added_mask = torch.zeros_like(mask, dtype=torch.bool)
        if self.collect_coverage_stats:
            zero = base_count.new_zeros(())
            self._coverage_stats = {
                "coverage_gt_base0": (coverage_group & base_count.eq(0)).sum(),
                "coverage_gt_base1": (coverage_group & base_count.eq(1)).sum(),
                "coverage_gt_base2": (coverage_group & base_count.eq(2)).sum(),
                "coverage_gt_base3plus": (coverage_group & base_count.ge(3)).sum(),
                "coverage_gt_ordinary": (coverage_group & ~elongated).sum(),
                "coverage_gt_elongated": (coverage_group & elongated).sum(),
                "coverage_triggered_gt": zero.clone(),
                "coverage_supplement_candidates": zero.clone(),
                "coverage_supplement_topk": zero.clone(),
                "coverage_supplement_conflict": zero.clone(),
                "coverage_supplement_final": zero.clone(),
            }

        if self.candidate_expand_coverage_triggered or self.candidate_expand_coverage_tiered:
            desired_count = torch.full_like(base_count, self.candidate_expand_coverage_min)
            if self.candidate_expand_coverage_tiered:
                ordinary_target = max(self.candidate_expand_coverage_min - 1, 1)
                desired_count = torch.where(elongated, desired_count, ordinary_target)
                desired_count = torch.where(base_count == 0, self.candidate_expand_coverage_min, desired_count)
            eligible = valid_gt & short_side_8_16.any(-1) & (base_count < desired_count)
            if eligible.any():
                coverage_wh = expanded_wh.clone()
                coverage_target = original_wh + self.candidate_expand_strength * (
                    self.candidate_expand_coverage_target - original_wh
                )
                coverage_wh = torch.where(
                    eligible.unsqueeze(-1) & short_side_8_16,
                    coverage_target,
                    coverage_wh,
                )
                coverage_boxes = xywh2xyxy(torch.cat((gt_bboxes_xywh[..., :2], coverage_wh), dim=-1))
                expanded_mask = centers_in_boxes(coverage_boxes)
                ring_mask = expanded_mask & ~mask

                # Add only the nearest ring anchors needed to reach the desired count. This keeps the
                # baseline/core candidates and prevents a broad expansion from flooding TopK with noise.
                need = (desired_count - base_count).clamp_min(0)
                max_need = min(self.candidate_expand_coverage_min, ring_mask.shape[-1])
                ring_distance = ((xy_centers.view(1, 1, -1, 2) - gt_bboxes[..., None, :2]) ** 2).sum(-1)
                ring_distance = ring_distance.masked_fill(~ring_mask, float("inf"))
                nearest_distance, nearest_idx = torch.topk(ring_distance, max_need, dim=-1, largest=False)
                ranks = torch.arange(max_need, device=xy_centers.device).view(1, 1, -1)
                take = (ranks < need.unsqueeze(-1)) & torch.isfinite(nearest_distance) & eligible.unsqueeze(-1)
                ring_selected = torch.zeros_like(mask)
                ring_selected.scatter_(-1, nearest_idx, take.to(mask.dtype))
                self._coverage_added_mask = ring_selected.bool()
                if self.collect_coverage_stats:
                    self._coverage_stats["coverage_triggered_gt"] = eligible.sum()
                    self._coverage_stats["coverage_supplement_candidates"] = ring_selected.sum()
                mask = torch.where(eligible.unsqueeze(-1), mask | ring_selected, mask)

        if self._stride_tensor is not None and len(self.stride) > 0 and self.stride[0] < 8:
            p2_mask = (self._stride_tensor == self.stride[0]).squeeze(-1)  # (h*w,)
            if p2_mask.any():
                gt_wh = gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]
                gt_area = gt_wh.clamp_min_(0).prod(-1)  # (b, n_boxes)
                is_small = gt_area < self.small_area_threshold  # (b, n_boxes)
                non_small = (~is_small).unsqueeze(-1)  # (b, n_boxes, 1)
                p2_expanded = p2_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, h*w)
                mask = mask & ~(non_small & p2_expanded)

        return mask

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

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
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

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        gt_bboxes_clone = gt_bboxes.clone()
        expand_threshold = 8
        expand_target = 16
        wh_mask = gt_bboxes_clone[..., 2:4] < expand_threshold
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(expand_target, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
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
        mask = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box

        if self._stride_tensor is not None and len(self.stride) > 0 and self.stride[0] < 8:
            p2_mask = (self._stride_tensor == self.stride[0]).squeeze(-1)  # (h*w,)
            if p2_mask.any():
                gt_wh = gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2] if gt_bboxes.shape[-1] >= 4 else gt_bboxes_clone[..., 2:4]
                gt_area = gt_wh.clamp_min_(0).prod(-1)  # (b, n_boxes)
                is_small = gt_area < self.small_area_threshold  # (b, n_boxes)
                non_small = (~is_small).unsqueeze(-1)  # (b, n_boxes, 1)
                p2_expanded = p2_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, h*w)
                mask = mask & ~(non_small & p2_expanded)

        return mask


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
