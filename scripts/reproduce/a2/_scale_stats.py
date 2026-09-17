"""Training-time scale-aware assignment statistics for the A2 protocol."""

from __future__ import annotations

from collections.abc import MutableMapping

import torch

SMALL_MAX_AREA = 32**2
LARGE_MIN_AREA = 96**2
SCALE_BUCKETS = ("small", "medium", "large")
SCALE_FIELDS = (
    "gt",
    "pre_pos_total",
    "post_pos_total",
    "zero_pre",
    "zero_post",
)


def empty_scale_stats() -> dict[str, dict[str, int]]:
    """Return zeroed per-scale counters."""
    return {bucket: {field: 0 for field in SCALE_FIELDS} for bucket in SCALE_BUCKETS}


def scale_stat_fieldnames() -> tuple[str, ...]:
    """Return flat CSV/W&B field names for all scale counters and derived values."""
    fields: list[str] = []
    for bucket in SCALE_BUCKETS:
        fields.extend(
            (
                f"{bucket}_gt",
                f"{bucket}_pre_pos_total",
                f"{bucket}_post_pos_total",
                f"{bucket}_pre_avg_pos",
                f"{bucket}_post_avg_pos",
                f"{bucket}_zero_pre",
                f"{bucket}_zero_post",
                f"{bucket}_zero_pre_ratio",
                f"{bucket}_zero_post_ratio",
            )
        )
    return tuple(fields)


def flatten_scale_stats(scale_stats: dict[str, dict[str, int]]) -> dict[str, float | int]:
    """Flatten raw counters and calculate per-GT averages/zero-positive ratios."""
    output: dict[str, float | int] = {}
    for bucket in SCALE_BUCKETS:
        values = scale_stats[bucket]
        gt = int(values["gt"])
        pre_total = int(values["pre_pos_total"])
        post_total = int(values["post_pos_total"])
        zero_pre = int(values["zero_pre"])
        zero_post = int(values["zero_post"])
        output.update(
            {
                f"{bucket}_gt": gt,
                f"{bucket}_pre_pos_total": pre_total,
                f"{bucket}_post_pos_total": post_total,
                f"{bucket}_pre_avg_pos": pre_total / gt if gt else 0.0,
                f"{bucket}_post_avg_pos": post_total / gt if gt else 0.0,
                f"{bucket}_zero_pre": zero_pre,
                f"{bucket}_zero_post": zero_post,
                f"{bucket}_zero_pre_ratio": zero_pre / gt if gt else 0.0,
                f"{bucket}_zero_post_ratio": zero_post / gt if gt else 0.0,
            }
        )
    return output


def _assigned_per_gt(fg_mask: torch.Tensor, target_gt_idx: torch.Tensor, n_gt: int) -> torch.Tensor:
    """Count final foreground anchors for each GT without expanding a GT-anchor tensor."""
    batch_size = fg_mask.shape[0]
    if n_gt == 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=fg_mask.device)
    counts = torch.zeros((batch_size, n_gt), dtype=torch.long, device=fg_mask.device)
    indices = target_gt_idx.to(dtype=torch.long).clamp(0, n_gt - 1)
    counts.scatter_add_(1, indices, fg_mask.bool().to(dtype=torch.long))
    return counts


def update_scale_stats(
    scale_stats: MutableMapping[str, MutableMapping[str, int]],
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    result: tuple[torch.Tensor, ...],
    *,
    pre_assigned: torch.Tensor | None = None,
) -> None:
    """Accumulate per-scale assignment counts from one actual assigner forward.

    ``gt_bboxes`` are the transformed boxes passed to the assigner, so the
    resulting scale split follows the training-time geometry rather than the
    original image annotation geometry. ``result[3]`` and ``result[4]`` are the
    final foreground mask and GT indices returned by TAL.
    """
    if gt_bboxes.ndim != 3 or gt_bboxes.shape[-1] != 4:
        raise ValueError(f"gt_bboxes must have shape (batch, gt, 4), got {tuple(gt_bboxes.shape)}")
    if mask_gt.shape[:2] != gt_bboxes.shape[:2]:
        raise ValueError("mask_gt and gt_bboxes must have matching batch/GT dimensions")

    n_gt = gt_bboxes.shape[1]
    fg_mask, target_gt_idx = result[3], result[4]
    post_counts = _assigned_per_gt(fg_mask, target_gt_idx, n_gt)
    if pre_assigned is None:
        pre_counts = post_counts
    else:
        if pre_assigned.shape != post_counts.shape:
            raise ValueError("pre_assigned must have one count per batch/GT")
        pre_counts = pre_assigned.to(device=post_counts.device, dtype=torch.long)

    valid = mask_gt.squeeze(-1).bool()
    widths_heights = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).float().clamp_min(0)
    areas = widths_heights.prod(dim=-1)
    bucket_masks = {
        "small": valid & (areas < SMALL_MAX_AREA),
        "medium": valid & (areas >= SMALL_MAX_AREA) & (areas < LARGE_MIN_AREA),
        "large": valid & (areas >= LARGE_MIN_AREA),
    }

    for bucket, bucket_mask in bucket_masks.items():
        values = scale_stats[bucket]
        values["gt"] += int(bucket_mask.sum().item())
        selected_pre = pre_counts.masked_select(bucket_mask)
        selected_post = post_counts.masked_select(bucket_mask)
        values["pre_pos_total"] += int(selected_pre.sum().item())
        values["post_pos_total"] += int(selected_post.sum().item())
        values["zero_pre"] += int((selected_pre == 0).sum().item())
        values["zero_post"] += int((selected_post == 0).sum().item())
