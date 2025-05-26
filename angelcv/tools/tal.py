"""
TaskAlignedAssigner Module

This module implements the task-aligned assigner for one-stage object detection.
It computes alignment metrics between predicted boxes and ground truths and selects
positive/negative samples according to the TOOD algorithm.

References:
    - https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py
    - https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou


class TaskAlignedAssigner(nn.Module):
    """
    TaskAlignedAssigner implements the assignment strategy for one-stage detectors
    by aligning regression quality and classification confidence.

    Args:
        topk (int): The number of top candidate cells to select for each ground truth.
        alpha (float): Exponent for classification scores. Higher values (>1.0) prioritize
            classification confidence, while lower values (<1.0) reduce its impact,
            useful when classification is noisy or objects are varied.
        beta (float): Exponent for IoU scores. Higher values (>1.0) demand better
            localization precision, while lower values (<1.0) relax this requirement.
            The ratio of alpha/beta determines whether to prioritize classification or localization.
        eps (float): A small value to avoid division by zero.
        num_classes (int): Number of foreground classes.
    """

    def __init__(
        self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9, num_classes: int = 80
    ) -> None:
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes
        self.bg_index = num_classes  # Background index

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        cell_centers_scaled: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        pad_gt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign predicted boxes to ground truths.

        The steps are:
            1. Compute alignment metrics between predicted boxes and gt boxes.
            2. Select top-k cells per ground truth based on these metrics.
            3. Only consider cells whose centers lie within the gt boxes.
            4. Resolve conflicts where a cell is assigned to multiple ground truths by
               choosing the one with the highest IoU.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            cell_centers_scaled (Tensor, float32): pre-defined cell centers scaled by the stride, shape(L, 2), xy format
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)

        Returns:
            assigned_labels (Tensor): Tensor of shape (B, L) with each cell's assigned label.
            assigned_bboxes (Tensor): Tensor of shape (B, L, 4) with assigned bounding boxes.
            assigned_scores (Tensor): Tensor of shape (B, L, C) with modified scores.
            foreground_mask (Tensor): Tensor of shape (B, L) (bool) indicating positive samples.
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3
        batch_size, num_cells, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # Return all background assignments if there are no ground truths.
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_cells], self.bg_index, device=pred_scores.device)
            assigned_bboxes = torch.zeros([batch_size, num_cells, 4], device=pred_bboxes.device)
            assigned_scores = torch.zeros([batch_size, num_cells, num_classes], device=pred_scores.device)
            foreground_mask = torch.zeros([batch_size, num_cells], dtype=torch.bool, device=pred_scores.device)
            return assigned_labels, assigned_bboxes, assigned_scores, foreground_mask

        # Check which cell centers are inside each ground truth box.
        # -> Output shape: (B, n, L)
        is_in_gts = check_points_inside_bboxes(cell_centers_scaled, gt_bboxes, eps=self.eps)

        # Compute IoU for each (gt, pred_bbox) pair for every batch element.
        # Complete_box_iou does not support batched inputs so we loop over the batch.
        ious = torch.stack([complete_box_iou(gt_bboxes[i], pred_bboxes[i]) for i in range(batch_size)], dim=0)
        # Replace NaNs with zeros and clamp negative values.
        ious.nan_to_num_(0).clamp_(0)
        # Mask out cells whose centers are outside the ground truths.
        ious *= pad_gt_mask * is_in_gts

        # Reorder pred_scores from (B, L, C) to (B, C, L) using einops.rearrange.
        pred_scores = rearrange(pred_scores, "b l c -> b c l")
        gt_labels = gt_labels.long()  # Ensure labels are integer type

        # Vectorized gathering of class scores from predictions based on gt_labels.
        # Expand indices to match the shape for gathering along the class dim.
        gt_labels_exp = gt_labels.expand(-1, -1, num_cells)  # (B, n, L)
        # Gather: for each batch, select class score for each ground truth.
        bbox_cls_scores = torch.gather(pred_scores, dim=1, index=gt_labels_exp)

        # Compute the alignment metric between each ground truth and each cell.
        # alignment_metrics shape: (B, n, L)
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # For each gt, select top-k cells based on the alignment metric.
        is_in_topk = gather_topk_cells(
            metrics=alignment_metrics,
            topk=self.topk,
            topk_mask=pad_gt_mask.repeat(1, 1, self.topk).bool(),
            eps=self.eps,
        )
        # Only cells inside gt boxes are allowed and must be valid.
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # Resolve cases where a cell has been assigned to multiple ground truths.
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(1, num_max_boxes, 1)
            is_max_iou = compute_max_iou_cell(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)

        # For each cell, select the ground truth with the maximum alignment metric.
        assigned_gt_index = mask_positive.argmax(dim=-2)  # (B, L)
        batch_ind = torch.arange(batch_size, dtype=gt_labels.dtype, device=pred_scores.device).unsqueeze(-1)
        # Flatten gt_labels so that indexing works per batch.
        assigned_gt_index_flat = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gt_labels.flatten()[assigned_gt_index_flat]
        # If no positive assignment, assign background.
        assigned_labels = torch.where(
            (mask_positive_sum > 0), assigned_labels, torch.full_like(assigned_labels, self.bg_index)
        )
        # Retrieve assigned bounding boxes.
        assigned_bboxes = gt_bboxes.reshape(-1, 4)[assigned_gt_index_flat]

        # Build assigned scores via one-hot and modify them using the alignment metrics.
        # TODO [LOW]: optimize this (profiler shows that this is a bottleneck)
        assigned_scores = F.one_hot(assigned_labels, num_classes=self.num_classes + 1)
        # Remove the background column.
        assigned_scores = assigned_scores[:, :, : self.bg_index].float()

        # Rescale the alignment metrics based on the maximum for each ground truth.
        alignment_metrics = alignment_metrics * mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True)[0]
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True)[0]
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        # For each cell, take the maximum alignment metric over all ground truths.
        alignment_metrics = alignment_metrics.max(dim=-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        # Create foreground mask: cells with at least one valid assignment.
        foreground_mask = mask_positive_sum > 0

        return assigned_labels, assigned_bboxes, assigned_scores, foreground_mask


def gather_topk_cells(
    metrics: torch.Tensor,
    topk: int,
    largest: bool = True,
    topk_mask: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Select the top-k cells for each ground truth based on the given metrics.

    Args:
        metrics (Tensor): The alignment metrics with shape (B, n, L).
        topk (int): Number of top cells to select per ground truth.
        largest (bool): If True, select the largest values; otherwise, select the smallest.
        topk_mask (Tensor | None): A boolean mask of shape (B, n, topk) to ignore some cells.
        eps (float): A small constant to stabilize numerical operations.

    Returns:
        Tensor: A tensor of shape (B, n, L) with values 1.0 where the cell is in the topk, 0.0 otherwise.
    """
    num_cells = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True)[0] > eps).expand_as(topk_idxs)
    topk_idxs = torch.where(condition=topk_mask, input=topk_idxs, other=torch.zeros_like(topk_idxs))
    # Convert indices to one-hot encoding and then collapse the topk dimension.
    is_in_topk = F.one_hot(topk_idxs, num_classes=num_cells).sum(dim=-2)
    # Prevent counting a cell more than once.
    is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)


def compute_max_iou_cell(ious: torch.Tensor) -> torch.Tensor:
    """
    For each cell, identify which ground truth has the maximum IoU.

    Args:
        ious (Tensor): IoU values between predicted boxes and
                      ground truth boxes, shape (B, n, L).
                      Where B is batch size, n is number of ground truths,
                      and L is number of predicted boxes.

    Returns:
        Tensor: A tensor of shape (B, n, L) with values 1.0 where
               the cell has the maximum IoU with the ground truth,
               0.0 otherwise.
    """
    # Find the maximum IoU for each cell across all ground truths.
    max_ious = ious.max(dim=1, keepdim=True)[0]
    # Create a mask where IoU equals the maximum (allowing ties).
    # Shape: (B, n, L) where 1.0 indicates max IoU match.
    return (ious == max_ious).to(ious.dtype)


def check_points_inside_bboxes(
    points: torch.Tensor,
    bboxes: torch.Tensor,
    center_radius_tensor: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Check whether each point lies inside each bounding box.
    Optionally, restrict the check to a central region defined by a radius.

    Args:
        points (Tensor): Cell centers scaled of shape (L, 2) in [x, y] format.
        bboxes (Tensor): Ground truth boxes of shape (B, n, 4) in [xmin, ymin, xmax, ymax] format.
        center_radius_tensor (Tensor | None): Tensor of shape (L, 1) defining the radius for a central region.
        eps (float): Small value to ensure numerical stability.

    Returns:
        Tensor: A tensor of shape (B, n, L) with 1.0 where the point is inside the (possibly restricted) bbox,
                and 0.0 otherwise.
    """
    # Use einops to reshape cell centers scaled from (L, 2) to (1, 1, L, 2)
    points = rearrange(points, "l c -> 1 1 l c")  # shape: (1, 1, L, 2)
    x, y = points.chunk(2, dim=-1)  # shapes: both (1, 1, L, 1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, dim=-1)  # each is (B, n, 1, 1)

    if center_radius_tensor is not None:
        # Expand center_radius_tensor from (L, 1) to (1, 1, L, 1) using einops.
        center_radius_tensor = rearrange(center_radius_tensor, "l c -> 1 1 l c")
        bboxes_cx = (xmin + xmax) / 2.0
        bboxes_cy = (ymin + ymax) / 2.0
        xmin_sampling = bboxes_cx - center_radius_tensor
        ymin_sampling = bboxes_cy - center_radius_tensor
        xmax_sampling = bboxes_cx + center_radius_tensor
        ymax_sampling = bboxes_cy + center_radius_tensor

        xmin = torch.maximum(xmin, xmin_sampling)
        ymin = torch.maximum(ymin, ymin_sampling)
        xmax = torch.minimum(xmax, xmax_sampling)
        ymax = torch.minimum(ymax, ymax_sampling)

    left = x - xmin  # distance from point to left side
    top = y - ymin  # distance from point to top side
    right = xmax - x  # distance from point to right side
    bottom = ymax - y  # distance from point to bottom side
    # Concatenate all distances, then check if the minimum distance is positive.
    bbox_ltrb = torch.cat([left, top, right, bottom], dim=-1)  # shape: (B, n, L, 4)
    return (bbox_ltrb.min(dim=-1)[0] > eps).to(bboxes.dtype)
