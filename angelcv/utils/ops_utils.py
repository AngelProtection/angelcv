import math

import torch
from torch import Tensor


def round_to_multiple(value: float, multiple: int) -> int:
    """
    Computes the smallest integer that is both greater than or equal to the input value
    and evenly divisible by the specified multiple.

    Args:
        value (float): Target value to find the next multiple for
        multiple (int): The multiple that the result must be divisible by

    Returns:
        int: Smallest number >= value that is divisible by multiple
    """
    next_multiple_factor = math.ceil(value / multiple)
    result = multiple * next_multiple_factor

    return result


# TODO [LOW]: figure out a better way to handle eps (based on type?)
# NOTE: this is equivalent to torch.diagonal(torchvision.ops.complete_box_iou(boxes1, boxes2))
def complete_box_iou_pairwise(boxes1: Tensor, boxes2: Tensor, keepdim: bool = False, eps: float = 1e-7) -> Tensor:
    """
    Calculate Complete IoU (CIoU) between two sets of bounding boxes.

    CIoU adds a shape constraint and a normalized distance term to the IoU calculation,
    helping to minimize the normalized distance between predicted box and target box
    while maintaining the consistency of aspect ratios.

    Args:
        boxes1: First set of bounding boxes (N, 4) in (x1, y1, x2, y2) format
        boxes2: Second set of bounding boxes (N, 4) in (x1, y1, x2, y2) format
        keepdim: Whether to keep the output tensor's dimensions the same as input
        eps: Small value to prevent division by zero

    Returns:
        Tensor: CIoU values for each pair of boxes
    """
    # Extract coordinates from boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.chunk(4, -1)  # (N, 1) each
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.chunk(4, -1)  # (N, 1) each

    # Calculate box dimensions
    b1_w = b1_x2 - b1_x1
    b1_h = b1_y2 - b1_y1 + eps  # Add eps to height to avoid division by zero
    b2_w = b2_x2 - b2_x1
    b2_h = b2_y2 - b2_y1 + eps  # Add eps to height to avoid division by zero

    # Calculate intersection area
    intersection_left = torch.maximum(b1_x1, b2_x1)
    intersection_right = torch.minimum(b1_x2, b2_x2)
    intersection_top = torch.maximum(b1_y1, b2_y1)
    intersection_bottom = torch.minimum(b1_y2, b2_y2)

    intersection_width = (intersection_right - intersection_left).clamp_(0)
    intersection_height = (intersection_bottom - intersection_top).clamp_(0)
    intersection = intersection_width * intersection_height

    # Calculate union area
    union = (b1_w * b1_h) + (b2_w * b2_h) - intersection + eps

    # Calculate IoU
    iou = intersection / union

    # Calculate additional CIoU terms
    # 1. Convex diagonal squared
    convex_width = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    convex_height = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    convex_diagonal_squared = convex_width.pow(2) + convex_height.pow(2) + eps

    # 2. Center distance squared
    center_distance_squared = (
        ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)).pow(2) + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)).pow(2)
    ) / 4

    # 3. Aspect ratio consistency term
    v = (4 / (torch.pi**2)) * (torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h)).pow(2)

    # Calculate alpha (trade-off parameter)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    # Combine all terms for CIoU
    ciou = iou - (
        center_distance_squared / convex_diagonal_squared  # Distance term
        + v * alpha  # Aspect ratio consistency term
    )

    # Handle keepdim parameter
    return ciou if keepdim else ciou.squeeze(-1)
