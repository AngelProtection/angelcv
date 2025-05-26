from dataclasses import dataclass

from einops import rearrange
import lightning as L
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from angelcv.config import Config
from angelcv.tools.tal import TaskAlignedAssigner
from angelcv.utils.block_utils import box_to_distribution, distribution_to_box, generate_cell_centers_and_strides
from angelcv.utils.ops_utils import complete_box_iou_pairwise


@dataclass
class DetectionLoss:
    """Container for object detection loss components and their weighted sum.

    Stores IoU, classification, and distribution focal losses for monitoring
    and backpropagation.
    """

    iou: torch.Tensor
    cls: torch.Tensor
    dfl: torch.Tensor
    total: torch.Tensor

    @classmethod
    def from_device(cls, device: torch.device) -> "DetectionLoss":
        """Create a zero-initialized DetectionLoss on the specified device."""
        return cls(
            iou=torch.tensor(0.0, device=device),
            cls=torch.tensor(0.0, device=device),
            dfl=torch.tensor(0.0, device=device),
            total=torch.tensor(0.0, device=device),
        )

    def __add__(self, other: "DetectionLoss") -> "DetectionLoss":
        """Add two DetectionLoss instances component-wise."""
        if not isinstance(other, DetectionLoss):
            raise TypeError(f"Cannot add DetectionLoss with {type(other)}")

        if self.iou.device != other.iou.device:
            raise ValueError(
                f"Cannot add DetectionLoss instances on different devices: {self.iou.device} vs {other.iou.device}"
            )

        return DetectionLoss(
            iou=self.iou + other.iou,
            cls=self.cls + other.cls,
            dfl=self.dfl + other.dfl,
            total=self.total + other.total,
        )

    def calculate_total(self, batch_size: int) -> None:
        """Update total loss as weighted sum of components scaled by batch size."""
        self.total = (self.iou + self.cls + self.dfl) * batch_size

    def to_dict(self) -> dict[str, Tensor]:
        """Return loss components as a dictionary."""
        return {"iou": self.iou, "cls": self.cls, "dfl": self.dfl}

    def detach_components_(self) -> "DetectionLoss":
        """Detach all loss components from the computation graph and return self for chaining."""
        self.iou.detach_()
        self.cls.detach_()
        self.dfl.detach_()
        return self


# https://arxiv.org/pdf/2006.04388
class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) for training object detection models.

    This loss function computes the Distribution Focal Loss, which is designed to handle
    the distribution of bounding box coordinates during training. It focuses on the
    distribution of predicted distributions to the target bounding boxes.

    Attributes:
        num_dfl_bins (int): The number of bins for the distribution. Default is 16.
        eps (float): Small value to prevent edge cases where targets might
                     fall exactly on the highest bin edge. Default is 1e-3
    """

    def __init__(self, num_dfl_bins: int = 16, eps: float = 1e-3) -> None:
        super().__init__()
        self.num_dfl_bins = num_dfl_bins
        self.eps = eps

    def forward(
        self,
        predicted_distribution: Tensor,
        assigned_boxes: Tensor,
        cell_centers: Tensor,
        foreground_mask: Tensor,
    ) -> Tensor:
        """
        Calculate the Distribution Focal Loss (DFL).

        Args:
            predicted_distribution (Tensor): Predicted distribution of bounding box coordinates.
                                             Shape: (N, 4 * num_dfl_bins) or (B, A, 4 * num_dfl_bins)
                                             where N is the number of positive samples, B is batch size,
                                             A is number of anchors/cells.
            assigned_boxes (Tensor): Ground truth bounding box coordinates for positive samples.
                                     Shape: (N, 4) or (B, A, 4) matching foreground_mask.
            cell_centers (Tensor): Reference cell center coordinates corresponding to predictions.
                                   Shape: (A, 2) or (B, A, 2).
            foreground_mask (Tensor): Boolean mask indicating positive samples.
                                      Shape: (B, A) or (N,) if predictions are already filtered.

        Returns:
            Tensor: The computed Distribution Focal Loss.
        """
        assigned_distribution = box_to_distribution(assigned_boxes, cell_centers, self.num_dfl_bins - 1)

        predicted_distribution_masked = predicted_distribution[foreground_mask].view(-1, self.num_dfl_bins)
        assigned_distribution_masked = assigned_distribution[foreground_mask]

        # Clamp assigned values to ensure they are within valid range
        assigned_distribution_masked.clamp_(min=0, max=self.num_dfl_bins - 1 - self.eps)

        # Convert assigned to long type for indexing
        assigned_left = assigned_distribution_masked.floor()  # Assigned for the left bin
        assigned_right = assigned_distribution_masked.ceil()  # Assigned for the right bin

        # Calculate weights for the left and right assigneds
        weight_left = assigned_right - assigned_distribution_masked  # Weight for the left bin
        weight_right = 1 - weight_left  # Weight for the right bin

        # Compute the cross-entropy loss for both left and right assigneds
        loss_left = (
            F.cross_entropy(predicted_distribution_masked, assigned_left.long().flatten(), reduction="none").view(
                assigned_left.shape
            )
            * weight_left
        )
        loss_right = (
            F.cross_entropy(predicted_distribution_masked, assigned_right.long().flatten(), reduction="none").view(
                assigned_right.shape
            )
            * weight_right
        )

        # Mean loss across the last dimension, keeping the dimensions
        loss_dfl = (loss_left + loss_right).mean(dim=-1)
        return loss_dfl


class IouLoss(nn.Module):
    """Computes the Intersection over Union (IoU) loss for object detection.

    This loss function measures the accuracy of predicted bounding boxes by comparing
    their overlap with ground truth boxes. It uses complete IoU (CIoU) for a more
    comprehensive evaluation of box alignment.

    Note: The loss calculated here is the raw IoU loss. Weighting by prediction scores
    and normalization typically happens in the main loss computation logic.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_boxes: Tensor,
        assigned_boxes: Tensor,
        foreground_mask: Tensor,
    ) -> Tensor:
        """Calculate the IoU loss between predicted and assigned boxes.

        Args:
            predicted_boxes (Tensor): Predicted bounding box coordinates
            assigned_boxes (Tensor): Ground truth bounding box coordinates
            foreground_mask (Tensor): Boolean mask indicating positive samples

        Returns:
            Tensor: The computed IoU loss
        """
        iou = complete_box_iou_pairwise(predicted_boxes[foreground_mask], assigned_boxes[foreground_mask])
        loss_iou = 1.0 - iou
        return loss_iou


class AuxiliaryDetectionLoss:
    """
    Loss criterion for auxiliary detection tasks.

    Computes the overall loss as a weighted sum of the IoU loss,
    classification loss (BCE), and distribution focal loss (DFL) for object detection.
    """

    def __init__(self, config: Config, tal_topk: int):
        """
        Initialize the AuxiliaryDetectionLoss.

        Args:
            config (Config): The configuration object.
            tal_topk (int): Top-k candidate selection for the task-aligned assigner.
        """
        self.config = config
        head_module_config = config.model.architecture["head"][0]["v10Detect"]
        self.feature_map_strides = torch.tensor(head_module_config.args["feature_map_strides"])
        self.num_classes = head_module_config.args["num_classes"]
        self.num_dfl_bins = head_module_config.args["num_dfl_bins"]

        self.proj = torch.arange(self.num_dfl_bins, dtype=torch.float)

        # NOTE: https://arxiv.org/pdf/2108.07755 shows that alpha=0.5 is best for mAP@50
        self.tal = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.num_classes, alpha=self.config.train.loss.matcher.tal_alpha
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IouLoss()
        self.df_loss = DistributionFocalLoss(self.num_dfl_bins)

    def __call__(self, predictions: list[Tensor], batch: dict[str, Tensor]) -> DetectionLoss:
        """
        Compute the total loss for the given predictions and batch targets.

        Args:
            predictions (List[Tensor]): Model predictions from different feature levels.
            batch (dict): Batch dictionary containing ground truth boxes and labels. Keys:
                            - "boxes": Tensor of shape (batch_size, num_gt, 4) in normalized coordinates.
                            - "labels": Tensor of shape (batch_size, num_gt, 1).

        Returns:
            DetectionLoss: The detection loss components including total loss and individual components (iou, cls, dfl).
        """
        # All shapes shown for an input of 640x640, 640/32 = 20 (hl), 640/16 = 40 (ml), 640/8 = 80 (ll)
        # predictions [torch.Size([4, 144, 80, 80]), torch.Size([4, 144, 40, 40]), torch.Size([4, 144, 20, 20])]
        # where 144 is 64 (16 * 4) + 80 (num_classes)

        # predictions: [B,144,H,W] per level (144 = 64 box + 80 cls)
        cat_predictions = torch.cat([rearrange(f, "b c h w -> b (h w) c") for f in predictions], dim=1)
        device = cat_predictions.device

        # Split regression and classification components
        box_channels = 4 * self.num_dfl_bins
        predicted_distribution, predicted_scores = cat_predictions.split((box_channels, self.num_classes), dim=-1)
        predicted_distribution = predicted_distribution.contiguous()  # [B, 8400, 64]
        predicted_scores = predicted_scores.contiguous()  # [B, 8400, 80]

        # cell_centers[8400,2], cell_strides[8400,1]
        cell_centers, cell_strides = generate_cell_centers_and_strides(predictions, self.feature_map_strides)

        # Multiply feature map cells for the stride to get the image size inputted to the model
        height_width = (
            torch.tensor(predictions[0].shape[2:], device=device, dtype=batch["boxes"].dtype)
            * self.feature_map_strides[0]
        )

        # Ground truth: boxes[B,num_gt,4], labels[B,num_gt,1], mask[B,num_gt,1]
        gt_boxes = batch["boxes"] * height_width.repeat(2)
        gt_labels = batch["labels"]
        gt_mask = gt_boxes.sum(dim=2, keepdim=True).gt_(0.0)

        # Convert distributions to boxes
        predicted_distribution_bins = rearrange(
            predicted_distribution, "b a (four bins) -> b a four bins", four=4, bins=self.num_dfl_bins
        )
        predicted_distribution_bins = predicted_distribution_bins.softmax(dim=-1)
        predicted_distribution_bins = torch.matmul(
            predicted_distribution_bins, self.proj.type_as(predicted_distribution_bins)
        )
        predicted_boxes = distribution_to_box(predicted_distribution_bins, cell_centers)  # [B,8400,4]

        # Convert to pixel space
        predicted_scores_probs = predicted_scores.detach().sigmoid()  # [B,8400,80]
        predicted_boxes_scaled = (predicted_boxes.detach() * cell_strides).type_as(gt_boxes)  # [B,8400,4]
        cell_centers_scaled = (cell_centers * cell_strides).type_as(gt_boxes)  # [8400,2]

        # Task alignment: boxes[B,8400,4], scores[B,8400,80], mask[B,8400]
        _, assigned_boxes, assigned_scores, foreground_mask = self.tal(
            predicted_scores_probs,
            predicted_boxes_scaled,
            cell_centers_scaled,
            gt_labels,
            gt_boxes,
            gt_mask,
        )

        # Ensure a non-zero denominator
        assigned_scores_sum = max(assigned_scores.sum(), 1)

        detection_loss = DetectionLoss.from_device(device)
        # Classification loss
        detection_loss.cls = (
            self.bce_loss(predicted_scores, assigned_scores.type_as(predicted_scores)).sum() / assigned_scores_sum
        )

        # IoU & DFL loss
        if foreground_mask.sum():
            assigned_boxes /= cell_strides
            box_loss_weight = assigned_scores.sum(dim=-1)[foreground_mask]  # Shape: (N,)

            dfl_loss = self.df_loss(
                predicted_distribution,
                assigned_boxes,
                cell_centers,
                foreground_mask,
            )
            iou_loss = self.iou_loss(
                predicted_boxes,
                assigned_boxes,
                foreground_mask,
            )

            detection_loss.dfl = (dfl_loss * box_loss_weight).sum() / assigned_scores_sum
            detection_loss.iou = (iou_loss * box_loss_weight).sum() / assigned_scores_sum

        # Apply loss weights
        detection_loss.iou *= self.config.train.loss.weights.iou_loss
        detection_loss.cls *= self.config.train.loss.weights.cls_loss
        detection_loss.dfl *= self.config.train.loss.weights.df_loss

        batch_size = predicted_scores.shape[0]
        detection_loss.calculate_total(batch_size)
        return detection_loss.detach_components_()


class EndToEndDetectionLoss:
    """Criterion class for computing end-to-end training losses in object detection.

    This class encapsulates the loss calculation for both one-to-many and one-to-one detection tasks.
    It combines various loss components to provide a comprehensive evaluation of the model's performance.
    """

    def __init__(self, config: Config):
        """
        Initialize the EndToEndDetectionLoss.

        Args:
            config (Config): The configuration object.
        """
        self.config = config
        self.one_to_many = AuxiliaryDetectionLoss(config, tal_topk=config.train.loss.matcher.tal_topk)
        self.one_to_one = AuxiliaryDetectionLoss(config, tal_topk=1)

    def __call__(self, predictions: dict[str, Tensor], batch: dict[str:Tensor]) -> DetectionLoss:
        """Calculate the total loss for IoU, class, and distribution focal loss.

        This method computes the sum of the losses from one-to-many and one-to-one detection tasks,
        providing a comprehensive evaluation of the model's performance. The losses are aggregated
        and returned as a DetectionLoss instance, which can be used for optimization during training.

        Args:
            predictions (dict[str, Tensor]): A dictionary containing model predictions.
                - "one_to_many": Predictions for one-to-many detection tasks.
                - "one_to_one": Predictions for one-to-one detection tasks.
            batch (dict[str, Tensor]): A dictionary containing the ground truth data for the current batch.
                - Contains the necessary information for loss computation, such as target boxes and labels.

        Returns:
            DetectionLoss: The combined loss components including total loss and individual components.
        """
        loss_o2m_components = self.one_to_many(predictions["one_to_many"], batch)
        loss_o2o_components = self.one_to_one(predictions["one_to_one"], batch)
        return loss_o2m_components + loss_o2o_components
