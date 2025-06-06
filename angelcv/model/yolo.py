from collections import deque
import inspect
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from angelcv.config import ConfigManager
from angelcv.config.config_registry import BlockConfig, Config
from angelcv.interface.inference_result import InferenceResult
from angelcv.model.blocks import v10Detect
from angelcv.tools.loss import DetectionLoss, EndToEndDetectionLoss
from angelcv.utils.block_utils import get_block_name_to_impl_dict
from angelcv.utils.logging_manager import get_logger
from angelcv.utils.ops_utils import round_to_multiple
from angelcv.utils.source_utils import ImageCoordinateMapper

# Configure logging
logger = get_logger(__name__)


class YoloDetectionModel(pl.LightningModule):
    """
    YOLOv10 detection model implementation.

    This module implements a complete object detection model with:
    1. Configurable backbone and detection head
    2. Support for various input configurations
    3. Automatic stride calculation for feature pyramids
    4. Proper weight initialization for optimal training
    5. Full PyTorch Lightning integration for training
    6. Visualization of detections during training and validation

    The model follows the YOLOv10 architecture while maintaining flexibility
    for different configurations and use cases.
    """

    def __init__(self, config: Config):
        """
        Initialize YOLOv10 detection model.

        Args:
            config_path: Path to model config file or config dictionary
            in_channels: Number of input image channels
            num_classes: Optional override for number of detection classes
        """
        super().__init__()

        # Save class parameters (constructor arguments)
        # NOTE: accessible through self.hparams with dot notation (i.e. self.hparams.config)
        self.save_hyperparameters()

        self.config = config
        self.loss_fn = None
        self.experiment_dir = None

        # Build model architecture
        # NOTE: self.config element are modified (i.e. out_channels scaled by channels_scale)
        self.blocks, self.save_indices = model_arch_from_config(self.config)

        # Initialize model weights
        self._initialize_model_weights()

        # Validation mAP
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy", iou_thresholds=[0.5, 0.75], rec_thresholds=np.linspace(0.0, 1.00, 101).tolist()
        )

        # Disable the many detections warning (YOLO returns a lot of detections)
        self.map_metric.warn_on_many_detections = False

        self.loss_buffers = {
            "loss/total/train": deque(maxlen=20),
            "loss/iou/train": deque(maxlen=20),
            "loss/clf/train": deque(maxlen=20),
            "loss/dfl/train": deque(maxlen=20),
        }

        # Visualization settings
        self.vis_batch_indices = []
        self.vis_interval = 10  # Save visualizations every N epochs
        self.vis_samples_per_batch = 4  # Number of samples to visualize per batch
        self.vis_confidence_threshold = 0.3  # Minimum confidence for displaying detections

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the model (inference).

        Args:
            images: Either raw image tensor (inference)

        Returns:
            torch.Tensor: Model predictions
        """
        # Store intermediate outputs by source index
        source_idx_to_feats: dict[int, torch.Tensor] = {}
        current_feats = images

        for module in self.blocks:
            if module.source != [-1]:  # if the source isn't only the previous layer
                current_feats = [
                    current_feats if source_idx == -1 else source_idx_to_feats[source_idx]
                    for source_idx in module.source
                ]
            # NOTE: no "else" required as current_feats is already updated

            current_feats = module(current_feats)
            if module.index in self.save_indices:
                source_idx_to_feats[module.index] = current_feats

        return current_feats

    def _initialize_model_weights(self) -> None:
        """
        Initialize model weights using standard techniques for YOLO architectures.
        Applies appropriate initialization for different layer types.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                # INFO: introduced by MMYOLO - to improve training stability on small datasets
                module.eps = 0.001  # default 1e-5
                module.momentum = 0.03  # default 0.1
            elif isinstance(module, v10Detect):
                module.initialize_biases()

    # NOTE: this is just created as "load_from_checkpoint" (immplemented by Lightning)
    # isn't working with the current implementation
    @classmethod
    def load_from_checkpoint_custom(cls, checkpoint_path: Path | str) -> "YoloDetectionModel":
        """
        Load a model checkpoint into the YOLO detection model using config from the checkpoint.
        """

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading checkpoint from: {checkpoint_path} to {map_location}")
        # TODO [MID]: think if needs to reload if the users wants to run on CPU (but has a CUDA available)
        checkpoint_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        checkpoint_hparams = checkpoint_dict.get("hyper_parameters", {})
        config_from_checkpoint: Config | None = checkpoint_hparams.get("config", None)

        if config_from_checkpoint is None:
            raise ValueError(f"No config found in checkpoint: {checkpoint_path}")

        logger.info(f"Initializing model with config from checkpoint: {Path(checkpoint_path).name}")
        model = cls(config_from_checkpoint)  # This will set model.config

        pretrained_weights_state_dict = checkpoint_dict.get("state_dict", {})
        if not pretrained_weights_state_dict:
            logger.warning(
                f"No state_dict found in checkpoint: {Path(checkpoint_path).name}. "
                "Model will have its initial random weights."
            )
            raise ValueError("No state_dict found in checkpoint")

        model_state_dict = model.state_dict()
        intersected_state_dict = {}

        for key, value_in_checkpoint in pretrained_weights_state_dict.items():
            if key in model_state_dict:
                value_in_model = model_state_dict[key]
                if value_in_model.shape == value_in_checkpoint.shape:
                    intersected_state_dict[key] = value_in_checkpoint
                else:
                    logger.warning(
                        f"Shape mismatch for key '{key}': model has {value_in_model.shape}, "
                        f"checkpoint has {value_in_checkpoint.shape}. Skipping this key."
                    )
            else:
                logger.debug(f"Key '{key}' from checkpoint not found in the initialized model. Skipping.")

        model.load_state_dict(intersected_state_dict, strict=False)

        loaded_count = len(intersected_state_dict)
        model_key_count = len(model_state_dict)
        checkpoint_key_count = len(pretrained_weights_state_dict)

        loaded_percentage_model = (loaded_count / model_key_count * 100) if model_key_count > 0 else 0

        logger.info(
            f"Loaded {loaded_count}/{model_key_count} ({loaded_percentage_model:.2f}%) "
            f"matching keys from pretrained weights in {Path(checkpoint_path).name}."
        )
        if loaded_count < checkpoint_key_count:
            keys_not_in_model = checkpoint_key_count - loaded_count
            logger.warning(
                f"{keys_not_in_model} key(s) from checkpoint were not loaded"
                "(e.g., missing in current model, shape mismatch)."
            )

        return model

    def _calculate_loss(
        self, batch: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor] | None = None
    ) -> DetectionLoss:
        """
        Common step for training, validation, and testing.

        Args:
            batch: Dictionary containing images and labels
            predictions: Dictionary containing model predictions. If None, predictions will be computed
                        from batch["images"]

        Returns:
            DetectionLoss: Object containing total loss and individual loss components (iou, cls, dfl)
        """

        if self.loss_fn is None:
            self.loss_fn = EndToEndDetectionLoss(self.config)

        # NOTE: return_both_paths=True is required for the loss calculation
        if predictions is None:
            predictions = self.forward(batch["images"])
        return self.loss_fn(predictions, batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        detection_loss = self._calculate_loss(batch)

        self.log_dict(
            {
                "loss/total/train": detection_loss.total,
                "loss/iou/train": detection_loss.iou,
                "loss/clf/train": detection_loss.cls,
                "loss/dfl/train": detection_loss.dfl,
            }  # default on_step=True, on_epoch=False
        )

        # Update rolling averages
        self.loss_buffers["loss/total/train"].append(detection_loss.total.item())
        self.loss_buffers["loss/iou/train"].append(detection_loss.iou.item())
        self.loss_buffers["loss/clf/train"].append(detection_loss.cls.item())
        self.loss_buffers["loss/dfl/train"].append(detection_loss.dfl.item())

        # Compute the rolling average for each metric
        avg_train_loss = sum(self.loss_buffers["loss/total/train"]) / len(self.loss_buffers["loss/total/train"])
        avg_train_loss_iou = sum(self.loss_buffers["loss/iou/train"]) / len(self.loss_buffers["loss/iou/train"])
        avg_train_loss_clf = sum(self.loss_buffers["loss/clf/train"]) / len(self.loss_buffers["loss/clf/train"])
        avg_train_loss_dlf = sum(self.loss_buffers["loss/dfl/train"]) / len(self.loss_buffers["loss/dfl/train"])

        # Log the rolling averages to the progress bar
        self.log_dict(
            {
                "loss/total/train_ra": avg_train_loss,
                "loss/iou/train_ra": avg_train_loss_iou,
                "loss/clf/train_ra": avg_train_loss_clf,
                "loss/dfl/train_ra": avg_train_loss_dlf,
            },
            prog_bar=True,
        )

        return detection_loss.total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # Get predictions
        preds_feats_dict = self.forward(batch["images"])

        # Send the batch through the model and calculate the loss
        detection_loss = self._calculate_loss(batch, preds_feats_dict)

        # NOTE: the batch_size is required as there will be aggregation on_epoch=True,
        # lightning need to properly weight the contributions, since the last batch might
        # be of different size
        self.log_dict(
            {
                "loss/total/val": detection_loss.total,  # NOTE: used for EarlyStopping
                "loss/iou/val": detection_loss.iou,
                "loss/clf/val": detection_loss.cls,
                "loss/dfl/val": detection_loss.dfl,
            },  # default on_step=False, on_epoch=True
            batch_size=batch["images"].shape[0],
            sync_dist=True,  # to sync logging across all GPU workers (may have performance impact)
        )

        # Format predictions and targets for mAP calculation
        formatted_predictions = format_predictions(preds_feats_dict["predictions"].detach(), batch)
        formatted_targets = format_targets(batch)

        # Update mAP metric
        self.map_metric.update(formatted_predictions, formatted_targets)

        # Save visualization samples for the validation set
        if batch_idx == 0 and (self.current_epoch % self.vis_interval == 0 or self.current_epoch == 0):
            self._save_detection_visualizations(batch, preds_feats_dict)

        return detection_loss.total

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # Get predictions
        preds_feats_dict = self.forward(batch["images"])

        # Send the batch through the model and calculate the loss
        detection_loss = self._calculate_loss(batch, preds_feats_dict)

        # Log metrics with test/ prefix
        self.log_dict(
            {
                "loss/total/test": detection_loss.total,
                "loss/iou/test": detection_loss.iou,
                "loss/clf/test": detection_loss.cls,
                "loss/dfl/test": detection_loss.dfl,
            },
            batch_size=batch["images"].shape[0],
            sync_dist=True,  # to sync logging across all GPU workers
        )

        # Format predictions and targets for mAP calculation
        formatted_predictions = format_predictions(preds_feats_dict["predictions"].detach(), batch)
        formatted_targets = format_targets(batch)

        # Update mAP metric
        self.map_metric.update(formatted_predictions, formatted_targets)

        return detection_loss.total

    def on_validation_epoch_end(self):
        map_dict = self.map_metric.compute()

        # Log different MAP values
        self.log("val_map", map_dict["map"], on_epoch=True)  # for ModelCheckpoint compatibility
        self.log("map/50-95/val", map_dict["map"], on_epoch=True)
        self.log("map/50/val", map_dict["map_50"], on_epoch=True)
        self.log("map/75/val", map_dict["map_75"], on_epoch=True)
        self.log("map/small/val", map_dict["map_small"], on_epoch=True)
        self.log("map/medium/val", map_dict["map_medium"], on_epoch=True)
        self.log("map/large/val", map_dict["map_large"], on_epoch=True)

        # Get the current values from the logged metrics
        val_loss = self.trainer.callback_metrics.get("loss/total/val")
        val_loss_iou = self.trainer.callback_metrics.get("loss/iou/val")
        val_loss_clf = self.trainer.callback_metrics.get("loss/clf/val")
        val_loss_dlf = self.trainer.callback_metrics.get("loss/dfl/val")

        # Print them in a formatted way
        print("Validation Epoch End:")
        print(f"val_loss: {val_loss:.3f}")
        print(f"Losses => IoU:   {val_loss_iou:2.3f} | Clf: {val_loss_clf:2.3f} | Dlf: {val_loss_dlf:2.3f}")
        print(
            f"mAP    => Total: {map_dict['map']:2.3f} | @50: {map_dict['map_50']:2.3f} | @75: {map_dict['map_75']:2.3f}"
        )

        # Reset metric states at the end of the epoch
        self.map_metric.reset()  # TODO [LOW]: figure out if required

    def on_test_epoch_end(self):
        map_dict = self.map_metric.compute()

        # Log different MAP values
        self.log("map/50-95/test", map_dict["map"], on_epoch=True)
        self.log("map/50/test", map_dict["map_50"], on_epoch=True)
        self.log("map/75/test", map_dict["map_75"], on_epoch=True)
        self.log("map/small/test", map_dict["map_small"], on_epoch=True)
        self.log("map/medium/test", map_dict["map_medium"], on_epoch=True)
        self.log("map/large/test", map_dict["map_large"], on_epoch=True)

        # Get the current values from the logged metrics
        test_loss = self.trainer.callback_metrics.get("loss/total/test")
        test_loss_iou = self.trainer.callback_metrics.get("loss/iou/test")
        test_loss_clf = self.trainer.callback_metrics.get("loss/clf/test")
        test_loss_dlf = self.trainer.callback_metrics.get("loss/dfl/test")

        # Print them in a formatted way
        print("Test Epoch End:")
        print(f"test_loss: {test_loss:.3f}")
        print(f"Losses => IoU:   {test_loss_iou:2.3f} | Clf: {test_loss_clf:2.3f} | Dlf: {test_loss_dlf:2.3f}")
        print(
            f"mAP    => Total: {map_dict['map']:2.3f} | @50: {map_dict['map_50']:2.3f} | @75: {map_dict['map_75']:2.3f}"
        )

        # Reset metric states at the end of the epoch
        self.map_metric.reset()  # TODO [LOW]: figure out if required

    def configure_optimizers(self):
        # Calculate necessary constants
        # Adjust steps_per_epoch based on overfit_batches
        if isinstance(self.trainer.overfit_batches, int):
            self.steps_per_epoch = self.trainer.overfit_batches
        elif isinstance(self.trainer.overfit_batches, float) and self.trainer.overfit_batches > 0.0:
            # Calculate based on a fraction of the dataset
            self.steps_per_epoch = int(self.trainer.estimated_stepping_batches * self.trainer.overfit_batches)
        else:
            # NOTE: done with "estimated_stepping batches" as
            # self.trainer.train_dataloader is not accessible here
            self.steps_per_epoch = self.trainer.estimated_stepping_batches
        self.warmup_steps = self.config.train.scheduler.warmup_epochs * self.steps_per_epoch
        self.total_steps = self.trainer.max_epochs * self.steps_per_epoch

        # Get all BatchNorm layers
        bn_types = tuple(v for k, v in get_block_name_to_impl_dict().items() if "Norm" in k)

        # Split parameters into two groups
        params_decay = []
        params_no_decay = []

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                # BatchNorm parameters and biases don't need weight decay
                if isinstance(module, bn_types) or "bias" in module_name + "." + param_name:
                    params_no_decay.append(param)
                else:
                    params_decay.append(param)

        # Create optimizer with distinct weight decay groups (https://arxiv.org/abs/1711.05101)
        optimizer = torch.optim.AdamW(
            [
                {"params": params_no_decay, "weight_decay": 0.0},  # No decay for BN/bias
                {
                    "params": params_decay,
                    "weight_decay": self.config.train.optimizer.args["weight_decay"],
                },  # Apply decay to other weights
            ],
            lr=self.config.train.optimizer.args["max_lr"],
            betas=(0.9, 0.999),  # https://arxiv.org/abs/1412.6980
        )

        # --- Schedulers ---
        # Scheduler 1: Linear Warmup
        # Gradually increases LR from near-zero to `max_lr` over `warmup_steps`.
        # Helps stabilize training in early stages.
        # Ref: He et al. (ResNet paper) popularized warmup. https://arxiv.org/abs/1512.03385
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.config.train.scheduler.args["start_factor"] + 1e-12,  # Start near zero
            end_factor=self.config.train.scheduler.args["warmup_factor"],  # Reach 1.0 * max_lr
            total_iters=self.warmup_steps,  # Duration of warmup
        )

        # Scheduler 2: Linear Decay
        # Linearly decreases LR from `max_lr` down to near-zero over the remaining steps.
        # Allows larger updates early and finer adjustments later. Cosine decay is another common alternative.
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=self.config.train.scheduler.args["warmup_factor"],  # Start from 1.0 * max_lr (end of warmup)
            end_factor=self.config.train.scheduler.args["end_factor"],  # End at 0.01 * max_lr
            total_iters=self.total_steps - self.warmup_steps,  # Duration of decay
        )

        # Combine schedulers: Warmup followed by Decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.warmup_steps],  # Switch after warmup
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",  # Update LR at each step
                "frequency": 1,
                "name": "SequentialLR",
            }
        ]

    def on_train_start(self):
        self.current_stage = "train"

    def on_validation_start(self):
        self.current_stage = "validate"

    def on_test_start(self):
        self.current_stage = "test"

    def on_predict_start(self):
        self.current_stage = "inference"

    @property
    def current_stage(self) -> str:
        return self._current_stage

    @current_stage.setter
    def current_stage(self, stage: Literal["train", "validate", "test", "inference"]) -> None:
        self._current_stage = stage
        # Propagate stage to v10Detect head if present
        if (
            hasattr(self, "blocks")
            and isinstance(self.blocks, nn.ModuleList)
            and len(self.blocks) > 0
            and isinstance(self.blocks[-1], v10Detect)
        ):
            self.blocks[-1].current_stage = stage

    def update_num_classes(self, num_classes: int) -> None:
        """
        Updates the model's detection head for a new number of classes.

        This method optimizes transfer learning when adapting a pretrained model to a
        dataset with a different number of classes. It:
        1. Updates the detection head using its weight-preserving implementation
        2. Updates the config to reflect the new class count
        3. Resets the loss function

        Args:
            num_classes: The new number of classes
        """
        head = self.blocks[-1]
        if not isinstance(head, v10Detect):
            raise ValueError("Last block is not a v10Detect detection head. Cannot update number of classes.")

        # Store the original number of classes for logging
        original_num_classes = head.num_classes

        if original_num_classes == num_classes:
            logger.info(f"Number of classes is already {num_classes}. No update needed.")
            return

        logger.info(f"Updating number of classes from {original_num_classes} to {num_classes}")

        # Update the detection head (preserving weights where possible)
        head.update_num_classes(num_classes)

        # Clear out the loss function to force a re-initialization
        self.loss_fn = None

        logger.info(f"Successfully updated model for {num_classes} classes")

    def _save_detection_visualizations(
        self, batch: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor]
    ) -> None:
        """
        Save visualization of detection results during training or validation.

        Args:
            batch: Dictionary containing images and ground truth
            predictions: Dictionary containing model predictions
        """
        # TODO [LOW]: when training with multiple GPUs, this will overwrite the same files, think how to handle this
        if not self.experiment_dir:
            logger.warning("Experiment directory not set. Skipping visualization.")
            return

        # Create output directory if it doesn't exist
        output_dir = self.experiment_dir / "visualizations" / f"epoch_{self.current_epoch:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which samples to visualize
        batch_size = batch["images"].shape[0]

        # Initialize or update visualization sample indices
        if not self.vis_batch_indices:
            self.vis_batch_indices = list(range(min(batch_size, self.vis_samples_per_batch)))

        # Get class labels from the config if available
        class_labels = getattr(self.config.dataset, "names", None)

        # Process each selected sample
        for idx in self.vis_batch_indices:
            if idx >= batch_size:
                continue

            # Get the image and its predictions
            image = batch["images"][idx].cpu().permute(1, 2, 0).numpy()  # HWC format
            image = (image * 255).astype(np.uint8)

            # Get predictions for this image
            image_preds = predictions["predictions"][idx]

            # Create coordinate mapper - assumes input image has been resized/padded
            h, w = image.shape[:2]
            img_mapper = ImageCoordinateMapper(
                original_width=w,
                original_height=h,
                transformed_width=w,
                transformed_height=h,
                scale_x=1.0,
                scale_y=1.0,
                padding_x=0,
                padding_y=0,
            )

            # Create inference result object
            result = InferenceResult(
                model_output=image_preds,
                original_image=image,
                confidence_th=self.vis_confidence_threshold,
                img_coordinate_mapper=img_mapper,
                class_labels=class_labels,
            )

            # Save the annotated image
            output_path = output_dir / f"sample_{idx:02d}.jpg"
            result.save(output_path)

            # Log the image to TensorBoard if available
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_image"):
                try:
                    # Convert back to tensor for TensorBoard (CHW format)
                    annotated_img = result.annotate_image()
                    annotated_tensor = torch.from_numpy(annotated_img).permute(2, 0, 1) / 255.0

                    # Add to TensorBoard
                    self.logger.experiment.add_image(
                        f"{self.current_stage}/detections/epoch={self.current_epoch}_sample={idx:02d}", annotated_tensor
                    )
                except Exception as e:
                    logger.warning(f"Failed to log image to TensorBoard: {e}")


def model_arch_from_config(config: Config) -> tuple[nn.ModuleList, list[int]]:
    """
    Constructs YOLO model architecture from configuration, handling layer creation,
    skip connections, and feature pyramid setup.

    Args:
        config (Config): Model configuration with architecture specs, scaling factors,
                         and channel constraints [NOT MODIFIED].

    Returns:
        tuple: (blocks, save_indices)
            - blocks (nn.ModuleList): Model layers/blocks
            - save_indices (list[int]): Indices of layers needed for skip connections

    Note:
        Handles special cases like detection blocks, multi-source inputs, and channel
        scaling. Source indices of -1 refer to the previous layer's output.
    """

    logger.debug("\nLayer Structure:")
    logger.debug(f"{'Index':>5} {'Source':>10} {'Params':>10}  {'Type':<20} {'Arguments'}")

    # Get mapping of block names to their implementations
    block_name_to_impl_dict = get_block_name_to_impl_dict()

    # Track channel output dimensions for each block
    out_channels_list: list[int] = []
    blocks = nn.ModuleList()
    save_indices: list[int] = []

    # Combine backbone and head configurations
    model_sections: list[dict[str, BlockConfig]] = []
    for _section_name, section_blocks in config.model.architecture.items():
        model_sections.extend(section_blocks)

    for i, block_dict in enumerate(model_sections):
        # Get block implementation from name
        # NOTE: each block config is a dict with a single key-value pair
        block_name: str = next(iter(block_dict.keys()))
        block_config: BlockConfig = next(iter(block_dict.values()))

        # Get block implementation class
        if block_name not in block_name_to_impl_dict:
            raise ValueError(f"Unknown block type: {block_name}")
        block_class = block_name_to_impl_dict[block_name]

        # Process source blocks
        source_blocks = block_config.source
        # Sum channels for concatenation
        if not out_channels_list:
            # NOTE: only used for first block
            in_channels = block_config.args["in_channels"]
        else:
            in_channels = sum(out_channels_list[src] for src in source_blocks)

        # NOTE: copy to avoid modifying the original config, if we modify the original config,
        # when loading the config from checkpoint we would be applying the scaling factors again
        block_args = block_config.args.copy()

        # Create block instance
        if block_name in ["Detect", "v10Detect"]:
            # Special handling for detection blocks
            block_args["in_channels_list"] = [out_channels_list[idx] for idx in source_blocks]
        elif "in_channels" in inspect.signature(block_class).parameters:
            # Check if block class requires in_channels parameter
            block_args["in_channels"] = in_channels
            tmp_out_channels = min(block_args["out_channels"], config.model.max_channels) * config.model.channels_scale
            out_channels = round_to_multiple(tmp_out_channels, multiple=8)
            if abs(tmp_out_channels - out_channels) / tmp_out_channels > 0.02:
                logger.warning(f"Rounded out_channels from {tmp_out_channels} to {out_channels}")
            block_args["out_channels"] = out_channels
        else:
            # Not head (v10Detect) nor has in_channels parameter
            # Examples: nn.Upsample, Concat
            out_channels = in_channels

        # Scale repeats if present, with minimum of 1 (C2f, C2fCIB)
        if "repeats" in block_args:
            block_args["repeats"] = max(round(block_args["repeats"] * config.model.repeats_scale), 1)

        block = block_class(**block_args)

        # Attach metadata (to nn.Module)
        block.index = i
        block.source = source_blocks

        # Print block information
        num_params = sum(p.numel() for p in block.parameters())
        logger.debug(f"{i:>5} {str(source_blocks):>10} {num_params:>10,d}  {block_name:<20} {str(block_args)}")

        # Track if this block's output should be saved
        save_indices.extend([src for src in source_blocks if src != -1])
        blocks.append(block)
        out_channels_list.append(out_channels)

    # Make save_indices unique and sorted
    save_indices = sorted(list(set(save_indices)))

    return blocks, save_indices


def format_predictions(batch_predictions: torch.Tensor, batch: dict[str : torch.Tensor]) -> dict[str : torch.Tensor]:
    """
    Format model predictions for mAP calculation (torchmetrics), the boxes:
    from inference shape xyxy (not normalized) to xyxy normalized
    """
    height, width = batch["images"].shape[-2:]
    whwh = torch.tensor([width, height, width, height], device=batch_predictions.device)  # to normalize boxes
    # Format model predictions for mAP calculation
    # each dictionary corresponds to a single image
    formatted_batch_predictions = []
    for img_predictions in batch_predictions:
        formatted_batch_predictions.append(
            {
                "boxes": img_predictions[:, :4] / whwh,  # xyxy_norm
                "scores": img_predictions[:, 4],  # order verified
                "labels": img_predictions[:, 5].int(),
            }
        )
    return formatted_batch_predictions


def format_targets(batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """
    Convert collate_fn output to torchmetrics object detection format.

    Args:
        batch: dict with keys:
            - "images": tensor (bs, c, h, w)
            - "boxes": tensor (bs, max_boxes, 4) in xyxy_norm
            - "labels": tensor (bs, max_boxes, 1)

    Returns:
        List of dicts (one per image) each containing:
            - "boxes": tensor (N, 4) in xyxy_norm
            - "labels": tensor (N,)
    """
    boxes = batch["boxes"]  # (bs,  max_boxes, 4)
    labels = batch["labels"].squeeze(-1)  # (bs,  max_boxes)
    bs, _, _ = boxes.shape

    results = []
    for i in range(bs):
        # Remove padded boxes (assumed padded boxes have all zeros)
        valid = boxes[i].sum(dim=1) != 0
        xyxy_norm_boxes = boxes[i][valid]
        valid_labels = labels[i][valid]
        if xyxy_norm_boxes.numel() == 0:
            results.append(
                {
                    "boxes": torch.empty((0, 4), device=boxes.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=boxes.device),
                }
            )
        else:
            results.append({"boxes": xyxy_norm_boxes, "labels": valid_labels})
    return results


if __name__ == "__main__":
    from angelcv.config import ConfigManager

    config = ConfigManager.upsert_config(model_file="yolov10n.yaml", dataset_file="coco.yaml")
    model = YoloDetectionModel(config)
    model.eval()

    # Calculate strides using sample forward pass
    sample_size = 640
    sample_batch = torch.zeros(1, 3, sample_size, sample_size)
    output = model.forward(sample_batch)
    print(output)
