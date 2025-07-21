from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from angelcv.config.config_registry import Config
from angelcv.dataset.custom_transforms import MosaicFromDataset

# NOTE: Using (114, 114, 114) because 114/255 = 0.447, which is similar to the mean pixel value of ImageNet.
# This background color is used in augmentation for many Computer Vision models, including YOLO.
AUGMENTATION_BG_COLOR = (114, 114, 114)
# TODO [LOW]: test with (124, 116, 103), average value pixel of ImageNet (RGB)


def build_training_transforms(config: Config, dataset: Dataset = None) -> Callable:
    """
    Build training data transformations based on configuration.

    Args:
        config: Full configuration object containing training parameters
        dataset: Dataset instance for mosaic augmentation (optional)

    Returns:
        Callable: Composed albumentations transforms for training
    """
    max_size = config.train.data.image_size

    # NOTE: doesn't seem necessary to normalize the images with ImageNet values
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # simply dividing by 255
    # TODO [MID]: set the probabilities and key parameters from the config
    return A.Compose(
        transforms=[
            # -----------------  MOSAIC TRANSFORMS ------------------
            A.OneOrOther(
                first=A.Sequential(  # MosaicFromDatase + Affine
                    p=1.0,
                    transforms=[
                        # NOTE: Mosaic augmentation creates a canvas of size (cell_shape * grid_yx), then crops a
                        # random region of size target_size. target_size needs to be larger than max_size to avoid
                        # losing resolution if zooming.
                        # If target_size = cell_shape * grid_yx, then the mosaic will be a perfect square.
                        MosaicFromDataset(
                            p=1.0,
                            dataset=dataset,  # Getting images from the entire dataset
                            target_size=(int(max_size * 2), int(max_size * 2)),  # Size of random crop of canvas
                            cell_shape=(max_size, max_size),  # Space allocated for each cell
                            fill=AUGMENTATION_BG_COLOR,
                        ),
                        # NOTE: Affine before LongestMaxSize to not lose resolution in case of zooming
                        A.Affine(
                            p=1.0,
                            rotate=0,
                            translate_percent=(-0.35, 0.35),
                            scale=(0.95, 1.6),  # Not much zoom out as already zoomed out by mosaic
                            shear=0,
                            fill=AUGMENTATION_BG_COLOR,
                        ),
                    ],
                ),
                second=A.Affine(  # No Mosaic
                    p=1.0,
                    rotate=0,
                    translate_percent=(-0.1, 0.1),
                    scale=(0.7, 1.3),
                    shear=0,
                    fill=AUGMENTATION_BG_COLOR,
                ),
                p=0.0,  # probability of using Mosaic
            ),
            # -----------------  RESIZE TRANSFORMS ------------------
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, fill=AUGMENTATION_BG_COLOR),
            # -------------- TRANSFORMS WITHOUT RESIZE --------------
            # NOTE: high val_shift_limit range to simulate different lighting conditions
            # NOTE: high values of sat_shift_limit range introduce artifacts in the images
            A.HueSaturationValue(
                p=0.8, hue_shift_limit=(-20, 20), sat_shift_limit=(-50, 30), val_shift_limit=(-70, 70)
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=(3, 7)),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.ImageCompression(quality_range=(70, 90)),
                ],
                p=0.05,
            ),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.01),
            # ----------------- FORMAT TRANSFORMS -----------------
            A.Normalize(mean=0, std=1, max_pixel_value=255),  # This divides by 255
            ToTensorV2(),
        ],
        # NOTE: min_* to filter out tiny boxes, since those are not useful for training
        bbox_params=A.BboxParams(
            format="albumentations", label_fields=["labels"], min_width=4, min_height=4, min_area=20
        ),
    )


def build_val_transforms(config: Config) -> Callable:
    """
    Build validation/test data transformations based on configuration.

    Args:
        config: Full configuration object containing validation parameters

    Returns:
        Callable: Composed albumentations transforms for validation/testing
    """
    max_size = config.validation.data.image_size

    # NOTE: doens't seem necessary to normalize the iamges with ImageNet values
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # simply dividing by 255
    return A.Compose(
        transforms=[
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, fill=AUGMENTATION_BG_COLOR),
            A.Normalize(mean=0, std=1, max_pixel_value=255),  # This divides by 255
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
    )


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import torch
    import torchvision.utils as vutils

    from angelcv.config import ConfigManager
    from angelcv.dataset.coco_datamodule import CocoDataModule
    from angelcv.utils.annotation_utils import generate_distinct_colors

    # Parameters to control the script's output
    PLOT_IMAGES = False  # Set to True to display images with bounding boxes
    PRINT_BBOXES = True  # Set to True to print bounding box details to the terminal
    SMALL_BBOX_AREA_THRESHOLD = 5 * 5  # Area in pixels to trigger a warning

    def draw_bboxes_on_image(image_tensor, boxes_tensor, labels_tensor, colors):
        """Draw bounding boxes on a single image tensor."""
        # Convert tensor to numpy (C, H, W) -> (H, W, C)
        image_np = (image_tensor * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        h, w = image_np.shape[:2]

        # Draw bboxes
        for box, label in zip(boxes_tensor, labels_tensor):
            # Skip padding (zeros)
            if torch.all(box == 0):
                continue

            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # Get color for this class
            class_id = int(label.item()) if hasattr(label, "item") else int(label)
            color = colors[class_id % len(colors)]

            # Draw rectangle (OpenCV uses BGR format)
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color[::-1], 2)

        # Convert back to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

    # Create config element
    config = ConfigManager.upsert_config(dataset_file="coco.yaml")
    config.train.data.batch_size = 8

    # Create CocoDataModule with default transforms (val transforms have no augmentation)
    datamodule = CocoDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("validate")  # only create val dataset

    # Use validation loader (which has no augmentations)
    val_loader = datamodule.val_dataloader()

    # Create augmentation transforms to apply manually
    augmentation_transforms = build_training_transforms(config, datamodule.val_dataset)

    # Generate colors for classes
    num_classes = len(config.dataset.names)
    colors = generate_distinct_colors(num_classes)

    n_samples = 9e9
    for i, batch in enumerate(val_loader):
        images_original = batch["images"]  # Shape: (B, C, H, W) - no augmentations applied
        boxes_original = batch["boxes"]  # Shape: (B, max_boxes, 4)
        labels_original = batch["labels"].squeeze(-1)  # Shape: (B, max_boxes)

        # Convert back to numpy for augmentation (reverse the ToTensorV2 and normalization)
        # The images are normalized with mean=0, std=1, max_pixel_value=255
        # So we need to denormalize: pixel_value = (normalized_value * 255)
        images_numpy = (images_original * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        # Apply augmentations to each image in the batch
        augmented_images = []
        augmented_boxes_list = []
        augmented_labels_list = []

        for img_idx in range(images_numpy.shape[0]):
            img = images_numpy[img_idx]  # Shape: (H, W, C)
            h_orig, w_orig, _ = img.shape

            # Get original bboxes and labels for this image (filter out padding)
            orig_boxes = boxes_original[img_idx]
            orig_labels = labels_original[img_idx]

            # Filter out padding (boxes with all zeros)
            valid_mask = torch.any(orig_boxes != 0, dim=1)
            valid_boxes = orig_boxes[valid_mask]
            valid_labels = orig_labels[valid_mask].cpu().numpy()

            # Apply augmentations with bboxes
            augmented = augmentation_transforms(image=img, bboxes=valid_boxes.cpu().numpy(), labels=valid_labels)
            augmented_img = augmented["image"]  # This will be a tensor
            augmented_boxes = torch.tensor(augmented["bboxes"])
            augmented_labels = torch.tensor(augmented["labels"])

            if PRINT_BBOXES and len(augmented_boxes) > 0:
                _, h_aug, w_aug = augmented_img.shape
                print(f"--- Augmented Image {img_idx} (shape: {h_aug}x{w_aug}) ---")
                for j, box in enumerate(augmented_boxes):
                    x1, y1, x2, y2 = box.tolist()
                    abs_x1, abs_y1, abs_x2, abs_y2 = x1 * w_aug, y1 * h_aug, x2 * w_aug, y2 * h_aug
                    area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                    print(
                        f"  Box {j}: [x1={abs_x1:.1f}, y1={abs_y1:.1f}, x2={abs_x2:.1f}, y2={abs_y2:.1f}], "
                        f"Area: {area:.1f}"
                    )
                    width = abs_x2 - abs_x1
                    height = abs_y2 - abs_y2
                    if area < SMALL_BBOX_AREA_THRESHOLD:
                        print(f"  [WARNING] Augmented box is very small, width {width:.1f}, height {height:.1f}")

            augmented_images.append(augmented_img)
            augmented_boxes_list.append(augmented_boxes)
            augmented_labels_list.append(augmented_labels)

        # Stack augmented images back to batch
        images_augmented = torch.stack(augmented_images)

        # Draw bounding boxes on original images (no augmentations)
        images_original_with_boxes = []
        for img_idx in range(images_original.shape[0]):
            img_with_boxes = draw_bboxes_on_image(
                images_original[img_idx], boxes_original[img_idx], labels_original[img_idx], colors
            )
            images_original_with_boxes.append(img_with_boxes)
        images_original_with_boxes = torch.stack(images_original_with_boxes)

        # Draw bounding boxes on augmented images
        images_augmented_with_boxes = []
        for img_idx in range(images_augmented.shape[0]):
            # Pad augmented boxes to match original format if needed
            aug_boxes = augmented_boxes_list[img_idx]
            aug_labels = augmented_labels_list[img_idx]

            # Create padded tensors
            max_boxes = boxes_original.shape[1]
            padded_boxes = torch.zeros((max_boxes, 4))
            padded_labels = torch.zeros(max_boxes)

            if len(aug_boxes) > 0:
                num_valid = min(len(aug_boxes), max_boxes)
                padded_boxes[:num_valid] = aug_boxes[:num_valid]
                padded_labels[:num_valid] = aug_labels[:num_valid]

            img_with_boxes = draw_bboxes_on_image(images_augmented[img_idx], padded_boxes, padded_labels, colors)
            images_augmented_with_boxes.append(img_with_boxes)
        images_augmented_with_boxes = torch.stack(images_augmented_with_boxes)

        # Create side-by-side comparison
        if PLOT_IMAGES:
            fig, axes = plt.subplots(1, 2, figsize=(15, 10))

            # Original images grid with bboxes (from validation set - no augmentation)
            grid_original = vutils.make_grid(images_original_with_boxes, nrow=4, normalize=True, scale_each=True)
            axes[0].imshow(grid_original.permute(1, 2, 0).cpu().numpy())
            axes[0].set_title(f"Batch {i} - Original (Validation - No Augmentation) - shape: {images_original.shape}")
            axes[0].axis("off")

            # Augmented images grid with bboxes (manually applied training augmentations)
            grid_augmented = vutils.make_grid(images_augmented_with_boxes, nrow=4, normalize=True, scale_each=True)
            axes[1].imshow(grid_augmented.permute(1, 2, 0).cpu().numpy())
            axes[1].set_title(f"Batch {i} - With Training Augmentation - shape: {images_augmented.shape}")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

        if i >= n_samples:
            break
