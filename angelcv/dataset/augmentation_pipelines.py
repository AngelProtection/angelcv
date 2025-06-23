from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from angelcv.dataset.custom_transforms import MosaicFromDataset


# TODO [MID]: use the config element to setup the augmentation parameters
def default_train_transforms(max_size: int = 640, dataset: Dataset = None) -> Callable:
    """
    Default training data transformations.
    """
    # NOTE: doesn't seem necessary to normalize the images with ImageNet values
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # simply dividing by 255
    # TODO [MID]: implement mosaic augmentation, not trivial with albumentations framework
    return A.Compose(
        transforms=[
            # NOTE: mosaic augmentation needs to be before the LongestMaxSize, PadIfNeeded, and Normalize transforms
            MosaicFromDataset(dataset=dataset, target_size=(max_size, max_size), p=1.0),  # TODO [LOW]: check
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max_size, min_width=max_size),
            # ---------------- START AUGMENTATION ----------------
            A.Affine(p=0.5, rotate=(0, 15), translate_percent=0.1, scale=(0.5, 1.5), shear=(0, 10)),
            A.HueSaturationValue(p=0.8, hue_shift_limit=5, sat_shift_limit=175, val_shift_limit=100),
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
            # ----------------- END AUGMENTATION -----------------
            A.Normalize(mean=0, std=1, max_pixel_value=255),  # This divides by 255
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
    )


def default_val_transforms(max_size: int = 640) -> Callable:
    """
    Default validation/test data transformations.
    """
    # NOTE: doens't seem necessary to normalize the iamges with ImageNet values
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # simply dividing by 255
    return A.Compose(
        transforms=[
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max_size, min_width=max_size),
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

    # Create CocoDataModule with train_transforms set to val_transforms (no augmentations)
    datamodule = CocoDataModule(
        config,
        train_transforms=default_val_transforms(
            max_size=config.train.data.image_size
        ),  # Use val transforms for train too
        val_transforms=default_val_transforms(max_size=config.train.data.image_size),
    )
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Create augmentation transforms to apply manually
    augmentation_transforms = default_train_transforms(
        max_size=config.train.data.image_size, dataset=datamodule.train_dataset
    )

    # Generate colors for classes
    num_classes = len(config.dataset.names)
    colors = generate_distinct_colors(num_classes)

    n_samples = 50
    for i, batch in enumerate(train_loader):
        images_original = batch["images"]  # Shape: (B, C, H, W)
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

            # Get original bboxes and labels for this image (filter out padding)
            orig_boxes = boxes_original[img_idx]
            orig_labels = labels_original[img_idx]

            # Filter out padding (boxes with all zeros)
            valid_mask = torch.any(orig_boxes != 0, dim=1)
            valid_boxes = orig_boxes[valid_mask].cpu().numpy()
            valid_labels = orig_labels[valid_mask].cpu().numpy()

            # Apply augmentations with bboxes
            augmented = augmentation_transforms(image=img, bboxes=valid_boxes, labels=valid_labels)
            augmented_img = augmented["image"]  # This will be a tensor
            augmented_boxes = torch.tensor(augmented["bboxes"])
            augmented_labels = torch.tensor(augmented["labels"])

            augmented_images.append(augmented_img)
            augmented_boxes_list.append(augmented_boxes)
            augmented_labels_list.append(augmented_labels)

        # Stack augmented images back to batch
        images_augmented = torch.stack(augmented_images)

        # Draw bounding boxes on original images
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Original images grid with bboxes
        grid_original = vutils.make_grid(images_original_with_boxes, nrow=4, normalize=True, scale_each=True)
        axes[0].imshow(grid_original.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title(f"Batch {i} - Original (No Augmentation) - shape: {images_original.shape}")
        axes[0].axis("off")

        # Augmented images grid with bboxes
        grid_augmented = vutils.make_grid(images_augmented_with_boxes, nrow=4, normalize=True, scale_each=True)
        axes[1].imshow(grid_augmented.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title(f"Batch {i} - With Augmentation - shape: {images_augmented.shape}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        if i >= n_samples:
            break
