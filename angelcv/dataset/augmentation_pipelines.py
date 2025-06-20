from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2


# TODO [MID]: use the config element to setup the augmentation parameters
def default_train_transforms(max_size: int = 640) -> Callable:
    """
    Default training data transformations.
    """
    # NOTE: doesn't seem necessary to normalize the images with ImageNet values
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # simply dividing by 255
    # TODO [MID]: implement mosaic augmentation, not trivial with albumentations framework
    return A.Compose(
        transforms=[
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(min_height=max_size, min_width=max_size),
            # ---------------- START AUGMENTATION ----------------
            A.Affine(p=0.2, rotate=(-30, 30), shear=(-10, 10), scale=(0.8, 1.2), translate_percent=(0.1, 0.2)),
            A.OneOf(
                [
                    A.Blur(blur_limit=(3, 7)),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=(3, 7)),
                ],
                p=0.3,
            ),  # 30% chance of applying one of these blur operations
            A.ToGray(p=0.1),
            A.CLAHE(p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.RandomGamma(p=0.1),
            # A.ImageCompression(quality_range=(70, 90), p=0.2),  # TODO: uncomment (training server issues)
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
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
    import matplotlib.pyplot as plt
    import torch
    import torchvision.utils as vutils

    from angelcv.config import ConfigManager
    from angelcv.dataset.coco_datamodule import CocoDataModule

    config = ConfigManager.upsert_config(dataset_file="coco.yaml")

    # Create CocoDataModule with train_transforms set to val_transforms (no augmentations)
    coco_dm = CocoDataModule(
        config,
        train_transforms=default_val_transforms(
            max_size=config.train.data.image_size
        ),  # Use val transforms for train too
        val_transforms=default_val_transforms(max_size=config.train.data.image_size),
    )
    coco_dm.prepare_data()
    coco_dm.setup()

    train_loader = coco_dm.train_dataloader()
    val_loader = coco_dm.val_dataloader()

    # Create augmentation transforms to apply manually
    augmentation_transforms = default_train_transforms(max_size=config.train.data.image_size)

    n_samples = 50
    for i, batch in enumerate(train_loader):
        images_original = batch["images"]  # Shape: (B, C, H, W)

        # Convert back to numpy for augmentation (reverse the ToTensorV2 and normalization)
        # The images are normalized with mean=0, std=1, max_pixel_value=255
        # So we need to denormalize: pixel_value = (normalized_value * 255)
        images_numpy = (images_original * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        # Apply augmentations to each image in the batch
        augmented_images = []
        for img_idx in range(images_numpy.shape[0]):
            img = images_numpy[img_idx]  # Shape: (H, W, C)

            # Apply augmentations (we don't need bboxes for visualization)
            augmented = augmentation_transforms(image=img, bboxes=[], labels=[])
            augmented_img = augmented["image"]  # This will be a tensor
            augmented_images.append(augmented_img)

        # Stack augmented images back to batch
        images_augmented = torch.stack(augmented_images)

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Original images grid
        grid_original = vutils.make_grid(images_original, nrow=4, normalize=True, scale_each=True)
        axes[0].imshow(grid_original.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title(f"Batch {i} - Original (No Augmentation) - shape: {images_original.shape}")
        axes[0].axis("off")

        # Augmented images grid
        grid_augmented = vutils.make_grid(images_augmented, nrow=4, normalize=True, scale_each=True)
        axes[1].imshow(grid_augmented.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title(f"Batch {i} - With Augmentation - shape: {images_augmented.shape}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        if i >= n_samples:
            break
