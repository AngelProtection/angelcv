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
    coco_dm = CocoDataModule(config)
    coco_dm.prepare_data()
    coco_dm.setup()

    train_loader = coco_dm.train_dataloader()
    val_loader = coco_dm.val_dataloader()

    n_samples = 10
    for i, batch in enumerate(train_loader):
        images = batch["images"]
        # If images is (B, C, H, W), make a grid
        if isinstance(images, torch.Tensor):
            grid = vutils.make_grid(images, nrow=4, normalize=True, scale_each=True)
            plt.figure(figsize=(12, 8))
            plt.title(f"Batch {i} - shape: {images.shape}")
            plt.axis("off")
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()
        else:
            print(f"Batch {i}: {images.shape} (not a torch.Tensor, cannot display grid)")
        if i >= n_samples:
            break
