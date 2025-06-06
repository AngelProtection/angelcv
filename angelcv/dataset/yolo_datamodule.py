import os
from pathlib import Path
from typing import Callable

import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from angelcv.config.config_registry import Config
from angelcv.dataset.augmentation import default_train_transforms, default_val_transforms
from angelcv.utils.logging_manager import get_logger

logger = get_logger(__name__)


class YOLODetectionDataset(Dataset):
    """
    Dataset class for YOLO-format object detection.
    Expects a directory containing images and, for each image, a corresponding
    annotation text file located in a parallel labels directory.

    Each annotation file should have one line per object with the format:
        class_id cx cy w h
    where (cx, cy, w, h) are normalized values. The dataset converts these into
    normalized (x1, y1, x2, y2) bounding boxes.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        classes: dict[int, str],
        transforms: Callable | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = classes  # e.g. {0: "person", 1: "car", ...}
        self.transforms = transforms

        # Allowed image extensions; adjust if necessary
        allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_files = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in allowed_exts])

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_image(self, image_path: Path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, image_path: Path) -> tuple[list, list]:
        """
        Loads and converts annotations from a YOLO-format text file.
        Returns:
            A tuple (bboxes, labels) where:
            - bboxes is a list of [x1, y1, x2, y2] in normalized format.
            - labels is a list of integer class IDs.
        """
        # Compute the relative path from the images directory to the image.
        relative_path = image_path.relative_to(self.images_dir)
        label_path = (self.labels_dir / relative_path).with_suffix(".txt")
        eps = np.finfo(np.float32).eps

        if label_path.exists():
            try:
                data = np.loadtxt(label_path, ndmin=2)
            except Exception:
                data = np.empty((0, 5))
            if data.size == 0:
                return [], []
            # Each row in data is: [class_id, cx, cy, w, h]
            labels = data[:, 0].astype(int)
            cxcy = data[:, 1:3]
            wh = data[:, 3:5]

            # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
            x1y1 = cxcy - wh / 2.0
            x2y2 = cxcy + wh / 2.0
            boxes_xyxy = np.concatenate([x1y1, x2y2], axis=1)
            # Check if any values are outside valid range and warn if needed
            if (boxes_xyxy > 1).any() or (boxes_xyxy <= 0.0).any():
                logger.warning(
                    f"Found invalid normalized coordinates in {image_path.name} - "
                    f"values outside range (0, 1]. Clipping to valid range."
                )
                # Clip to ensure the normalized values lie within (eps, 1)
                boxes_xyxy = np.clip(boxes_xyxy, a_min=eps, a_max=1)

            return boxes_xyxy.tolist(), labels.tolist()
        else:
            # logger.info(f"No label file found for image {image_path}, using as background image.")
            return [], []

    def __getitem__(self, index: int):
        image_path = self.image_files[index]
        image = self._load_image(image_path)
        bboxes, labels = self._load_target(image_path)

        if self.transforms:
            # Expecting transforms to work with a dict with keys: 'image', 'bboxes', 'labels'
            transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Reassemble the target list as a list of dictionaries.
        target = []
        for bbox, label in zip(bboxes, labels):
            target.append(
                {
                    "image_id": index,
                    "bbox": bbox,  # Normalized (x1, y1, x2, y2)
                    "class_id": label,  # Integer class label
                    "filename": image_path.name,  # unused
                }
            )

        return image, target


class YOLODataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for YOLO formatted datasets.

    The dataset configuration is provided via a Config object that contains
    the dataset configuration including paths, names, etc.

    When the images are located in a folder (for example, ./images/train), the
    corresponding labels are assumed to be in ./labels/train, following the same
    subdirectory naming.
    """

    def __init__(
        self,
        config: Config,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
    ):
        super().__init__()

        # Store the config
        self.config = config
        self.num_workers = max(1, os.cpu_count() // 2) if config.num_workers == -1 else config.num_workers

        # Get dataset information from the config
        self.dataset_root = Path(config.dataset.path)
        self.train_dir = self.dataset_root / config.dataset.train
        self.val_dir = self.dataset_root / config.dataset.val
        self.test_dir = None
        if config.dataset.test:
            self.test_dir = self.dataset_root / config.dataset.test

        # Compute the corresponding labels directories based on images directories,
        # following the convention: ./images/<split> <--> ./labels/<split>
        self.train_labels_dir = self._get_labels_dir(self.train_dir)
        self.val_labels_dir = self._get_labels_dir(self.val_dir)
        self.test_labels_dir = self._get_labels_dir(self.test_dir) if self.test_dir is not None else None

        self.train_transforms = train_transforms or default_train_transforms(max_size=config.train.data.image_size)
        self.val_transforms = val_transforms or default_val_transforms(max_size=config.train.data.image_size)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_labels_dir(self, images_dir: Path) -> Path:
        """
        Given an images directory, returns the corresponding labels directory.
        Follows the convention:
          If images_dir is .../images/<subset>, then labels_dir should be .../labels/<subset>
        """
        if images_dir is None:
            raise ValueError("images_dir must not be None")
        if images_dir.parent.name.lower() == "images":
            # For example, if images_dir is /path/to/dataset/images/train,
            # then labels_dir becomes /path/to/dataset/labels/train.
            return images_dir.parent.parent / "labels" / images_dir.name
        else:
            # Fallback: perform a string replacement of 'images' with 'labels'
            return Path(str(images_dir).replace("images", "labels", 1))

    def prepare_data(self) -> None:
        """
        Check that the train, val (and optionally test) directories exist.
        """
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        if not self.val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")
        if self.test_dir and not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

    def setup(self, stage: str | None = None) -> None:
        """
        Set up the train, validation, and test datasets.
        This hook is called on every process (for distributed setups).
        """
        if stage in (None, "fit"):
            self.train_dataset = YOLODetectionDataset(
                images_dir=self.train_dir,
                labels_dir=self.train_labels_dir,
                classes=self.config.dataset.names,
                transforms=self.train_transforms,
            )
            self.val_dataset = YOLODetectionDataset(
                images_dir=self.val_dir,
                labels_dir=self.val_labels_dir,
                classes=self.config.dataset.names,
                transforms=self.val_transforms,
            )

        if stage in (None, "test") and self.test_dir is not None:
            self.test_dataset = YOLODetectionDataset(
                images_dir=self.test_dir,
                labels_dir=self.test_labels_dir,
                classes=self.config.dataset.names,
                transforms=self.val_transforms,
            )

    def _collate_fn(self, batch: list[tuple[torch.Tensor, list]]) -> dict[str, torch.Tensor]:
        """
        Custom collate function for batching YOLO detection samples.
        """
        images = torch.stack([item[0] for item in batch])
        batch_size = len(batch)
        max_boxes = max(len(item[1]) for item in batch)

        boxes = torch.zeros((batch_size, max_boxes, 4))
        labels = torch.zeros((batch_size, max_boxes), dtype=torch.long)

        for i, (_, targets) in enumerate(batch):
            if targets:
                num_boxes = len(targets)
                boxes[i, :num_boxes] = torch.stack([torch.tensor(ann["bbox"], dtype=torch.float32) for ann in targets])
                labels[i, :num_boxes] = torch.tensor([ann["class_id"] for ann in targets], dtype=torch.long)
        return {
            "images": images,  # shape: (batch_size, C, H, W)
            "boxes": boxes,  # shape: (batch_size, max_boxes, 4)
            "labels": labels.unsqueeze(-1),  # shape: (batch_size, max_boxes, 1)
        }

    def train_dataloader(self) -> DataLoader:
        """
        Returns the train DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.data.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.data.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader | None:
        """
        Returns the test DataLoader if a test directory is provided.
        """
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.train.data.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            collate_fn=self._collate_fn,
        )

    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.config.dataset.names)


if __name__ == "__main__":
    # Example usage:
    # Create a YAML config (e.g., dataset.yaml) with:
    #
    # names:
    #   0: person
    #   1: car
    # path: /path/to/your/dataset
    # train: ./images/train/
    # val: ./images/val/
    #
    # This is provided through the ConfigManager.
    from angelcv.config import ConfigManager

    # Example usage with Config object
    config = ConfigManager.upsert_config(
        model_file="yolov10n.yaml",
        dataset_file=str(Path("~/Code/defendry-dataset/export-custom-v3/dataset.yaml").expanduser()),
    )

    yolo_dm = YOLODataModule(config=config)

    yolo_dm.prepare_data()
    yolo_dm.setup()

    train_loader = yolo_dm.train_dataloader()
    val_loader = yolo_dm.val_dataloader()
    test_loader = yolo_dm.test_dataloader()

    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))
    if test_loader:
        print("Test loader length:", len(test_loader))
    else:
        print("Test loader not available.")

    first_train_batch = next(iter(train_loader))
    print("Train batch images shape:", first_train_batch["images"].shape)
    print("Train batch boxes shape:", first_train_batch["boxes"].shape)
    print("Train batch labels shape:", first_train_batch["labels"].shape)
