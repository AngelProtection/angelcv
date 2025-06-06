import os
from pathlib import Path
import shutil
from typing import Callable, Optional
import urllib.request
import zipfile

import cv2
import lightning as L
import numpy as np
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from angelcv.config.config_registry import Config
from angelcv.dataset.augmentation import default_train_transforms, default_val_transforms


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class CocoDetection(Dataset):
    def __init__(self, root: str | Path, ann_file: str | Path, transforms: Callable | None = None):
        self.root = Path(root)
        self.coco = COCO(str(ann_file))
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

        # Create category ID mapping
        self.cat_mapping = {cat: idx for idx, cat in enumerate(self.coco.getCatIds())}
        # image_id to image info dictionary
        self.image_id_to_info = self.coco.imgs

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(str(self.root / path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)

        # keys: 'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'
        image_info = self.image_id_to_info[image_id]
        img_h, img_w = image.shape[:2]

        # Convert all annotations to numpy array first for vectorized operations
        coco_bboxes = np.empty((0, 4), dtype=np.float32)  # Shape: (N, 4)
        labels = np.empty((0,), dtype=int)
        for ann in target:
            coco_bboxes = np.vstack([coco_bboxes, ann["bbox"]])
            labels = np.append(labels, self.cat_mapping[ann["category_id"]])

        # Vectorized conversion from COCO xywh_pix to xyxy_norm
        xyxy_norm_bboxes = np.empty_like(coco_bboxes)
        xyxy_norm_bboxes[:, 0] = coco_bboxes[:, 0] / img_w  # x1
        xyxy_norm_bboxes[:, 1] = coco_bboxes[:, 1] / img_h  # y1
        # x2
        xyxy_norm_bboxes[:, 2] = (coco_bboxes[:, 0] + np.maximum(coco_bboxes[:, 2], np.finfo(np.float32).eps)) / img_w
        # y2
        xyxy_norm_bboxes[:, 3] = (coco_bboxes[:, 1] + np.maximum(coco_bboxes[:, 3], np.finfo(np.float32).eps)) / img_h

        eps = np.finfo(np.float32).eps
        if np.any((xyxy_norm_bboxes > 1.0) | (xyxy_norm_bboxes <= 0.0)):
            if np.any((xyxy_norm_bboxes - eps > 1.0) | (xyxy_norm_bboxes + eps <= 0.0)):
                out_of_bounds = xyxy_norm_bboxes[(xyxy_norm_bboxes > 1.0) | (xyxy_norm_bboxes <= 0.0)]
                print(f"WARNING: bbox coordinate out of bounds: {out_of_bounds}")

            # Clip values to ensure they're between 0 and 1
            # NOTE: eps is required instead of 0, because albumentations requires the yolo bboxes to be (0, 1]
            # and few bboxes in COCO are 0, so we need to add eps
            xyxy_norm_bboxes = np.clip(xyxy_norm_bboxes, a_min=eps, a_max=1)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=xyxy_norm_bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Create target dict to maintain compatibility
        target = []
        for bbox, label in zip(bboxes, labels):
            target.append(
                {
                    "image_id": image_id,  # unused
                    "bbox": bbox,  # (xyxy) normalized
                    "class_id": label,
                    # NOTE: those are image information elements, which are added to each detection
                    "original_wh": (image_info["width"], image_info["height"]),  # unused
                    "filename": image_info["file_name"],  # unused
                }
            )

        return image, target

    def __len__(self):
        return len(self.image_ids)


# TODO [LOW]: either make CocoDataModule inherit from YOLODataModule or
# make available for download the COCO dataset in YOLO format
class CocoDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: Config,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        task: str = "detection",
    ):
        """
        PyTorch Lightning DataModule for COCO dataset.

        Args:
            data_dir (Path): Root directory for COCO dataset
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of subprocesses for data loading
            pin_memory (bool): Use pinned memory for faster data transfer to GPU
            train_transforms (Optional[Callable]): Optional transform for training data
            val_transforms (Optional[Callable]): Optional transform for validation data
            task (str): COCO dataset task - 'detection' or 'caption'
        """
        super().__init__()

        self.config = config
        self.data_dir = Path(config.dataset.path).expanduser()
        self.num_workers = max(1, os.cpu_count() // 2) if config.num_workers == -1 else config.num_workers

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Default transforms if not provided
        self.train_transforms = train_transforms or default_train_transforms(max_size=config.train.data.image_size)
        self.val_transforms = val_transforms or default_val_transforms(max_size=config.train.data.image_size)

        # COCO dataset parameters
        self.task = task
        self.annotation_files = {
            "detection": {
                "train": "trainval_annotations/instances_train2017.json",
                "val": "trainval_annotations/instances_val2017.json",
                "test": "test_annotations/image_info_test2017.json",
            },
        }

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """
        Download COCO dataset if not already present.
        This method is from the main process, once per machine, do NOT assign state ere (i.e. self.x = x)

        Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single process,
        so you can safely add your downloading logic within.
        """
        # COCO dataset URLs
        urls = {
            "train2017": "http://images.cocodataset.org/zips/train2017.zip",
            "val2017": "http://images.cocodataset.org/zips/val2017.zip",
            "test2017": "http://images.cocodataset.org/zips/test2017.zip",
            "trainval_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "test_annotations": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
        }

        # Create temporary directory for downloads
        download_dir = self.data_dir / "tmp"

        try:
            # Check and download each component if missing
            for folder_name, url in urls.items():
                component_path = self.data_dir / folder_name

                if not component_path.exists():
                    download_dir.mkdir(parents=True, exist_ok=True)
                    zip_file = download_dir / f"{folder_name}.zip"
                    print(f"Downloading {folder_name}...")
                    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=folder_name) as t:
                        urllib.request.urlretrieve(url, zip_file, reporthook=t.update_to)  # noqa: S310

                    # Extract files
                    print(f"Extracting {folder_name}...")
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(component_path)

                    # Find and get rid of nested folders
                    # Move contents to final location
                    # Find the extracted folder (it should be the only one in extract_dir)
                    extracted_items = list(component_path.iterdir())
                    if len(extracted_items) == 1 and extracted_items[0].is_dir():
                        # If we have a single directory, move its contents
                        single_extracted_folder = extracted_items[0]

                        # Move all items from source_dir to component_path
                        for item in single_extracted_folder.iterdir():
                            shutil.move(str(item), str(component_path / item.name))

                        single_extracted_folder.rmdir()
        finally:
            # Clean up temporary files
            if download_dir.exists():
                shutil.rmtree(download_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup train, validation, and test datasets.

        Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when you
        need to build models dynamically or adjust something about them. This hook is called on every process when
        using DDP.

        Args:
            stage (Optional[str]): "fit", "test", "validate" or "predict" stage
        """
        # Splitting logic based on stage
        if stage in (None, "fit"):
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("val")

        if stage in (None, "test"):
            self.test_dataset = self._create_dataset("test")

    def _create_dataset(self, split: str) -> Dataset:
        """
        Create dataset for given split.

        Args:
            split (str): train, val, or test

        Returns:
            Dataset: COCO dataset for specified split
        """
        if self.task.lower() == "detection":
            transforms = self.train_transforms if split == "train" else self.val_transforms

            # Use the correct annotation file for the split
            ann_file = self.annotation_files["detection"][split]

            return CocoDetection(
                root=self.data_dir / f"{split}2017",
                ann_file=self.data_dir / ann_file,
                transforms=transforms,
            )
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def _collate_fn(self, batch: list[tuple[torch.Tensor, dict]]) -> dict[str, torch.Tensor]:
        """
        Custom collate function for batching COCO detection samples.

        Args:
            batch: List of tuples (one for each image in the batch), where each tuple contains:
                - torch.Tensor: Image tensor of shape (C, H, W)
                - list of dicts: each dict is a detection containing:
                    - 'image_id': ID of the image
                    - 'bbox': List of 4 floats [x1, y1, x2, y2] (normalized)
                    - 'class_id': Integer class label
                    - Other COCO annotation fields (not used)

        Returns:
            dict with:
                'images': tensor of shape (batch_size, channels, height, width)
                'boxes': tensor of shape (batch_size, max_boxes, 4) with padding (xyxy_norm)
                'labels': tensor of shape (batch_size, max_boxes) with padding
        """
        images = torch.stack([item[0] for item in batch])

        # Find max number of boxes in this batch
        max_boxes = max(len(item[1]) for item in batch)

        # Initialize tensors with padding
        batch_size = len(batch)
        boxes = torch.zeros((batch_size, max_boxes, 4))
        labels = torch.zeros((batch_size, max_boxes), dtype=torch.long)

        # Fill in the actual values
        for batch_idx, (_, targets) in enumerate(batch):
            if targets:  # if there are any targets
                num_boxes = len(targets)
                boxes[batch_idx, :num_boxes] = torch.stack([torch.tensor(ann["bbox"]) for ann in targets])
                labels[batch_idx, :num_boxes] = torch.tensor([ann["class_id"] for ann in targets])

        return {
            "images": images,  # shape: (batch_size, C, H, W)
            "boxes": boxes,  # shape: (batch_size, max_boxes, 4)
            "labels": labels.unsqueeze(-1),  # shape: (batch_size, max_boxes, 1)
        }

    def train_dataloader(self) -> DataLoader:
        """
        Train DataLoader.

        Returns:
            DataLoader: Training data loader
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
        Validation DataLoader.

        Returns:
            DataLoader: Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.data.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Test DataLoader.

        Returns:
            DataLoader: Test data loader
        """
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
        Get number of classes in the dataset.

        Returns:
            int: Number of classes
        """
        return len(self.config.dataset.names)


# Example usage
if __name__ == "__main__":
    from angelcv.config import ConfigManager

    # Instantiate the data module
    config = ConfigManager.upsert_config(dataset_file="coco.yaml")
    coco_dm = CocoDataModule(config)

    # Prepare and setup the data
    coco_dm.prepare_data()
    coco_dm.setup()

    # Access dataloaders
    train_loader = coco_dm.train_dataloader()
    val_loader = coco_dm.val_dataloader()
    test_loader = coco_dm.test_dataloader()

    # Lengths
    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))
    print("Test loader length:", len(test_loader))

    # Samples shape
    first_train_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))
    first_test_batch = next(iter(test_loader))

    print(
        "Train first sample shape:",
        first_train_batch["images"].shape,
        first_train_batch["boxes"].shape,
        first_train_batch["labels"].shape,
    )
    print(
        "Validation first sample shape:",
        first_val_batch["images"].shape,
        first_val_batch["boxes"].shape,
        first_val_batch["labels"].shape,
    )
    print(
        "Test first sample shape:",
        first_test_batch["images"].shape,
        first_test_batch["boxes"].shape,
        first_test_batch["labels"].shape,
    )
