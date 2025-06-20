from typing import Any

import albumentations as A
from albumentations.augmentations.mixing.transforms import Mosaic
from torch.utils.data import Dataset


class MosaicFromDataset(Mosaic):
    """
    A variation of the Mosaic augmentation that fetches additional images directly
    from a PyTorch Dataset at runtime, rather than requiring them to be passed
    via metadata.

    This transform is designed to be integrated into a PyTorch training pipeline,
    simplifying the data loading loop.

    Args:
        dataset (Dataset): An instance of a PyTorch-compatible dataset. The `__getitem__`
            method of this dataset must return a dictionary containing the keys expected
            by your Albumentations pipeline (e.g., 'image', 'bboxes', 'labels').
        **kwargs: All other keyword arguments accepted by the standard `Mosaic`
            transform (e.g., `grid_yx`, `target_size`, `p`).
    """

    def __init__(self, dataset: Dataset, **kwargs):
        # The metadata_key is irrelevant in this implementation, so we remove it
        # from kwargs if present to avoid confusion.
        kwargs.pop("metadata_key", None)
        super().__init__(**kwargs)
        self.dataset = dataset
        if len(self.dataset) == 0:
            raise ValueError("The provided dataset cannot be empty.")

    def _select_additional_items(
        self,
        data: dict[str, Any],
        num_additional_needed: int,
    ) -> list[dict[str, Any]]:
        """
        Overrides the parent method to source additional items from the dataset.
        """
        dataset_len = len(self.dataset)
        # It's important to use the transform's internal random generator for reproducibility.
        indices = [self.py_random.randint(0, dataset_len - 1) for _ in range(num_additional_needed)]

        additional_items = []
        for idx in indices:
            # This is the "contract": the dataset must return a dictionary
            # in a format that the rest of the Albumentations pipeline understands.
            # This fetched item is treated exactly like the items that would have
            # been in `mosaic_metadata`.
            item = self.dataset[idx]
            additional_items.append(item)

        return additional_items

    @property
    def targets(self) -> dict[str, Any]:
        """
        Overrides the parent `targets` property to remove the dependency on
        `mosaic_metadata`, as this transform sources data directly from the dataset.
        """
        # We start with the parent's targets...
        parent_targets = super().targets
        # ...and remove the metadata key, as it's not provided in the input.
        # We use the default key name directly to avoid an AttributeError during initialization,
        # as `self.metadata_key` may not be set yet when this is first called.
        parent_targets.pop("mosaic_metadata", None)
        return parent_targets

    @property
    def targets_as_params(self) -> list[str]:
        return []


if __name__ == "__main__":
    import albumentations as A
    import numpy as np
    from torch.utils.data import Dataset

    class FakeDataset(Dataset):
        """A dummy dataset for testing purposes."""

        def __init__(self, num_samples: int = 10, img_size: tuple = (416, 416)):
            self.num_samples = num_samples
            self.img_size = img_size

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> dict:
            img = np.random.randint(0, 255, size=(*self.img_size, 3), dtype=np.uint8)
            # One bounding box in the middle of the image
            bboxes = [
                [
                    self.img_size[1] * 0.25,
                    self.img_size[0] * 0.25,
                    self.img_size[1] * 0.75,
                    self.img_size[0] * 0.75,
                ]
            ]
            labels = [0]  # Single class
            return {"image": img, "bboxes": bboxes, "labels": labels}

    print("Running MosaicFromDataset test...")

    # 1. Create fake data source
    fake_dataset = FakeDataset(num_samples=10, img_size=(416, 416))

    # 2. Get a sample from the dataset to be transformed
    sample_to_transform = fake_dataset[0]

    # 3. Instantiate Mosaic transform and wrap it in a Compose pipeline.
    # The Mosaic transform depends on processors provided by Compose, so we must use it.
    transform_pipeline = A.Compose(
        [
            MosaicFromDataset(
                dataset=fake_dataset,
                p=1.0,  # Always apply for testing
            )
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    # 4. Apply the transform
    transformed = transform_pipeline(
        image=sample_to_transform["image"], bboxes=sample_to_transform["bboxes"], labels=sample_to_transform["labels"]
    )

    # 5. Print results to verify
    print("\n--- Mosaic Augmentation Test ---")
    print(f"Original image shape: {sample_to_transform['image'].shape}")
    print(f"Original bboxes: {sample_to_transform['bboxes']}")
    print("-" * 20)
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Transformed bboxes count: {len(transformed['bboxes'])}")
    if transformed["bboxes"]:
        print(f"Example transformed bbox: {transformed['bboxes'][0]}")
    else:
        print("No transformed bboxes.")
    print("\nTest finished. If you see this, the transform ran without crashing.")
