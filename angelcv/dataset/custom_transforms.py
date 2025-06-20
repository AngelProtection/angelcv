from typing import Any

import albumentations as A
from torch.utils.data import Dataset


class MosaicFromDataset(A.Mosaic):
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
            # Call `getitem` with `augment=False` to get a dictionary containing the
            # raw, untransformed data (e.g., numpy image). This ensures the
            # data format is correct for the Mosaic transform to process.
            item = self.dataset.getitem(idx, augment=False)
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
