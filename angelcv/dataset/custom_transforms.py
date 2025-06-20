import random

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset


class DatasetMosaic(A.DualTransform):
    """
    Simple Mosaic augmentation that samples additional images from a PyTorch Dataset.
    Compatible with albumentations A.Compose.
    """

    def __init__(self, dataset: Dataset, num_additional: int = 3, p: float = 0.5):
        super().__init__(p=p)
        self.dataset = dataset
        self.num_additional = num_additional

    def apply(self, img, additional_images=None, **params):
        """Apply mosaic augmentation by combining the main image with additional images."""
        if additional_images is None or len(additional_images) == 0:
            return img  # Return original image if no additional images

        # Get image dimensions
        h, w = img.shape[:2]

        # Create 2x2 mosaic grid
        mosaic_h, mosaic_w = h * 2, w * 2
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=img.dtype)

        # Resize main image and place in top-left
        resized_main = cv2.resize(img, (w, h))
        mosaic[0:h, 0:w] = resized_main

        # Place additional images in other quadrants
        positions = [(0, w), (h, 0), (h, w)]  # top-right, bottom-left, bottom-right

        for i, pos in enumerate(positions):
            if i < len(additional_images):
                additional_img = additional_images[i]
                if additional_img is not None:
                    # Resize to fit quadrant
                    resized_additional = cv2.resize(additional_img, (w, h))
                    y, x = pos
                    mosaic[y : y + h, x : x + w] = resized_additional

        # Resize back to original dimensions
        final_mosaic = cv2.resize(mosaic, (w, h))
        return final_mosaic

    def apply_to_bboxes(self, bboxes, **params):
        """Transform bounding boxes for mosaic (simplified - just scale down to top-left quadrant)."""
        if len(bboxes) == 0:
            return bboxes

        # Scale bboxes to fit in top-left quadrant (0.5x scale)
        scaled_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            # Scale and keep in top-left quadrant
            scaled_bbox = [x_min * 0.5, y_min * 0.5, x_max * 0.5, y_max * 0.5]
            # Add any additional fields
            if len(bbox) > 4:
                scaled_bbox.extend(bbox[4:])
            scaled_bboxes.append(scaled_bbox)
        return scaled_bboxes

    def apply_to_keypoints(self, keypoints, **params):
        """Transform keypoints for mosaic (simplified - just scale down to top-left quadrant)."""
        if len(keypoints) == 0:
            return keypoints

        scaled_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            # Scale to top-left quadrant
            scaled_kp = [x * 0.5, y * 0.5]
            # Add any additional fields
            if len(kp) > 2:
                scaled_kp.extend(kp[2:])
            scaled_keypoints.append(scaled_kp)
        return scaled_keypoints

    def get_params_dependent_on_data(self, params, data):
        """Sample additional images from the dataset."""
        try:
            # Sample random indices from dataset
            dataset_size = len(self.dataset)
            current_idx = data.get("index", None)

            # Get available indices (excluding current if known)
            available_indices = list(range(dataset_size))
            if current_idx is not None and current_idx in available_indices:
                available_indices.remove(current_idx)

            # Sample additional images
            num_to_sample = min(self.num_additional, len(available_indices))
            if num_to_sample > 0:
                sampled_indices = random.sample(available_indices, num_to_sample)

                additional_images = []
                for idx in sampled_indices:
                    try:
                        # Get sample from dataset
                        sample = self.dataset[idx]

                        # Extract image (handle different return formats)
                        if isinstance(sample, tuple) and len(sample) >= 1:
                            image = sample[0]  # (image, target) format
                        elif isinstance(sample, dict) and "image" in sample:
                            image = sample["image"]  # dict format
                        else:
                            image = sample  # direct image

                        # Convert tensor to numpy if needed
                        if hasattr(image, "numpy"):
                            image = image.numpy()
                            # Handle tensor format (C, H, W) -> (H, W, C)
                            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                                image = image.transpose(1, 2, 0)
                            # Denormalize if needed
                            if image.dtype == np.float32 and image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)

                        # Ensure uint8 format
                        if image.dtype != np.uint8:
                            image = image.astype(np.uint8)

                        additional_images.append(image)

                    except Exception as e:
                        print(f"Warning: Failed to load image at index {idx}: {e}")
                        continue

                return {"additional_images": additional_images}

        except Exception as e:
            print(f"Warning: Failed to sample additional images: {e}")

        return {"additional_images": []}
