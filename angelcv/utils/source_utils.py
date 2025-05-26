from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import albumentations as A
from einops import rearrange
import numpy as np
from PIL import Image
import requests
import torch

from angelcv.dataset.augmentation import default_val_transforms

# Define a type alias for source inputs
SourceType = str | Path | torch.Tensor | np.ndarray | Image.Image


@dataclass
class ImageCoordinateMapper:
    """Parameters and methods for mapping between original and transformed image coordinates."""

    original_width: int  # Width of original image
    original_height: int  # Height of original image
    transformed_width: int  # Width of transformed image
    transformed_height: int  # Height of transformed image
    scale_x: float  # Horizontal scaling factor
    scale_y: float  # Vertical scaling factor
    padding_x: float  # Horizontal padding added during transformation
    padding_y: float  # Vertical padding added during transformation

    def original_to_transformed(self, coords: np.ndarray) -> np.ndarray:
        """
        Maps coordinates from the original image space to the transformed image space.

        Args:
            coords: Coordinates in original image space, shape (N, 4) in format [x1, y1, x2, y2]
                   or shape (N, 2) in format [x, y]

        Returns:
            np.ndarray: Coordinates mapped to transformed image space
        """
        # Copy to avoid modifying the original
        transformed = coords.copy()

        # Check if we have boxes [x1, y1, x2, y2] or points [x, y]
        is_boxes = coords.shape[-1] == 4

        # Apply scaling
        if is_boxes:
            # For bounding boxes [x1, y1, x2, y2]
            transformed[:, 0] *= self.scale_x  # x1
            transformed[:, 2] *= self.scale_x  # x2
            transformed[:, 1] *= self.scale_y  # y1
            transformed[:, 3] *= self.scale_y  # y2
        else:
            # For points [x, y]
            transformed[:, 0] *= self.scale_x  # x
            transformed[:, 1] *= self.scale_y  # y

        # Add padding
        if is_boxes:
            transformed[:, 0] += self.padding_x  # x1
            transformed[:, 2] += self.padding_x  # x2
            transformed[:, 1] += self.padding_y  # y1
            transformed[:, 3] += self.padding_y  # y2
        else:
            transformed[:, 0] += self.padding_x  # x
            transformed[:, 1] += self.padding_y  # y

        return transformed

    def transformed_to_original(self, coords: np.ndarray) -> np.ndarray:
        """
        Maps coordinates from the transformed image space back to the original image space.

        Args:
            coords: Coordinates in transformed image space, shape (N, 4) in format [x1, y1, x2, y2]
                   or shape (N, 2) in format [x, y]

        Returns:
            np.ndarray: Coordinates mapped to original image space
        """
        # Copy to avoid modifying the original
        original = coords.copy()

        # Check if we have boxes [x1, y1, x2, y2] or points [x, y]
        is_boxes = coords.shape[-1] == 4

        # Remove padding
        if is_boxes:
            # For bounding boxes [x1, y1, x2, y2]
            original[:, 0] -= self.padding_x  # x1
            original[:, 2] -= self.padding_x  # x2
            original[:, 1] -= self.padding_y  # y1
            original[:, 3] -= self.padding_y  # y2
        else:
            # For points [x, y]
            original[:, 0] -= self.padding_x  # x
            original[:, 1] -= self.padding_y  # y

        # Apply inverse scaling
        if is_boxes:
            # For bounding boxes [x1, y1, x2, y2]
            if self.scale_x > 0:
                original[:, 0] /= self.scale_x  # x1
                original[:, 2] /= self.scale_x  # x2
            if self.scale_y > 0:
                original[:, 1] /= self.scale_y  # y1
                original[:, 3] /= self.scale_y  # y2
        else:
            # For points [x, y]
            if self.scale_x > 0:
                original[:, 0] /= self.scale_x  # x
            if self.scale_y > 0:
                original[:, 1] /= self.scale_y  # y

        return original


def preprocess_sources(
    source: SourceType | list[SourceType],
    image_size: int | None = None,
) -> tuple[list[torch.Tensor], list[np.ndarray], list[str], list[ImageCoordinateMapper]]:
    """
    Preprocesses various input sources for model inference.

    Args:
        source: Source for detection in various formats:
            - String: Path to image file or URL
            - Path: Path to image file
            - torch.Tensor: Image tensor in [C,H,W] or [B,C,H,W] format
            - np.ndarray: Image array in [H,W,C] format
            - PIL.Image.Image: PIL Image object
            - list: List containing any combination of the above
        image_size: Size to resize images to (longest edge)

    Returns:
        tuple containing:
        - List of preprocessed tensors ready for model input
        - List of original images
        - List of source identifiers (paths, URLs, or type descriptors)
        - List of ImageCoordinateMapper objects for mapping between original and transformed coordinates
    """
    # Ensure source is a list
    sources = [source] if not isinstance(source, list) else source

    processed_tensors = []
    original_images = []
    source_identifiers = []
    img_coordinate_mapper_list = []

    for src in sources:
        if isinstance(src, (str, Path)):
            tensor, orig_img, source_identifier, img_coordinate_mapper = _process_path_source(src, image_size)
        elif isinstance(src, Image.Image):
            tensor, orig_img, img_coordinate_mapper = _process_pil_image(src, image_size)
            source_identifier = "pil_image"
        elif isinstance(src, torch.Tensor):
            tensor, orig_img, img_coordinate_mapper = _process_tensor(src)
            source_identifier = "tensor"
        elif isinstance(src, np.ndarray):
            tensor, orig_img, img_coordinate_mapper = _process_numpy_array(src, image_size)
            source_identifier = "array"
        else:
            raise ValueError(f"Unsupported source type: {type(src)}")

        processed_tensors.append(tensor)
        original_images.append(orig_img)
        source_identifiers.append(source_identifier)
        img_coordinate_mapper_list.append(img_coordinate_mapper)

    return processed_tensors, original_images, source_identifiers, img_coordinate_mapper_list


def _process_path_source(
    src: str | Path, image_size: int | None = None
) -> tuple[torch.Tensor, np.ndarray, str, ImageCoordinateMapper]:
    """Process a source that is a file path or URL."""
    path = str(src)

    try:
        if path.lower().startswith(("http://", "https://")):
            # Handle URL
            response = requests.get(path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Handle local file
            img = Image.open(path).convert("RGB")

        numpy_img = np.array(img)
        tensor, img_coordinate_mapper = transform_image_for_inference(numpy_img, image_size=image_size)
        return tensor, numpy_img, path, img_coordinate_mapper

    except Exception as e:
        source_type = "URL" if path.lower().startswith(("http://", "https://")) else "file"
        raise ValueError(f"Failed to load image from {source_type}: {path}. Error: {str(e)}") from e


def _process_pil_image(
    src: Image.Image, image_size: int | None = None
) -> tuple[torch.Tensor, np.ndarray, ImageCoordinateMapper]:
    """Process a source that is a PIL Image."""
    if src.mode != "RGB":
        src = src.convert("RGB")
    numpy_img = np.array(src)
    tensor, img_coordinate_mapper = transform_image_for_inference(numpy_img, image_size=image_size)
    return tensor, numpy_img, img_coordinate_mapper


def _process_tensor(src: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, ImageCoordinateMapper]:
    """Process a source that is already a tensor."""
    # Add batch dimension if needed
    if src.dim() == 3:
        src = src.unsqueeze(0)

    # Convert to CPU for numpy conversion in case it's on GPU
    original_img = src.cpu()
    if original_img.shape[1] == 3:  # If tensor is in [B,C,H,W] format
        original_img = rearrange(original_img, "b c h w -> b h w c").numpy()

    # Extract the image (remove batch dimension if batch size is 1)
    numpy_img = original_img[0] if original_img.shape[0] == 1 else original_img

    # For tensor inputs, we assume it's already properly processed
    h, w = src.shape[2:4]
    img_coordinate_mapper = ImageCoordinateMapper(
        original_width=w,
        original_height=h,
        transformed_width=w,
        transformed_height=h,
        scale_x=1.0,
        scale_y=1.0,
        padding_x=0,
        padding_y=0,
    )

    return src, numpy_img, img_coordinate_mapper


def _process_numpy_array(
    src: np.ndarray, image_size: int | None = None
) -> tuple[torch.Tensor, np.ndarray, ImageCoordinateMapper]:
    """Process a source that is a numpy array."""
    tensor, img_coordinate_mapper = transform_image_for_inference(src, image_size=image_size)
    return tensor, src, img_coordinate_mapper


def transform_image_for_inference(
    img: np.ndarray, image_size: int | None = None
) -> tuple[torch.Tensor, ImageCoordinateMapper]:
    """
    Transforms an image for model inference by applying resizing, normalization, and conversion to tensor.

    Args:
        img: Input image as numpy array in [H,W,C] format with RGB channels
        image_size: Size of the longest side of the image to be resized to
    Returns:
        tuple containing:
        - torch.Tensor: Processed tensor ready for model input in [1,C,H,W] format
        - ImageCoordinateMapper: Transformation parameters for mapping between original and transformed coordinates
    """
    if image_size is None:
        # If no image_size is specified, just convert to tensor without resizing
        # This would need proper implementation based on your requirements
        # For now, using default transforms with original image size
        h, w = img.shape[:2]
        image_size = max(h, w)

    # Create transforms based on default_val_transforms but without bbox_params
    transforms_list = default_val_transforms(max_size=image_size).transforms

    # Create a separate transforms for tracking parameters
    # We need to include bbox_params to ensure the transformation parameters are tracked
    inference_transforms = A.Compose(
        transforms=transforms_list, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])
    )

    # Create a dummy bounding box for the entire image to track transformations
    h, w = img.shape[:2]
    dummy_bbox = [0, 0, w, h]  # [x_min, y_min, x_max, y_max] in Pascal VOC format

    # Apply transforms to image with dummy bounding box
    transformed = inference_transforms(image=img, bboxes=[dummy_bbox], labels=[0])
    img_tensor = transformed["image"]
    transformed_bbox = transformed["bboxes"][0]

    # Calculate transformation parameters
    original_width, original_height = w, h
    new_h, new_w = img_tensor.shape[1:3] if img_tensor.dim() == 3 else img_tensor.shape[2:4]
    transformed_width, transformed_height = new_w, new_h

    # Extract scale factors and padding from transformed bbox
    # The transformed bbox tells us how the entire original image was mapped
    x_min, y_min, x_max, y_max = transformed_bbox

    img_coordinate_mapper = ImageCoordinateMapper(
        original_width=original_width,
        original_height=original_height,
        transformed_width=transformed_width,
        transformed_height=transformed_height,
        scale_x=(x_max - x_min) / original_width,
        scale_y=(y_max - y_min) / original_height,
        padding_x=x_min,
        padding_y=y_min,
    )

    # Ensure the tensor has batch dimension
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, img_coordinate_mapper


if __name__ == "__main__":
    from time import time

    image_size = 640

    # 1. Test with a list of mixed types
    print("\n1. Testing with mixed list of sources...")
    mixed_sources = [
        Path("angelcv/images/city.jpg"),  # path
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",  # url
    ]
    start = time()
    tensors1, images1, identifiers1, img_coordinate_mapper1 = preprocess_sources(mixed_sources, image_size=image_size)
    print(f"✓ Mixed list: {len(tensors1)} tensors processed in {time() - start:.3f}s")
    for i, (tensor, identifier) in enumerate(zip(tensors1, identifiers1)):
        print(f"  - Source {i + 1}: {identifier}, Shape: {tensor.shape}")
