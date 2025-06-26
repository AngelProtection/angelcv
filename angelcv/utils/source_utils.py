from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange
import numpy as np
from PIL import Image
import requests
import torch

from angelcv.dataset.augmentation_pipelines import AUGMENTATION_BG_COLOR

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


def load_images(
    source: SourceType | list[SourceType],
) -> tuple[list[np.ndarray], list[str]]:
    """
    Loads various input sources and converts them to numpy arrays.

    Args:
        source: Source for detection in various formats:
            - String: Path to image file or URL
            - Path: Path to image file
            - torch.Tensor: Image tensor in [C,H,W] or [B,C,H,W] format
            - np.ndarray: Image array in [H,W,C] format
            - PIL.Image.Image: PIL Image object
            - list: List containing any combination of the above

    Returns:
        tuple containing:
        - List of numpy arrays representing the original images in [H,W,C] format, 0-255, RGB
        - List of source identifiers (paths, URLs, or type descriptors)
    """
    # Ensure source is a list
    sources = [source] if not isinstance(source, list) else source

    original_images = []
    source_identifiers = []

    for src in sources:
        if isinstance(src, (str, Path)):
            orig_img, source_identifier = _load_path_source(src)
        elif isinstance(src, Image.Image):
            orig_img = _load_pil_image(src)
            source_identifier = "pil_image"
        elif isinstance(src, torch.Tensor):
            orig_img = _load_tensor(src)
            source_identifier = "tensor"
        elif isinstance(src, np.ndarray):
            orig_img = _load_numpy_array(src)
            source_identifier = "array"
        else:
            raise ValueError(f"Unsupported source type: {type(src)}")

        original_images.append(orig_img)
        source_identifiers.append(source_identifier)

    return original_images, source_identifiers


def preprocess_for_inference(
    images: list[np.ndarray],
    image_size: int | None = None,
) -> tuple[torch.Tensor, list[ImageCoordinateMapper]]:
    """
    Preprocesses numpy arrays for model inference by applying resizing, normalization, and conversion to tensors.

    Args:
        images: List of numpy arrays in [H,W,C] format with RGB channels
        image_size: Size of the longest side of the image to be resized to

    Returns:
        tuple containing:
        - torch.Tensor: Batch of preprocessed tensors ready for model input in [B,C,H,W] format, 0-1, RGB
        - List of ImageCoordinateMapper objects for mapping between original and transformed coordinates
    """
    processed_tensors = []
    img_coordinate_mapper_list = []

    for img in images:
        tensor, img_coordinate_mapper = transform_image_for_inference(img, image_size=image_size)
        # Remove batch dimension from tensor (should be [1,C,H,W])
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        processed_tensors.append(tensor)
        img_coordinate_mapper_list.append(img_coordinate_mapper)

    # Stack tensors into a batch [B,C,H,W]
    batch_tensor = torch.stack(processed_tensors, dim=0)

    return batch_tensor, img_coordinate_mapper_list


def _load_path_source(src: str | Path) -> tuple[np.ndarray, str]:
    """Load a source that is a file path or URL."""
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
        return numpy_img, path

    except Exception as e:
        source_type = "URL" if path.lower().startswith(("http://", "https://")) else "file"
        raise ValueError(f"Failed to load image from {source_type}: {path}. Error: {str(e)}") from e


def _load_pil_image(src: Image.Image) -> np.ndarray:
    """Load a source that is a PIL Image."""
    if src.mode != "RGB":
        src = src.convert("RGB")
    numpy_img = np.array(src)
    return numpy_img


def _load_tensor(src: torch.Tensor) -> np.ndarray:
    """Load a source that is already a tensor."""
    # Convert to CPU for numpy conversion in case it's on GPU
    original_img = src.cpu()

    # Add batch dimension if needed
    if original_img.dim() == 3:
        original_img = original_img.unsqueeze(0)

    if original_img.shape[1] == 3:  # If tensor is in [B,C,H,W] format
        original_img = rearrange(original_img, "b c h w -> b h w c").numpy()

    # Extract the image (remove batch dimension if batch size is 1)
    numpy_img = original_img[0] if original_img.shape[0] == 1 else original_img

    # Ensure proper range and data type
    if numpy_img.dtype != np.uint8:
        # If normalized [0,1], convert to [0,255]
        if numpy_img.max() <= 1.0:
            numpy_img = (numpy_img * 255).astype(np.uint8)
        else:
            numpy_img = numpy_img.astype(np.uint8)

    return numpy_img


def _load_numpy_array(src: np.ndarray) -> np.ndarray:
    """Load a source that is a numpy array."""
    # Ensure proper shape [H,W,C]
    if src.ndim == 3 and src.shape[0] == 3:  # If in [C,H,W] format
        src = np.transpose(src, (1, 2, 0))

    # Ensure proper range and data type
    if src.dtype != np.uint8:
        # If normalized [0,1], convert to [0,255]
        if src.max() <= 1.0:
            src = (src * 255).astype(np.uint8)
        else:
            src = src.astype(np.uint8)

    return src


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

    # Create transforms manually for inference (same as builds_val_transfors, without config)
    transforms_list = A.Compose(
        transforms=[
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, fill=AUGMENTATION_BG_COLOR),
            A.Normalize(mean=0, std=1, max_pixel_value=255),  # This divides by 255
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
    ).transforms

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
        np.array(Image.open("angelcv/images/city.jpg")),  # numpy array
        torch.randn(3, 640, 640),  # tensor
        Image.open("angelcv/images/city.jpg"),  # PIL image
    ]
    start = time()

    # Load images first
    images, identifiers = load_images(mixed_sources)
    load_time = time() - start
    print(f"✓ Loading: {len(images)} images loaded in {load_time:.3f}s")
    for i, (img, identifier) in enumerate(zip(images, identifiers)):
        print(f"  - Source {i + 1}: {identifier}, Shape: {img.shape}")

    # Then preprocess for inference
    start = time()
    tensors, mappers = preprocess_for_inference(images, image_size=image_size)
    preprocess_time = time() - start
    print(f"✓ Preprocessing: {len(tensors)} tensors processed in {preprocess_time:.3f}s")

    for i, (tensor, identifier) in enumerate(zip(tensors, identifiers)):
        print(f"  - Source {i + 1}: {identifier}, Shape: {tensor.shape}")

    print(
        f"✓ Total time: {load_time + preprocess_time:.3f}s (load: {load_time:.3f}s, preprocess: {preprocess_time:.3f}s)"
    )
