from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from angelcv.utils.logging_manager import get_logger
from angelcv.utils.source_utils import ImageCoordinateMapper

logger = get_logger(__name__)


class Boxes:
    """
    A class to handle bounding box conversions between different coordinate formats.

    This class takes model detection outputs and provides properties to access
    the bounding boxes in various formats, both in absolute pixel coordinates
    and normalized (0-1) coordinates.

    Supported formats:
    - xyxy: [x1, y1, x2, y2] (top-left and bottom-right corners)
    - xywh: [x, y, width, height] (top-left corner, width, height)
    - cxcywh: [center_x, center_y, width, height] (center point, width, height)

    Each format is available in both pixel coordinates and normalized (0-1) coordinates.
    """

    def __init__(
        self,
        model_output: np.ndarray | torch.Tensor,
        original_width: int,
        original_height: int,
        img_coordinate_mapper: ImageCoordinateMapper,
        class_labels: dict[int, str] | None = None,
    ):
        """
        Initialize a Boxes object with model output and image dimensions.

        Args:
            model_output: Detection results tensor/array with shape (num_detections, 6+)
                          where each row contains [x1, y1, x2, y2, confidence, class_id, ...]
                          The first 4 values are bounding box coordinates in xyxy format.
                          The 5th value is the confidence/probability.
                          The 6th value is the class ID (integer).
                          Any additional values are ignored.
            original_width: Width of the original image in pixels
            original_height: Height of the original image in pixels
            img_coordinate_mapper: ImageCoordinateMapper object containing transformation parameters
            class_labels: Optional dictionary mapping class indexes to class names
        """
        model_output_np = model_output.cpu().numpy() if isinstance(model_output, torch.Tensor) else model_output
        self.original_width = original_width
        self.original_height = original_height

        # Convert from inference dimensions to original dimensions
        xyxy_pix_inference = model_output_np[:, :4]
        # NOTE: this conversion takes into account the padding and resizing applied to the image
        xyxy_pix_original = img_coordinate_mapper.transformed_to_original(xyxy_pix_inference)

        self._xyxy_pix = self._clean_xyxy_pix(xyxy_pix_original)
        self.confidences = model_output_np[:, 4]
        self.class_label_ids = model_output_np[:, 5].astype(int)
        self._class_labels = class_labels
        self.labels = self._create_labels_from_class_ids()

    def _create_labels_from_class_ids(self) -> list[str]:
        """Create labels list from class IDs using the class_labels mapping."""
        # If no class_label_ids exist, return empty list
        if self.class_label_ids.size == 0:
            return []

        # Create labels for each class ID, using provided mapping where available
        # and falling back to "class_{i}" format for missing classes
        labels = []
        for class_id in self.class_label_ids:
            if self._class_labels and class_id in self._class_labels:
                labels.append(self._class_labels[class_id])
            else:
                labels.append(f"class_{class_id}")

        return labels

    def _clean_xyxy_pix(self, xyxy_pix: np.ndarray) -> np.ndarray:
        """
        Clean the bounding box coordinates to ensure they are within the image dimensions.

        Args:
            xyxy_pix: Array of bounding box coordinates in xyxy format.

        Returns:
            np.ndarray: Cleaned bounding box coordinates in xyxy format.
        """
        xyxy_pix_orig = xyxy_pix.copy()
        xyxy_pix[:, 0] = np.clip(xyxy_pix[:, 0], 0, self.original_width)
        xyxy_pix[:, 1] = np.clip(xyxy_pix[:, 1], 0, self.original_height)
        xyxy_pix[:, 2] = np.clip(xyxy_pix[:, 2], 0, self.original_width)
        xyxy_pix[:, 3] = np.clip(xyxy_pix[:, 3], 0, self.original_height)

        if (xyxy_pix != xyxy_pix_orig).any():
            logger.debug(
                f"Some bounding boxes were outside image dimensions and have been clipped, original: {xyxy_pix_orig}"
            )

        return xyxy_pix

    @property
    def class_labels(self) -> list[str]:
        """Get the list of class labels."""
        return self._class_labels

    @class_labels.setter
    def class_labels(self, class_labels: list[str]):
        """Set the class labels and update the detection labels."""
        self._class_labels = class_labels
        self.labels = self._create_labels_from_class_ids()

    @property
    def xyxy(self) -> np.ndarray:
        """Bounding boxes in [x1, y1, x2, y2] format (absolute pixel coordinates)"""
        return self._xyxy_pix

    @property
    def xyxy_norm(self) -> np.ndarray:
        """Bounding boxes in normalized [x1, y1, x2, y2] format (0-1 range)"""
        xyxy_norm = self.xyxy.copy()
        xyxy_norm[:, 0] /= self.original_width  # x1
        xyxy_norm[:, 1] /= self.original_height  # y1
        xyxy_norm[:, 2] /= self.original_width  # x2
        xyxy_norm[:, 3] /= self.original_height  # y2
        return xyxy_norm

    @property
    def xywh(self) -> np.ndarray:
        """Bounding boxes in [x, y, width, height] format (absolute pixel coordinates)"""
        xywh = self.xyxy_pix.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]  # width = x2 - x1
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]  # height = y2 - y1
        return xywh

    @property
    def xywh_norm(self) -> np.ndarray:
        """Bounding boxes in normalized [x, y, width, height] format (0-1 range)"""
        xywh_norm = self.xywh.copy()
        xywh_norm[:, 0] /= self.original_width  # x
        xywh_norm[:, 1] /= self.original_height  # y
        xywh_norm[:, 2] /= self.original_width  # width
        xywh_norm[:, 3] /= self.original_height  # height
        return xywh_norm

    @property
    def cxcywh(self) -> np.ndarray:
        """Bounding boxes in [center_x, center_y, width, height] format (absolute pixel coordinates)"""
        cxcywh = self.xywh.copy()
        cxcywh[:, 0] = cxcywh[:, 0] + cxcywh[:, 2] / 2  # cx = x + w/2
        cxcywh[:, 1] = cxcywh[:, 1] + cxcywh[:, 3] / 2  # cy = y + h/2
        return cxcywh

    @property
    def cxcywh_norm(self) -> np.ndarray:
        """Bounding boxes in normalized [center_x, center_y, width, height] format (0-1 range)"""
        cxcywh_norm = self.cxcywh.copy()
        cxcywh_norm[:, 0] /= self.original_width  # cx
        cxcywh_norm[:, 1] /= self.original_height  # cy
        cxcywh_norm[:, 2] /= self.original_width  # width
        cxcywh_norm[:, 3] /= self.original_height  # height
        return cxcywh_norm


class InferenceResult:
    def __init__(
        self,
        model_output: torch.Tensor,
        original_image: np.ndarray,
        confidence_th: float = 0.0,
        img_coordinate_mapper: ImageCoordinateMapper = None,
        class_labels: dict[int, str] | None = None,
    ):
        """
        Initialize inference results with model output and image information.

        Args:
            model_output: Model detection output tensor with shape (1, num_detections, 6)
            original_image: Original input image as numpy array in RGB format
            img_coordinate_mapper: ImageCoordinateMapper object containing transformation parameters
            confidence_th: Confidence threshold for filtering detections, default 0.0 (no filtering)
            class_labels: Dictionary mapping class indexes to class names
        """
        self.model_output = model_output
        self.original_image = original_image
        self.confidence_th = confidence_th

        assert model_output.ndim == 2, "model_output must be a 2D tensor"
        assert model_output.shape[1] == 6, "model_output must have 6 columns"

        # Extract bounding boxes, confidence and class labels
        self.boxes = Boxes(
            model_output=model_output,
            original_width=original_image.shape[1],
            original_height=original_image.shape[0],
            img_coordinate_mapper=img_coordinate_mapper,
            class_labels=class_labels,
        )

    def __str__(self):
        return f"InferenceResults(model_output={self.model_output})"

    @property
    def class_labels(self) -> list[str]:
        """Get the list of class labels, in the member boxes."""
        return self.boxes.class_labels

    # Forward the class labels to the boxes
    @class_labels.setter
    def class_labels(self, class_labels: list[str]):
        """Set the class labels and update the detection labels."""
        self.boxes.class_labels = class_labels

    def annotate_image(self, font_scale=0.5, thickness=2, show_conf=True):
        """
        Create an annotated copy of the original image with detection boxes and labels.

        Args:
            font_scale (float): Scale of font for the labels. Default is 0.5.
            thickness (int): Thickness of bounding box lines. Default is 2.
            show_conf (bool): Whether to show confidence scores. Default is True.

        Returns:
            np.ndarray: A copy of the original image (RGB format) with drawn bounding boxes and labels.
        """
        # Make a copy of the original image to avoid modifying it
        annotated_img = self.original_image.copy()

        # Determine the number of colors needed
        if self.boxes.class_labels:
            # If class labels are provided, use the number of labels
            num_colors_to_generate = len(self.boxes.class_labels)
        elif self.boxes.class_label_ids.size > 0:
            # If no labels, but class IDs exist, generate colors up to the max class ID.
            # This ensures each unique class ID (up to max_id) can map to a unique color.
            num_colors_to_generate = int(np.max(self.boxes.class_label_ids)) + 1
        else:
            # Default to 1 color if no labels and no class_ids (e.g., no detections)
            num_colors_to_generate = 1

        # Ensure at least one color is generated to avoid issues with range or modulo operations.
        num_colors_to_generate = max(1, num_colors_to_generate)

        # Generate random colors
        colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # noqa: S311
            for _ in range(num_colors_to_generate)
        ]

        # Draw each detection
        for i, box in enumerate(self.boxes.xyxy):
            # Get integer coordinates for drawing
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Get class information
            class_id = self.boxes.class_label_ids[i]
            conf = self.boxes.confidences[i]

            # Skip detections below confidence threshold
            if conf < self.confidence_th:
                continue

            # Get the color for this class
            color = colors[class_id % len(colors)]

            # Draw rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

            # Create label text with class name and optional confidence
            if self.boxes.class_labels:
                label = f"{self.boxes.labels[i]}"
                if show_conf:
                    label += f" {conf:.2f}"
            else:
                label = f"Class {class_id}"
                if show_conf:
                    label += f" {conf:.2f}"

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(annotated_img, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(
                annotated_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )

        return annotated_img

    def show(self, window_name="Inference Result", block=True):
        """
        Display the annotated image in a window using matplotlib.

        Args:
            window_name (str): Title for the matplotlib figure. Default is "Inference Result".
            block (bool): If True, blocks execution until the window is closed. Default is True.
        """
        annotated_img_rgb = self.annotate_image()

        # The annotated image is already in RGB format, so no conversion needed for matplotlib
        # Create a figure with the specified title
        plt.figure(figsize=(10, 8))
        plt.title(window_name)

        # Display the image
        plt.imshow(annotated_img_rgb)
        plt.axis("off")  # Hide axes

        # Show the plot
        if block:
            plt.show()  # This will block until window is closed
        else:
            plt.show(block=False)
            plt.pause(0.001)  # Small pause to render the window

    def save(self, output_path: str | Path, show_conf: bool = True):
        """
        Save the annotated image to a file.

        Args:
            output_path (str or Path): Path where to save the annotated image.
            show_conf (bool): Whether to show confidence scores in the saved image. Default is True.

        Returns:
            Path: The path to the saved image.
        """
        output_path = Path(output_path)

        # Create the parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the annotated image (in RGB format)
        annotated_img_rgb = self.annotate_image(show_conf=show_conf)

        # Convert RGB to BGR for OpenCV's imwrite which expects BGR
        annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)

        # Save the image
        cv2.imwrite(str(output_path), annotated_img_bgr)

        return output_path


if __name__ == "__main__":
    from PIL import Image

    from angelcv.utils.source_utils import ImageCoordinateMapper

    # Load an image
    img = Image.open("angelcv/images/city.jpg").convert("RGB")
    img = np.array(img)

    # Create a sample model output with 3 detections
    detections = torch.tensor(
        [
            [120, 120, 280, 280, 0.95, 0],  # Detect the red rectangle
            [350, 150, 550, 350, 0.85, 1],  # Detect the green circle
            [50, 50, 150, 150, 0.65, 2],  # Another detection
        ]
    )

    # Create class labels
    class_labels = ["rectangle", "circle", "other"]

    # Create an ImageCoordinateMapper
    h, w = img.shape[:2]
    img_coordinate_mapper = ImageCoordinateMapper(
        original_width=w,
        original_height=h,
        transformed_width=640,
        transformed_height=480,
        scale_x=640 / w,
        scale_y=480 / h,
        padding_x=0,
        padding_y=0,
    )

    # Create the inference result
    inference_result = InferenceResult(
        detections,
        original_image=img,
        img_coordinate_mapper=img_coordinate_mapper,
        class_labels=class_labels,
    )

    # Example 1: Display the annotated image
    print("Displaying annotated image. Press any key to continue...")
    inference_result.show()

    """
    # Example 2: Save the annotated image
    output_path = inference_result.save("detection_demo.jpg")
    print(f"Saved annotated image to {output_path}")
    """
