# AngelCV Interfaces

AngelCV provides Python interfaces to simplify interaction with models and results. The primary interfaces are `ObjectDetectionModel` for interacting with the detection models and `InferenceResult` (along with `Boxes`) for handling model outputs.

## `ObjectDetectionModel` Interface

Path: `angelcv.interface.object_detection.ObjectDetectionModel`

This is the main high-level class for object detection tasks. It acts as a wrapper around the underlying `YoloDetectionModel` (or other future model types) and PyTorch Lightning, providing a simplified API for training, inference, validation, testing, and model export.

Refer to the [Object Detection](./object_detection.md#high-level-interface-objectdetectionmodel) section for a detailed guide on its methods like `__init__`, `predict`, `train`, `validation`, `test`, and `export`.

Key responsibilities:

*   Loading models from configuration files (`.yaml`) or checkpoints (`.pt`, `.ckpt`).
*   Managing dataset configurations and data modules.
*   Orchestrating training, validation, and testing loops via PyTorch Lightning.
*   Preprocessing inputs for inference.
*   Postprocessing raw model outputs into `InferenceResult` objects.
*   Exporting models to deployment formats (e.g., ONNX).

## `InferenceResult` Interface

Path: `angelcv.interface.inference_result.InferenceResult`

When you run `model.predict(...)` or `model(...)`, you get back a list of `InferenceResult` objects, one for each image processed.

An `InferenceResult` object encapsulates everything related to the detection output for a single image.

```python
# Example usage after model.predict()
results: list[InferenceResult] = model.predict(source)
single_result: InferenceResult = results[0]
```

### Key Attributes and Methods

*   **`original_image: np.ndarray`**: The original input image (as a NumPy array in RGB format) that was processed.
*   **`boxes: Boxes`**: An instance of the `Boxes` class (see below), which holds all bounding box information, confidences, and class labels.
*   **`model_output: torch.Tensor`**: The raw, filtered output tensor from the model for this specific image, after initial confidence thresholding (shape: `[num_detections, 6]`, where columns are `x1, y1, x2, y2, confidence, class_id`). Coordinates are usually in the model's input resolution space before mapping back to original image coordinates (this mapping is handled by the `Boxes` class).
*   **`confidence_th: float`**: The confidence threshold that was applied to generate these results (note: `Boxes` also has its own thresholding for some operations if needed, but this reflects the threshold used during the `predict` call).
*   **`class_labels: list[str]`**: A list of class names. This can be set or accessed directly. If provided during `ObjectDetectionModel` initialization (via dataset config), results will have these populated.

*   **`annotate_image(font_scale=0.5, thickness=2, show_conf=True) -> np.ndarray`**:
    *   Draws bounding boxes, class labels, and (optionally) confidence scores on a *copy* of the original image.
    *   Returns the annotated image as a NumPy array (RGB).
*   **`show(window_name="Inference Result", block=True)`**:
    *   Displays the annotated image using OpenCV's `imshow`. Press any key to close the window.
    *   `block=True` means it will wait for a key press.
*   **`save(output_path: str | Path, show_conf: bool = True)`**:
    *   Saves the annotated image to the specified `output_path`.

## `Boxes` Interface

Path: `angelcv.interface.inference_result.Boxes`

This class is a component of `InferenceResult` (accessible via `result.boxes`) and is responsible for managing and converting bounding box coordinates. It takes the model's raw bounding box outputs (which are typically in the coordinate system of the resized/padded image fed to the model) and handles the mapping back to the original image's coordinate system.

### Initialization (Internal)

The `Boxes` object is created internally by `InferenceResult`.

```python
boxes_instance = Boxes(
    model_output: np.ndarray | torch.Tensor, # Filtered detections [N, 6]
    original_width: int,
    original_height: int,
    img_coordinate_mapper: ImageCoordinateMapper,
    class_labels: list[str] | None = None,
)
```

*   `img_coordinate_mapper`: An important utility (`angelcv.utils.source_utils.ImageCoordinateMapper`) that stores the transformation (padding, resizing) applied to the original image and knows how to map coordinates back and forth.

### Key Attributes (Properties)

All bounding box properties return NumPy arrays of shape `(num_detections, 4)` or `(num_detections,)` for confidences/labels.

*   **Coordinate Systems**: `Boxes` provides detected bounding boxes in multiple formats:
    *   **`xyxy`**: Absolute pixel coordinates `[x_min, y_min, x_max, y_max]` in the original image.
    *   **`xyxy_norm`**: Normalized coordinates `[x_min, y_min, x_max, y_max]` (0-1 range) relative to the original image dimensions.
    *   **`xywh`**: Absolute pixel coordinates `[x_min, y_min, width, height]` in the original image.
    *   **`xywh_norm`**: Normalized coordinates `[x_min, y_min, width, height]` (0-1 range).
    *   **`cxcywh`**: Absolute pixel coordinates `[center_x, center_y, width, height]` in the original image.
    *   **`cxcywh_norm`**: Normalized coordinates `[center_x, center_y, width, height]` (0-1 range).

*   **`confidences: np.ndarray`**: An array of confidence scores for each detection (shape `(num_detections,)`).
*   **`class_label_ids: np.ndarray`**: An array of integer class IDs for each detection (shape `(num_detections,)`).
*   **`labels: list[str]`**: A list of string class labels for each detection. Populated if `class_labels` were provided.
*   **`class_labels: list[str]` (Settable Property)**: The master list of class names. You can set this on a `Boxes` instance (or `InferenceResult` instance) if it wasn't available during initialization, and it will update the `labels` attribute.

### Example: Accessing Box Information

```python
# After: single_result: InferenceResult = model.predict("image.jpg")[0]

num_detections = len(single_result.boxes.xyxy)

for i in range(num_detections):
    # Pixel coordinates in original image
    x1, y1, x2, y2 = single_result.boxes.xyxy[i]
    
    # Normalized center_x, center_y, width, height
    cx_n, cy_n, w_n, h_n = single_result.boxes.cxcywh_norm[i]
    
    confidence = single_result.boxes.confidences[i]
    class_id = single_result.boxes.class_label_ids[i]
    label_name = single_result.boxes.labels[i] # if class_labels were set
    
    print(f"Detection {i}: Label={label_name} (ID:{class_id}), Conf={confidence:.2f}, Box (xyxy): {x1,y1,x2,y2}")
```

These interfaces aim to provide a clean and powerful way to work with AngelCV's object detection models and their outputs. 