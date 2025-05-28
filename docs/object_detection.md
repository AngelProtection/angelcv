# Object Detection with AngelCV

AngelCV provides a comprehensive suite for object detection tasks, currently centered around the YOLOv10 model architecture. This section delves into the details of using the `YoloDetectionModel` and its high-level interface `ObjectDetectionModel`.

## Core Model: `YoloDetectionModel`

The `YoloDetectionModel` (located in `angelcv.model.yolo`) is the backbone of our object detection capabilities. It's a PyTorch Lightning module that implements the YOLOv10 architecture.

Key features:

*   **Configurable Architecture**: The model's backbone and detection head can be customized through configuration files.
*   **PyTorch Lightning Integration**: Leverages PyTorch Lightning for streamlined training, validation, and testing loops, along with features like multi-GPU support, mixed-precision training, and logging.
*   **Weight Initialization**: Implements proper weight initialization techniques crucial for stable and effective training of YOLO models.
*   **Loss Calculation**: Utilizes `DetectionLoss` and `EndToEndDetectionLoss` for computing losses during training.
*   **Metrics**: Integrates `torchmetrics.detection.mean_ap.MeanAveragePrecision` for robust mAP calculation.

### Initialization

Typically, you won't interact directly with `YoloDetectionModel` for initialization. Instead, you'll use the `ObjectDetectionModel` interface.

However, understanding its core constructor is useful:

```python
# from angelcv.model.yolo import YoloDetectionModel
# from angelcv.config import Config
# model = YoloDetectionModel(config: Config)
```

*   `config`: A `Config` object containing the model architecture, training parameters, dataset information, etc.

### Loading from Checkpoint

A crucial method for resuming training or deploying a trained model is `load_from_checkpoint_custom`:

```python
# from angelcv.model.yolo import YoloDetectionModel
# from pathlib import Path
# model = YoloDetectionModel.load_from_checkpoint_custom(checkpoint_path: Path | str)
```

This class method loads model weights and configuration from a saved checkpoint file (`.ckpt` or `.pt`). It intelligently matches weights, even if the model architecture in the checkpoint differs slightly from the current definition (e.g., changes in the number of classes).

### Forward Pass (Inference)

The `forward` method processes an input image tensor and returns raw model predictions.

```python
# images: torch.Tensor # Input image(s)
# predictions = model.forward(images)
```

### Training, Validation, and Testing Steps

These methods are standard PyTorch Lightning hooks:

*   `training_step(batch, batch_idx)`: Processes a batch of data, calculates loss, and logs training metrics.
*   `validation_step(batch, batch_idx)`: Processes a validation batch, calculates loss, and updates mAP metrics.
*   `test_step(batch, batch_idx)`: Similar to `validation_step`, but for the test set.

Epoch-level hooks like `on_validation_epoch_end` and `on_test_epoch_end` compute and log aggregated metrics (e.g., mAP over the entire validation set).

### Optimizer Configuration

The `configure_optimizers` method sets up the optimizer (AdamW) and learning rate schedulers (Linear Warmup followed by Linear Decay). It separates parameters into groups with and without weight decay (BatchNorm layers and biases typically don't use weight decay).

### Updating Number of Classes

The `update_num_classes(num_classes: int)` method is vital for transfer learning. If you load a pre-trained model (e.g., on COCO with 80 classes) and want to fine-tune it on a custom dataset with a different number of classes, this method adapts the model's detection head. It attempts to preserve weights where possible for a smoother transition.

## High-Level Interface: `ObjectDetectionModel`

The `ObjectDetectionModel` (in `angelcv.interface.object_detection`) provides a user-friendly API, abstracting many of the complexities of `YoloDetectionModel` and PyTorch Lightning.

```python
from angelcv.interface import ObjectDetectionModel
```

### Initialization

```python
# Initialize from a configuration file (e.g., "yolov10n.yaml")
model = ObjectDetectionModel("yolov10n.yaml")

# Load a pre-trained model from a checkpoint file
model = ObjectDetectionModel("yolov10n.ckpt") # or "path/to/your/model.ckpt"
```

*   If a `.yaml` file is provided, a new model is built based on that configuration.
*   If a `.ckpt` or `.pt` file is provided, a pre-trained model is loaded from that checkpoint.

AngelCV can automatically resolve file paths, looking in predefined locations (like `angelcv/config/model/`) or S3 buckets if configured.

### Prediction (Inference)

The primary way to perform inference is by calling the model instance or using the `predict` method:

```python
# results = model("image.jpg") # Equivalent to model.predict(...)
results = model.predict(
    source: str | Path | torch.Tensor | np.ndarray | list[...],
    confidence_th: float = 0.3,
    image_size: int | None = None,
)
```

*   `source`: Can be a single image path, URL, PIL image, NumPy array, PyTorch tensor, or a list of these for batch processing.
*   `confidence_th`: Threshold to filter detections. Default is 0.3.
*   `image_size`: Target size for the longest side of the image during preprocessing. If `None`, uses the model's default image size.

This method returns a list of `InferenceResult` objects, one for each input image.

### Training

```python
results = model.train(
    dataset: str | Path, # Path to dataset YAML config (e.g., "coco.yaml")
    image_size: int = None,
    batch_size: int = None,
    num_workers: int = None,
    patience: int = 0, # For early stopping
    max_epochs: int = 100,
    # ... other PyTorch Lightning Trainer arguments (accelerator, devices, etc.)
)
```

Key arguments:

*   `dataset`: Path to a dataset configuration YAML file (see Configuration section).
*   `image_size`, `batch_size`, `num_workers`: Override default training parameters.
*   `patience`: Number of epochs to wait for validation loss improvement before early stopping.
*   `**kwargs`: You can pass any valid argument for the [PyTorch Lightning `Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) (e.g., `accelerator="gpu"`, `devices=1`, `precision="16-mixed"`).

This method handles:

1.  Updating the model configuration with the dataset information.
2.  Adjusting the model's number of classes if the new dataset has a different count.
3.  Setting up the appropriate data module (`CocoDataModule` or `YOLODataModule`).
4.  Configuring PyTorch Lightning callbacks (ModelCheckpoint, EarlyStopping, LearningRateMonitor, etc.).
5.  Running the training loop.

Returns a dictionary with training results.

### Validation

```python
results = model.validation(
    dataset: str | Path | None = None, # Optional: path to dataset YAML for validation
    # ... other arguments similar to train() for overriding datamodule settings
)
```

If `dataset` is not provided, it attempts to use the dataset information from the last training run or the model's loaded configuration.

### Testing

```python
results = model.test(
    dataset: str | Path | None = None, # Optional: path to dataset YAML for testing
    # ... other arguments similar to train() for overriding datamodule settings
)
```

Similar to `validation`, this runs the model on a test set and computes metrics.

### Exporting Models

AngelCV supports exporting models to standard formats like ONNX for deployment.

```python
exported_path = model.export(
    format: str = "onnx", # Currently ONNX is the primary supported format
    output_path: str | None = None, # Optional: specify output file path
    # ... other format-specific export arguments (e.g., opset for ONNX)
)
```

*   `format`: The target export format (e.g., "onnx").
*   `output_path`: If not provided, a default path is generated.

Returns the `Path` to the exported model file. 