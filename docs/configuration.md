# Configuration in AngelCV

AngelCV uses a flexible configuration system based on YAML files and Pydantic models to define model architectures, training parameters, and dataset specifications. This approach allows for easy customization and reproducibility.

## Core Components

*   **`ConfigManager`**: Located in `angelcv.config.manager`, this class is responsible for loading, merging, and managing configurations. It can load settings from multiple YAML files (e.g., a base model config and a dataset-specific config) and provides a unified `Config` object.
*   **`Config` (Pydantic Model)**: Defined in `angelcv.config.config_registry.Config`, this is the main Pydantic model that holds the entire configuration structure. It includes nested models for model architecture, training parameters, dataset details, etc.
*   **`BlockConfig` (Pydantic Model)**: Also in `angelcv.config.config_registry.BlockConfig`, this model defines the structure for individual blocks or layers within the model architecture (e.g., a convolution layer, a C2f block).

## Configuration Files

AngelCV typically uses YAML files for configuration. You'll encounter:

*   **Model Configuration Files** (e.g., `yolov10n.yaml`): These define the architecture of a specific model (like YOLOv10n), including:
    *   `model.architecture`: Specifies the sequence of blocks, their types, source connections (for skip connections or feature fusion), and parameters (e.g., number of channels, kernel size, repeats).
    *   `model.channels_scale`, `model.repeats_scale`: Scaling factors to easily create model variants (e.g., YOLOv10s, YOLOv10m, YOLOv10l) from a base architecture.
    *   `model.max_channels`: A constraint on the maximum number of channels in any layer.
    *   `image_size`: Default input image size for the model.
*   **Dataset Configuration Files** (e.g., `coco.yaml`, `your_custom_dataset.yaml`): These define dataset-specific parameters:
    *   `dataset.type`: The type of dataset (e.g., "coco", "yolo").
    *   `dataset.path`: Path to the root directory of the dataset.
    *   `dataset.train_images`, `dataset.val_images`, `dataset.test_images`: Paths to directories or annotation files for train, validation, and test splits.
    *   `dataset.names`: A list of class names.
    *   `dataset.nc`: Number of classes (should match the length of `names`).
*   **Training Configuration** (often part of the model or dataset config, or a separate file):
    *   `train.optimizer`: Optimizer settings (e.g., type like "AdamW", learning rate, weight decay).
    *   `train.scheduler`: Learning rate scheduler settings (e.g., warmup epochs, decay strategy).
    *   `train.batch_size`, `train.num_workers`, `train.max_epochs`.

## How Configuration Works

1.  **Loading**: When you initialize an `ObjectDetectionModel` (e.g., `ObjectDetectionModel("yolov10n.yaml")`) or call its `train` method with a dataset file (e.g., `model.train(dataset="coco.yaml")`), the `ConfigManager` is used.
2.  **Upserting/Merging**: `ConfigManager.upsert_config(model_file=..., dataset_file=...)` loads the specified files. If both model and dataset files are provided, their configurations are intelligently merged. Dataset-specific parameters (like number of classes) can override model defaults.
3.  **Validation**: Pydantic models validate the loaded configuration, ensuring all required fields are present and have correct types.
4.  **Access**: The resulting `Config` object is then available as `self.model.config` within the `ObjectDetectionModel` and `YoloDetectionModel` instances.

## Example: Model Architecture Snippet (from a model YAML)

```yaml
# Example from a hypothetical yolov10n.yaml
model:
  architecture:
    backbone:
      - Focus:
          source: [-1] # -1 means input from the previous layer
          args:
            in_channels: 3
            out_channels: 32
            kernel_size: 3
      - Conv:
          source: [-1]
          args:
            in_channels: 32
            out_channels: 64
            kernel_size: 3
            stride: 2
      - C2f:
          source: [-1]
          args:
            in_channels: 64
            out_channels: 64
            repeats: 2
            shortcut: True
    head:
      - SPPF:
          source: [-1]
          args:
            in_channels: 64 # Example, actual channels would depend on backbone output
            out_channels: 64
            pool_size: 5
      # ... more head layers ...
      - v10Detect: # Detection head
          source: [10, 13, 16] # Example: input from specific backbone/neck layers
          args:
            num_classes: 80 # Will be overridden by dataset config if different
            # in_channels_list will be auto-filled based on source layers
            # other head-specific parameters

  channels_scale: 0.25 # For smaller variants like 'n'
  repeats_scale: 0.33 # For smaller variants like 'n'
  max_channels: 1024
  image_size: 640
```

## Example: Dataset Configuration Snippet (from a dataset YAML)

```yaml
# Example coco.yaml
dataset:
  type: coco
  path: "/path/to/your/coco_dataset/" # Root directory of COCO dataset
  train_images: "images/train2017"
  train_annotations: "annotations/instances_train2017.json"
  val_images: "images/val2017"
  val_annotations: "annotations/instances_val2017.json"
  # test_images: ... (optional)
  # test_annotations: ... (optional)
  names: ["person", "bicycle", ..., "toothbrush"] # 80 COCO class names
  nc: 80

train:
  batch_size: 16
  max_epochs: 300
  # ... other training overrides specific to this dataset
```

## Customizing Configurations

*   **Modify Existing YAMLs**: You can copy and modify the provided YAML files (e.g., to change learning rates, batch sizes, or model architecture for experiments).
*   **Create New Dataset YAMLs**: For your custom datasets, create a new YAML file following the structure shown above. Specify the dataset type (`yolo` for standard YOLO format datasets, or `coco` for COCO format), paths, and class names.
*   **Programmatic Overrides**: While less common for persistent changes, you can also modify the `model.config` object programmatically after loading a model, though this is generally discouraged for reproducibility if not tracked.

By understanding this configuration system, you can adapt AngelCV to various models, datasets, and training regimes effectively. 