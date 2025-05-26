# Getting Started with AngelCV

Welcome to AngelCV! This guide will help you get up and running with our computer vision library.

## What is AngelCV?

AngelCV is an open-source computer vision library designed to be powerful, flexible, and easy to use. Our core philosophy is to provide state-of-the-art models and tools that are:

*   **Commercially Friendly**: All our code and pre-trained models are released under the Apache 2.0 license, meaning you can use them freely in your commercial projects without restrictions.
*   **Extensible**: While we currently focus on YOLOv10 for object detection, AngelCV is built with future expansion in mind. We plan to incorporate more tasks (classification, oriented bounding boxes, segmentation, etc.) and models.
*   **User-Focused**: We aim to provide a clear and intuitive interface, comprehensive documentation, and helpful examples to make your development process as smooth as possible.

## Installation

AngelCV will be available on PyPI. You can install it using pip:

```bash
pip install angelcv
```

## Quick Start: Object Detection

Here's a simple example of how to load a pre-trained YOLOv10 model and perform inference on an image:

```python
from angelcv import ObjectDetectionModel

# Load a pre-trained YOLOv10n model (will download if not found locally)
# You can also specify a path to a local .pt or .ckpt file,
# or a .yaml configuration file to initialize a new model.
model = ObjectDetectionModel("yolov10n.pt")

# Perform inference on an image
# Source can be a file path, URL, PIL image, torch.Tensor, or numpy array.
results = model.predict("path/to/your/image.jpg")

# Process and display results
for result in results:
    print(f"Found {len(result.boxes.xyxy)} objects.")
    # Access bounding boxes (various formats available, e.g., result.boxes.xyxy_norm)
    # Access confidences: result.boxes.confidences
    # Access class IDs: result.boxes.class_label_ids
    # Access class labels (if available): result.boxes.labels

    # Show the annotated image
    result.show()

    # Save the annotated image
    result.save("output_image.jpg")
```

This is just a glimpse of what AngelCV can do. Explore the other documentation sections to learn more about training, configuration, and advanced features! 