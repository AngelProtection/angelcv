# AngelCV

**AngelCV is an open-source, commercially-friendly computer vision library designed for ease of use, power, and extensibility.**

AngelCV is a project by [**Angel Protection System**](https://angelprotection.com/), a company at the forefront of safeguarding schools, hospitals, and other vital community spaces. They specialize in intelligent security and surveillance systems, including cutting-edge firearm detection technology that provides critical, real-time information to 911 and first responders, playing a vital role in saving lives.

Our mission is to provide cutting-edge deep learning models and tools that you can seamlessly integrate into your projects, whether for research, personal use, or commercial applications. All our code and pre-trained models are under the **Apache 2.0 License**, giving you the freedom to innovate without restrictive licensing.

_A note on our open-source commitment: Angel Protection System initially developed AngelCV to enhance its advanced computer vision capabilities for security applications. We are excited to share it with the open-source community to foster innovation and allow everyone to benefit from and contribute to its development._

## ✨ Why AngelCV?

- **Open & Free for Commercial Use**: Build your next big thing without worrying about licensing fees or restrictions. Our Apache 2.0 license covers both the library and our provided pre-trained models.
- **State-of-the-Art Models**: We start with robust implementations like YOLOv10 for object detection and plan to expand to other vision tasks (classification, segmentation, oriented bounding boxes) and model architectures.
- **Developer-Friendly Interface**: A clean, intuitive API (see `ObjectDetectionModel` and `InferenceResult`) makes common tasks like training, inference, and evaluation straightforward.
- **Flexible Configuration**: Easily customize model architectures, training parameters, and datasets using YAML-based configuration files.
- **Community Driven (Future)**: We aim to build a community around AngelCV.

## 🚀 Getting Started

### Installation

AngelCV will be available on PyPI. You can install it using pip:

```bash
pip install angelcv
```

Make sure you have PyTorch installed, as it's a primary dependency. You can find PyTorch installation instructions at [pytorch.org](https://pytorch.org/).

### Quick Start: Object Detection

Here's a simple example of how to load a pre-trained YOLOv10 model and perform inference on an image:

```python
from angelcv import ObjectDetectionModel

# Load a pre-trained YOLOv10n model (will download if not found locally)
# You can also specify a path to a local .ckpt or .pt file,
# or a .yaml configuration file to initialize a new model.
model = ObjectDetectionModel("yolov10n.ckpt")

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

## 🚧 Development Status

> **⚠️ Repository Under Heavy Development**
>
> AngelCV is actively being developed. While core functionality is stable, we're continuously improving and expanding features.

### ✅ **Stable & Ready to Use**

- **Object Detection**: Training, validation, testing, and inference are fully stable
- **YOLOv10 Integration**: Robust implementation with pre-trained models
- **Core API**: `ObjectDetectionModel` and `InferenceResult` interfaces
- **Configuration System**: YAML-based model and training configuration
- **Model Export**: ONNX, TensorRT, and other deployment formats

### 🔄 **Worning On**

- **Data Augmentation**: Expanding augmentation techniques to improve training performance on large datasets
- **Performance Optimization**: Addressing slightly below-expected performance on big datasets
- **Documentation**: Comprehensive guides and examples

### 📋 **Coming Soon (TODO)**

- **Image Segmentation**: Semantic and instance segmentation models
- **Oriented Bounding Boxes**: Support for rotated object detection
- **Classification Models**: Standalone image classification capabilities
- **Additional Architectures**: Beyond YOLOv10 (YOLOv9, DETR, etc.)
- **Advanced Metrics**: Comprehensive evaluation and benchmarking tools

## 📚 Dive Deeper

For more detailed information, check out our documentation:

- **[Getting Started](https://angelprotection.github.io/angelcv/getting_started/)**: Your first stop for installation and a quick tour.
- **[Object Detection](https://angelprotection.github.io/angelcv/object_detection/)**: Learn about our object detection capabilities, focusing on YOLOv10.
- **[Configuration](https://angelprotection.github.io/angelcv/configuration/)**: Understand how to use and customize model, training, and dataset configurations.
- **[API Interfaces](https://angelprotection.github.io/angelcv/interfaces/)**: Explore the main Python classes you'll interact with.

## 🤝 Contributing

Interested in contributing? We welcome contributions of all kinds, from bug fixes to new features. (TODO: Link to contribution guidelines when ready).

## 🛠️ Development and Support

The primary developer and maintainer of AngelCV is [Iu Ayala](https://github.com/IuAyala) from **Gradient Insight**. Gradient Insight partners with businesses to design and build custom AI-powered computer vision systems, turning complex visual data into actionable insights. You can learn more about their work at [gradientinsight.com](https://gradientinsight.com).

## 📄 License

AngelCV is licensed under the **Apache 2.0 License**. See the `LICENSE` file for more details.
