# Welcome to AngelCV!

**AngelCV is an open-source, commercially-friendly computer vision library designed for ease of use, power, and extensibility.**

AngelCV is a project initiated by [**Angel Protection System**](https://angelprotection.com/), a company dedicated to safeguarding critical environments like schools and hospitals. They provide advanced security and surveillance systems, notably featuring firearm detection technology that delivers crucial, real-time alerts to 911 and emergency responders, contributing significantly to life-saving efforts.

Our mission is to provide cutting-edge deep learning models and tools that you can seamlessly integrate into your projects, whether for research, personal use, or commercial applications. All our code and pre-trained models are under the **Apache 2.0 License**, giving you the freedom to innovate without restrictive licensing.

*AngelCV was originally conceived to enhance the sophisticated computer vision capabilities required by Angel Protection System for their security solutions. By making it open-source, we aim to empower the broader community, encouraging collaborative development and widespread application of this technology.*

## Why AngelCV?

*   **Open & Free for Commercial Use**: Build your next big thing without worrying about licensing fees or restrictions. Our Apache 2.0 license covers both the library and our provided pre-trained models.
*   **State-of-the-Art Models**: We start with robust implementations like YOLOv10 for object detection and plan to expand to other vision tasks (classification, segmentation, oriented bounding boxes) and model architectures.
*   **Developer-Friendly Interface**: A clean, intuitive API (see `ObjectDetectionModel` and `InferenceResult`) makes common tasks like training, inference, and evaluation straightforward.
*   **Flexible Configuration**: Easily customize model architectures, training parameters, and datasets using YAML-based configuration files.
*   **Community Driven (Future)**: We aim to build a community around AngelCV. (TODO: Link to GitHub Discussions/Issues when ready).

## Dive In

Ready to get started? Here's how you can explore AngelCV:

*   **[Getting Started](./getting_started.md)**: Your first stop for installation and a quick tour.
*   **[Object Detection](./object_detection.md)**: Learn about our object detection capabilities, focusing on YOLOv10.
*   **[Configuration](./configuration.md)**: Understand how to use and customize model, training, and dataset configurations.
*   **[API Interfaces](./interfaces.md)**: Explore the main Python classes you'll interact with.

## Project Layout (for Contributors)

If you plan to contribute or explore the codebase:

    mkdocs.yml    # The MkDocs configuration file.
    docs/
        index.md  # This homepage.
        getting_started.md
        object_detection.md
        configuration.md
        interfaces.md
        ...       # Other markdown pages, images, etc.
    angelcv/
        config/     # Configuration files and management
        dataset/    # Data loading and processing modules
        interface/  # High-level user interfaces
        model/      # Model implementations (e.g., YoloDetectionModel)
        tools/      # Utility tools (e.g., loss functions)
        utils/      # General utilities
    tests/
        # Unit and integration tests
    # Other project files like setup.py, requirements.txt, etc.


We are excited to see what you build with AngelCV!

## Development & Support

The primary development of AngelCV is led by [Iu Ayala](https://github.com/IuAyala) of **Gradient Insight**. Gradient Insight specializes in creating custom AI-powered computer vision systems that help businesses transform visual data into valuable insights. Discover more about their services at [gradientinsight.com](https://gradientinsight.com).
