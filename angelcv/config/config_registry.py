from typing import Any

from pydantic import BaseModel, Field

# ------------ MODEL ARCHITECTURE ------------


class BlockConfig(BaseModel):
    args: dict[str, Any]
    source: list[int] = Field(default_factory=lambda: [-1])
    tags: str | None = None


class ModelConfig(BaseModel):
    architecture: dict[str, list[dict[str, BlockConfig]]]
    name: str = ""
    repeats_scale: float = Field(gt=0)
    channels_scale: float = Field(gt=0)
    max_channels: int = Field(gt=0)


# ---------------- TRAINING ----------------


class LossMatcherConfig(BaseModel):
    iou: str
    tal_topk: int
    tal_alpha: float
    tal_beta: float


class LossWeightsConfig(BaseModel):
    cls_loss: float
    iou_loss: float
    df_loss: float


class LossConfig(BaseModel):
    weights: LossWeightsConfig
    matcher: LossMatcherConfig


class OptimizerConfig(BaseModel):
    type: str
    args: dict[str, Any]


class SchedulerConfig(BaseModel):
    type: str
    warmup_epochs: int
    args: dict[str, Any]


class DataConfig(BaseModel):
    batch_size: int
    image_size: int
    num_workers: int
    shuffle: bool


class TrainConfig(BaseModel):
    max_epochs: int
    patience: int
    data: DataConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


class ValidationConfig(BaseModel):
    data: DataConfig


class TestConfig(BaseModel):
    data: DataConfig


# ---------------- DATASET ----------------


# NOTE: this document format is to match the fiftyone YOLOv5DatasetExporter, with extra fields for our convinience
class DatasetConfig(BaseModel):
    name: str = ""
    task: str = "detection"
    path: str
    train: str
    val: str
    test: str = ""
    names: dict[int, str] = Field(default_factory=dict)
    pin_memory: bool = True
    transforms: dict[str, Any] = Field(default_factory=dict)
    download_urls: dict[str, str] = Field(default_factory=dict)


# ---------------- GENERAL ----------------


class Config(BaseModel):
    model: ModelConfig | None = None
    train: TrainConfig
    validation: ValidationConfig
    test: TestConfig | None = None
    dataset: DatasetConfig | None = None
    image_size: int = Field(default=640, gt=0)
    num_workers: int = Field(default=-1, description="Number of workers for data loading, if -1 half of the CPU cores")
