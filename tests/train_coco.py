from datetime import datetime
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import torch

from angelcv.config import ConfigManager
from angelcv.dataset.coco_datamodule import CocoDataModule
from angelcv.dataset.yolo_datamodule import YOLODataModule
from angelcv.model.yolo import YoloDetectionModel
from angelcv.utils.env_utils import is_debug_mode

# Make proper use of the tensor cores
torch.set_float32_matmul_precision("medium")  # "high"

# TODO [MID]: decide how to handle this (shouldn't be the default, but proposed as an option if required)
# Disable peer-to-peer (P2P) direct memory access between GPUs
# NOTE: Some GPUs (specially consumer ones) have issues with P2P (tested with RTX A4000)
os.environ["NCCL_P2P_DISABLE"] = "1"

dataset = "coco"  # "yolo", "coco"

if is_debug_mode():
    print("Running in DEBUG mode")
    batch_size = 8
    num_workers = 2
    patience = -1
    overfit_batches = 256  # NOTE: 256 first multiple of 2 that val_loss != nan
    # overfit_batches = 0.0  # entire dataset
    num_sanity_val_steps = 2
else:
    print("Running in PRODUCTION mode")
    batch_size = 32
    num_workers = -1
    patience = 50
    overfit_batches = 0.0  # entire dataset
    num_sanity_val_steps = 0

config = ConfigManager.upsert_config(model_file="yolov10l.yaml", dataset_file="coco.yaml")
config.train.data.batch_size = batch_size
config.train.data.num_workers = num_workers
model = YoloDetectionModel(config)
# model.load_checkpoint(Path("checkpoints/2025-04-14_11-46-09/model-epoch=009-step=002305-val_loss=165.09.ckpt"))

if dataset == "coco":
    dm = CocoDataModule(config)
elif dataset == "yolo":
    dm = YOLODataModule(config)

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

callbacks = [
    LearningRateMonitor(logging_interval="step"),
    DeviceStatsMonitor(),
    RichProgressBar(),
]

if patience > 0:
    callbacks.append(EarlyStopping(monitor="val_loss", patience=patience))
    callbacks.append(
        ModelCheckpoint(
            dirpath=f"checkpoints/{date}",
            filename="model-{epoch:03d}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=2,
            save_last=True,
        ),
    )

l_trainer = L.Trainer(
    accelerator="gpu",
    devices=-1,  # all (-1)
    max_epochs=900,
    precision="16-mixed",  # modern GPU "bf16-mixed", general "32-true"
    callbacks=callbacks,
    logger=TensorBoardLogger(save_dir="tb_logs", name="angelcv", version=date),
    overfit_batches=overfit_batches,
    num_sanity_val_steps=num_sanity_val_steps,
    sync_batchnorm=True,  # specially recommended for small batch sizes
    # use_distributed_sampler=False,  # Exception without it, but can train in multi-GPU
)
# NOTE: find the biggest batch that fits in memory
# l_tuner = Tuner(l_trainer)
# l_tuner.scale_batch_size(model, datamodule=dm, mode="power")
l_trainer.fit(model, datamodule=dm)
