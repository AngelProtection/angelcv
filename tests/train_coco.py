from angelcv import ObjectDetectionModel
from angelcv.utils.env_utils import is_debug_mode

model = ObjectDetectionModel("yolov10m.yaml")

if is_debug_mode():
    print("Running in DEBUG mode")
    print("IMPORTANT: NOT Training the entire dataset, overfit_batches=64")
    train_result = model.train(
        dataset="coco.yaml",
        max_epochs=900,
        patience=50,
        image_size=640,
        batch_size=1,
        devices=-1,  # all devices
        overfit_batches=64,  # NOT Training the entire dataset
        num_workers=1,
    )
else:
    print("Running in PRODUCTION mode")
    train_result = model.train(
        dataset="coco.yaml",
        max_epochs=900,
        patience=50,
        image_size=640,
        batch_size=16,
        devices=-1,  # all devices
        num_workers=-1,
    )

print(train_result)
