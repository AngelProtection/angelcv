from angelcv import ObjectDetectionModel

model = ObjectDetectionModel("yolov10m.yaml")

train_result = model.train(
    dataset="coco.yaml",
    max_epochs=900,
    patience=50,
    image_size=640,
    batch_size=8,
    devices=-1,  # all devices
)

print(train_result)
