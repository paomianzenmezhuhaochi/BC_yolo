#scripts/train.py
from ultralytics import YOLO

# 1. 加载 YOLOv8l 预训练模型
model = YOLO("yolov8n.pt") # 数据集较小，其他容易过拟合

# 2. 直接调用 model.train() 进行训练
model.train(
    data="configs/dataset.yaml",
    epochs=50,
    batch=4,
    imgsz=1024,
    project="runs",
    name="baseline",
    freeze=10,
    exist_ok=True,
)