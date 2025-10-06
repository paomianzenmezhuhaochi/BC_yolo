#scripts/train_baseline.py
from ultralytics import YOLO

# 1. 加载 YOLOv8l 预训练模型
model = YOLO("yolov8l.pt")

# 2. 直接调用 model.train() 进行训练
model.train(
    data="configs/dataset.yaml",
    epochs=80,
    imgsz=1024,
    batch=4,
    fl_gamma=1.5,
    hsv_v=0.2,
    degrees=5.0,
    scale=0.1,
    shear=0.0,
    translate=0.0,
    perspective=0.0,
    flipud=0.5,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.10,
    copy_paste=0.15,
    erasing=0.1,
    project="runs",
    name="baseline",
    exist_ok=True,
    hyp="configs/baseline_hyp.yaml",
    optimizer="Adam",   # 使用 Adam 优化器
    seed=42,            # 固定随机种子
    freeze=10,          # 冻结前 10 层 backbone
    plots=True          # 自动绘制训练曲线
)