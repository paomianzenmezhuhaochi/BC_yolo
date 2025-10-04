# scripts/train_crop.py
from ultralytics import YOLO
import torch

# 数据集配置文件
data_yaml = "configs/dataset.yaml"

print(f"[INFO] 使用数据集配置文件: {data_yaml}")

# 加载模型
model = YOLO("yolov8s.pt")  # 建议使用更大的模型

# 开始训练
model.train(
    data=data_yaml,
    epochs=80,
    imgsz=1024,
    batch=4,
    lr0=0.01,  #
    patience=20,
    project="runs",
    name="with_crop_fixed",
    exist_ok=True,
    use_crop=True,
    verbose=True,  # 显示详细日志
    workers=4,              # 数据加载工作进程数
    optimizer='Adam',       # 使用Adam优化器
    resume=True,
    val=False,
    seed=42                 # 固定随机种子
)

