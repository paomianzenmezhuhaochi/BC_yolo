#scripts/train.py
import os
from ultralytics import YOLO

# 1. 加载 YOLOv8l 预训练模型
model = YOLO("yolov8n.pt") #数据集小的情况下l并不理想

# 2. 直接调用 model.train() 进行训练
model.train(
    cfg="configs/hyp.yaml", #TODO: 修改为你的模型配置文件
)