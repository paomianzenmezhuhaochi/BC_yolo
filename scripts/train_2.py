#scripts/train_2.py
from ultralytics import YOLO

# 1. 加载上一阶段训练好的模型权重
model = YOLO("runs/stage1/weights/best.pt") #请自行修改

# 2. 直接调用 model.train() 进行训练
model.train(
    cfg="configs/unfreeze.yaml"
)