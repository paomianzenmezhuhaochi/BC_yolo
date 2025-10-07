# scripts/train_crop.py
from ultralytics import YOLO


# 恢复训练
model = YOLO()  # 加载上次训练的权重

model.train(
    resume=True,           # 关键参数，自动恢复训练
    project="runs",      # 保持输出目录一致
    name="baseline"        # 保持实验名称一致
)


