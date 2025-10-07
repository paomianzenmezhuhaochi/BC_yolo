from ultralytics import YOLO
from ultralytics.utils.loss import FocalLoss
from ultralytics.engine.trainer import BaseTrainer

class FocalLossTrainer(BaseTrainer):
    def get_criterion(self):
        criterion = super().get_criterion()
        criterion.bce = FocalLoss(gamma=2.0, alpha=0.75)  # 可调参数
        return criterion

model = YOLO("runs/detect/baseline/weights/last.pt")  # 加载第一阶段权重