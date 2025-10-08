from ultralytics import YOLO
from ultralytics.utils.loss import FocalLoss
from ultralytics.models.yolo.detect import DetectionTrainer

class FocalLossTrainer(DetectionTrainer):
    def get_criterion(self):
        criterion = super().get_criterion()
        criterion.bce = FocalLoss(gamma=2.0, alpha=0.75)  # 可调参数
        return criterion

model = YOLO("runs/detect/baseline/weights/last.pt")  # TODO:加载上一阶段权重

model.train(
    trainer=FocalLossTrainer,
    cfg="configs/unfreeze.yaml", #TODO: 修改为你的模型配置文件
)