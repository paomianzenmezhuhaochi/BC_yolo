#eval_val_patch.py
from patched_yolo_infer import PatchInferencer
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/FT/weights/best.pt")

# 初始化 patch 推理器
inferencer = PatchInferencer(
    model=model,
    patch_size=1024,
    overlap=0.2,
    conf_thres=0.25,
    device='cuda'
)

# 对验证集或测试集进行 patch 验证
metrics = inferencer.evaluate(
    img_dir='datasets/val/images',
    label_dir='datasets/val/labels',
    save_results=True
)
