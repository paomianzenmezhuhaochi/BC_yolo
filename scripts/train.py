# scripts/train_crop.py
from ultralytics import YOLO

# 数据集配置文件（路径和类别信息）
data_yaml = "configs/dataset.yaml"

print(f"[INFO] 使用数据集配置文件: {data_yaml}")

# 加载 YOLOv8 模型 (可以换 yolov8m.pt 或 yolov8s.pt)
model = YOLO("yolov8n.pt")

# 开始训练，启用自定义 YoloRandomCropDataset
model.train(
    data=data_yaml,         # 数据集配置
    epochs=50,              # 训练轮数
    imgsz=1024,             # 输入尺寸
    batch=4,                # batch size
    freeze=10,              # 冻结前 10 层（可调节）
    project="runs",         # 输出目录
    name="with_crop",       # 实验名
    exist_ok=True,          # 覆盖已有结果
    use_crop=True           # ✅ 启用自定义裁剪数据集
)

# scripts/train.py
from ultralytics import YOLO

# 选择检测模型 (yolov8n.pt)
model = YOLO("/content/drive/MyDrive/BC_Yolov8/runs/baseline/weights/last.pt")  # 可换yolov8m.pt

# 启动训练
model.train(resume=True)