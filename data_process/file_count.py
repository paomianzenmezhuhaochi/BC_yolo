import os
# 统计标签与图片对应情况

# 获取项目根目录（假设 data_process 和 ISICDM2025_dataset 同级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 统计训练集和验证集图片数量
train_dir = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "images", "train")
val_dir = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "images", "val")

train_imgs = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
val_imgs = [f for f in os.listdir(val_dir) if f.lower().endswith('.png')]

print(f"训练集图片数量: {len(train_imgs)}")
print(f"验证集图片数量: {len(val_imgs)}")

# 统计训练集和验证集标签数量
train_label_dir = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "labels", "train")
val_label_dir = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "labels", "val")
train_labels = [f for f in os.listdir(train_label_dir) if f.lower().endswith('.txt')]
val_labels = [f for f in os.listdir(val_label_dir) if f.lower().endswith('.txt')]

print(f"训练集标签数量: {len(train_labels)}")
print(f"验证集标签数量: {len(val_labels)}")
