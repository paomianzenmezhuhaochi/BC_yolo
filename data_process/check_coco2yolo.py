import json
import cv2
import os

#检查单独一张图片的情况 需要更改路径

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径定义
img_name = "ISICDM2025_000916.png"
img_path = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "images", "test", img_name)
coco_json = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "annotations", "instances_test.json")
yolo_txt = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "labels", "test", "ISICDM2025_000916.txt")
debug_dir = os.path.join(PROJECT_ROOT, "debug")
os.makedirs(debug_dir, exist_ok=True)

# 打印 COCO id 和 YOLO txt 文件名
with open(coco_json, "r", encoding="utf-8") as f:
    coco_data = json.load(f)
img_info = next(img for img in coco_data["images"] if img["file_name"] == img_name)
print("COCO id:", img_info["id"])
print("YOLO txt:", yolo_txt)

# 读取图片
img = cv2.imread(img_path)

# COCO bbox绘制（红色）
for ann in coco_data["annotations"]:
    if ann["image_id"] == img_info["id"]:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

# YOLO bbox绘制（绿色）
with open(yolo_txt, "r", encoding="utf-8") as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.strip().split())
        x = int((xc - bw / 2) * img_info["width"])
        y = int((yc - bh / 2) * img_info["height"])
        w = int(bw * img_info["width"])
        h = int(bh * img_info["height"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 保存到 debug 路径
save_path = os.path.join(debug_dir, f"check_{img_name}")
cv2.imwrite(save_path, img)
print(f"已保存: {save_path}")
