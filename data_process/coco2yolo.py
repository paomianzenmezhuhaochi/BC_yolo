import json
import os
from tqdm import tqdm
from collections import defaultdict

# 获取项目根目录（假设 data_process 和 ISICDM2025_dataset 同级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def coco2yolo(json_file, output_dir):
    """
    将 COCO 格式的标注转换为 YOLO 格式
    :param json_file: COCO 标注文件路径
    :param output_dir: YOLO 标签输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # id 映射 filename 和 size
    id2filename = {img["id"]: img["file_name"] for img in data["images"]}
    id2size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    # 按图片分组收集所有标注
    imgid2anns = defaultdict(list)
    for ann in data["annotations"]:
        imgid2anns[ann["image_id"]].append(ann)

    # 遍历所有图片，生成标签文件
    for img_id, anns in tqdm(imgid2anns.items(), desc=f"Converting {os.path.basename(json_file)}"):
        filename = os.path.splitext(id2filename[img_id])[0] + ".txt"
        label_path = os.path.join(output_dir, filename)
        w, h = id2size[img_id]
        lines = []
        for ann in anns:
            category_id = ann["category_id"]
            bbox = ann["bbox"]  # [x,y,w,h]
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h
            lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        # 用 w 模式写入，确保每次都是干净的
        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

if __name__ == "__main__":
    dataset_root = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset")

    # 训练集
    coco2yolo(
        json_file=os.path.join(dataset_root, "annotations", "instances_train.json"),
        output_dir=os.path.join(dataset_root, "labels", "train")
    )

    # 验证集
    coco2yolo(
        json_file=os.path.join(dataset_root, "annotations", "instances_val.json"),
        output_dir=os.path.join(dataset_root, "labels", "val")
    )

    # 测试集（可选）
    coco2yolo(
        json_file=os.path.join(dataset_root, "annotations", "instances_test.json"),
        output_dir=os.path.join(dataset_root, "labels", "test")
    )
