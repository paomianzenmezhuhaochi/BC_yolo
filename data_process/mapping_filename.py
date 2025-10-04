import os
import json
import csv
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(PROJECT_ROOT, 'ISICDM2025_dataset')
mapping_file = os.path.join(data_dir, 'filename_mapping.csv')
annotations_dir = os.path.join(data_dir, 'annotations')
images_dir = os.path.join(data_dir, 'images')
json_files = ['instances_train.json', 'instances_val.json', 'instances_test.json']

# 读取映射表
def load_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['original_filename']] = row['anonymous_filename']
    return mapping

filename_map = load_mapping(mapping_file)

for json_file in json_files:
    json_path = os.path.join(annotations_dir, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    changed = False
    for img in data.get('images', []):
        orig_name = img.get('file_name')
        img_id = img.get('id')
        # 替换 file_name 为匿名名
        if orig_name in filename_map:
            img['file_name'] = filename_map[orig_name]
            changed = True
        else:
            raise ValueError(f"无法找到映射：{orig_name} 在 {json_file}, id={img_id}")
        # 获取图片路径（train/val/test子目录自动判断）
        found = False
        for split in ['train', 'val', 'test']:
            img_path = os.path.join(images_dir, split, img['file_name'])
            if os.path.exists(img_path):
                with Image.open(img_path) as im:
                    width, height = im.size
                img['width'] = width
                img['height'] = height
                found = True
                break
        if not found:
            raise FileNotFoundError(f"图片未找到：file_name={img['file_name']}, id={img_id} 在 {json_file}")
    if changed:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
print('文件名映射和图片尺寸信息添加完成。')
