import json

with open('ISICDM2025_dataset/annotations/instances_val.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

id2name = {img['id']: img['file_name'] for img in data['images']}

gt_list = []
for ann in data['annotations']:
    image_id = id2name[ann['image_id']]
    gt_list.append({
        "image_id": image_id,
        "category_id": ann['category_id'],
        "bbox": ann['bbox'],
        "score": 1.0
    })

with open('ISICDM2025_dataset/annotations/gt_val.json', 'w', encoding='utf-8') as f:
    json.dump(gt_list, f, indent=2)
