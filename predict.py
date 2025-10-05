# predict_test.py
import os
import json
from ultralytics import YOLO
from scripts.slide_infer import SlideInfer

#TODO：改为调用yolo标准库预测，imgsize=1024
def predict_test(model_path, test_images, save_json="predictions.json"):
    model = YOLO(model_path)
    infer = SlideInfer(model)

    results = []
    for img_path in test_images:
        det = infer.infer_image(img_path, conf_thres=0.25, iou_thres=0.7)
        if det is None:
            continue
        for x1, y1, x2, y2, conf, cls in det:
            results.append({
                "image_id": os.path.basename(img_path),
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf)
            })

    with open(save_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ 保存测试集预测结果: {save_json}")
