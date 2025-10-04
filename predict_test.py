# predict_test.py
import os
import json
from glob import glob
from ultralytics import YOLO
from slide_infer import SlideInfer

IMG_EXTS = ("*.png", "*.PNG")

def list_images(dir_path: str):
    files = []
    for pat in IMG_EXTS:
        files.extend(glob(os.path.join(dir_path, pat)))
    return sorted(files)


def predict_test(model_path, test_images_dir, save_json, conf_thres=0.1, iou_thres=0.6):
    model = YOLO(model_path)
    infer = SlideInfer(model)
    infer.conf_thres = float(conf_thres)
    infer.iou_thres = float(iou_thres)

    test_images = list_images(test_images_dir)
    if not test_images:
        raise FileNotFoundError(f"未在目录中找到图片: {test_images_dir}")

    results = []
    for img_path in test_images:
        det = infer.infer_image(img_path)
        if det is None or len(det) == 0:
            continue
        for x1, y1, x2, y2, conf, cls in det:
            results.append({
                "image_id": os.path.basename(img_path),
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf)
            })

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存测试集预测结果: {save_json} | 共 {len(results)} 条检测记录")


if __name__ == "__main__":
    model_path = "runs/with_crop/weights/last.pt"  # 替换为你的模型权重路径
    test_images_dir = "ISICDM2025_images_for_test"   # 替换为你的测试图片目录
    save_json = "final.json"
    predict_test(model_path, test_images_dir, save_json)
