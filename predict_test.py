"""批量对测试集 PNG 图片进行目标检测并保存为 JSON (COCO 检测结果风格简化版)。

输出每条检测记录字段:
  image_id: 文件名 (不含路径)
  category_id: 预测类别 (int)
  bbox: [x, y, w, h]  (左上角+宽高, 坐标为原图尺度, float)
  score: 置信度

"""
from __future__ import annotations
import os
import json
import glob
from typing import List, Optional

from ultralytics import YOLO

# ---------------- 用户可修改区域 -----------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(project_root, "runs/baseline/weights/best.pt")
IMAGES_DIR = os.path.join(project_root, "ISICDM2025_images_for_test")
OUT_JSON = os.path.join(project_root, "predict_test.json") #更改为你指定的输出目录
IMGSZ = 1024
CONF = 0.01
IOU = 0.6
DEVICE: Optional[str] = None   # 可设为 "cpu" 或 "0" 等
BATCH = 4
INCLUDE_EMPTY = False  # True 时为无检测图片写空记录
# -------------------------------------------------


def collect_png_images(folder: str) -> List[str]:
    patterns = ["*.png", "*.PNG"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = sorted(set(files))
    return files


def run_inference(
    model_path: str,
    images: List[str],
    out_json: str,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[str] = None,
    batch: int = 8,
    include_empty: bool = False,
) -> None:
    if not images:
        raise ValueError("未找到任何 PNG 图片，请检查路径或文件是否存在。")

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    results_records = []
    total = len(images)
    print(f"开始推理，共 {total} 张图片，batch={batch}, imgsz={imgsz}, conf={conf}, iou={iou}")

    for start in range(0, total, batch):
        batch_imgs = images[start : start + batch]
        preds = model(
            batch_imgs,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        for img_path, res in zip(batch_imgs, preds):
            file_name = os.path.basename(img_path)
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                if include_empty:
                    results_records.append(
                        {
                            "image_id": file_name,
                            "category_id": None,
                            "bbox": [],
                            "score": None,
                        }
                    )
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), sc, cls in zip(xyxy, scores, clss):
                results_records.append(
                    {
                        "image_id": file_name,
                        "category_id": int(cls),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(sc),
                    }
                )
        done = min(start + batch, total)
        print(f"进度: {done}/{total} ({done/total:.1%})")

    os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2, ensure_ascii=False)
    print(f"完成。保存 {len(results_records)} 条检测记录到 -> {out_json}")


if __name__ == "__main__":
    print("项目根目录:", project_root)
    print("图片目录:", IMAGES_DIR)
    images = collect_png_images(IMAGES_DIR)
    print(f"发现 {len(images)} 张 PNG 图片。示例: {images[:3]}")
    run_inference(
        model_path=MODEL_PATH,
        images=images,
        out_json=OUT_JSON,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        batch=BATCH,
        include_empty=INCLUDE_EMPTY,
    )
