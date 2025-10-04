# scripts/eval_yolo_val.py
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO-format val set with Ultralytics built-in val()")
    p.add_argument("--model", type=str, required=True, help="Path to trained weights e.g. runs/weights/best.pt")
    p.add_argument("--data", type=str, default="configs/dataset.yaml", help="Dataset YAML (YOLO format)")
    p.add_argument("--imgsz", type=int, default=1024, help="Evaluation image size")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for NMS during val")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS during val")
    p.add_argument("--batch", type=int, default=8, help="Batch size for val")
    p.add_argument("--device", type=str, default=None, help="CUDA device string, e.g. '0' or '0,1' or 'cpu'")
    p.add_argument("--half", action="store_true", help="Use half precision (FP16) if supported")
    p.add_argument("--plots", action="store_true", help="Save confusion matrix, PR curves, etc.")
    p.add_argument("--save_json", action="store_true", help="Export COCO JSON (if applicable)")
    p.add_argument("--project", type=str, default="runs/val", help="Project directory for outputs")
    p.add_argument("--name", type=str, default="exp", help="Run name")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    # 调用 Ultralytics 内置评估
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        plots=args.plots,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
        split="val",
        verbose=True,
    )

    # 注意：不同版本Ultralytics返回对象略有差异，下面两种打印方式尽量兼容
    try:
        # 新版本：results.results_dict 可能包含关键指标
        print("\nResults dict:")
        print(getattr(results, "results_dict", results))
    except Exception:
        pass

    try:
        # 常见：results.metrics 或 results.box.map 等
        metrics = getattr(results, "metrics", None)
        if metrics is not None:
            print("\nKey metrics:")
            for k, v in metrics.items() if isinstance(metrics, dict) else []:
                print(f"{k}: {v}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

