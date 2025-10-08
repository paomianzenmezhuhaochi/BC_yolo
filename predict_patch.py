import os
import cv2
from ultralytics import YOLO
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results,  # 可用于带 patch 可视化
    visualize_results_usual_yolo_inference,  # 非 patch 可视化
)

def predict_on_folder(folder_path: str, model_path: str, json_output_path: str):
    """
    Predicts on all PNG images in a folder and saves the results to a JSON file.

    Args:
        folder_path (str): Path to the folder containing PNG images.
        model_path (str): Path to the YOLO model weights.
        json_output_path (str): Path to the output JSON file.
    """

    all_predictions = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Get image dimensions
            img_height, img_width, _ = img.shape

            element_crops = MakeCropsDetectThem(
                image=img,
                model_path=model_path,
                segment=False,
                show_crops=False,
                shape_x=1024,
                shape_y=1024,
                overlap_x=50,
                overlap_y=50,
                conf=0.08,
                iou=0.7,
                classes_list=[0, 1, 2, 3, 5, 7],
                show_processing_status=True
            )
            result = CombineDetections(element_crops, match_metric="IOU",class_agnostic_nms=False,nms_threshold=0.1)

            # Process results for JSON output
            boxes = result.filtered_boxes
            confidences = result.filtered_confidences
            classes_ids = result.filtered_classes_id

            for box, confidence, class_id in zip(boxes, confidences, classes_ids):
                x1, y1, x2, y2 = box
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox = [float(x1), float(y1), float(bbox_width), float(bbox_height)]

                prediction = {
                    "image_id": filename,
                    "category_id": int(class_id),
                    "bbox": bbox,
                    "score": float(confidence),
                }
                all_predictions.append(prediction)

            print(f"Processed: {filename}")

    # Save all predictions to a single JSON file
    with open(json_output_path, 'w') as f:
        json.dump(all_predictions, f, indent=4)

    print(f"Saved predictions to JSON: {json_output_path}")

# usage:
input_folder = '/content/drive/MyDrive/BC_Yolov8/ISICDM2025_images_for_test' # Replace with your input folder path
model_weights_path = 'runs/stage3/weights/best.pt' # Ensure this path is correct relative to your working directory
json_output_path = '/content/drive/MyDrive/BC_Yolov8/predictions.json' # Path to save the JSON file

predict_on_folder(input_folder, model_weights_path, json_output_path)