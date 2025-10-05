# slide_infer.py
import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


class SlideInfer:
    def __init__(self, model, patch_size=1024, stride=800, conf_thres=0.3, iou_thres=0.6):
        """
        滑窗推理类
        Args:
            model: 训练好的 YOLO 模型 (YOLO 实例)
            patch_size: 每个小切片的大小
            stride: 滑窗步长
            conf_thres: 置信度阈值
            iou_thres: WBF 的 IoU 阈值（用于融合相近框）
        """
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def wbf(self, det: np.ndarray, img_shape, conf_thres: float | None = None, iou_thr: float | None = None) -> np.ndarray:
        """
        用 WBF 替代 NMS 的融合方法（类方法）
        参数:
            det: np.array, [num_boxes, 6] -> [x1, y1, x2, y2, conf, cls]
            img_shape: (h, w) 原图尺寸
            conf_thres: 置信度阈值（默认用 self.conf_thres）
            iou_thr: WBF IoU 阈值（默认用 self.iou_thres）
        返回:
            (np.ndarray) [num_final_boxes, 6] -> [x1, y1, x2, y2, conf, cls]
        """
        if det is None or len(det) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        conf = self.conf_thres if conf_thres is None else conf_thres
        iou = self.iou_thres if iou_thr is None else iou_thr

        # 1) 置信度筛选
        det = det[det[:, 4] > conf]
        if len(det) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # 2) 归一化到 [0,1]
        h, w = img_shape[:2]
        boxes = det[:, :4].astype(np.float32).copy()
        boxes[:, [0, 2]] /= max(w, 1e-6)
        boxes[:, [1, 3]] /= max(h, 1e-6)

        scores = det[:, 4].astype(float).tolist()
        labels = det[:, 5].astype(int).tolist()

        # 3) 按类别分组融合，避免不同类别相互融合
        final_boxes, final_scores, final_labels = [], [], []
        for cls in sorted(set(labels)):
            idxs = [i for i, l in enumerate(labels) if l == cls]
            if not idxs:
                continue
            cls_boxes = [boxes[i] for i in idxs]
            cls_scores = [scores[i] for i in idxs]
            cls_labels = [labels[i] for i in idxs]

            b_fused, s_fused, l_fused = weighted_boxes_fusion(
                [cls_boxes], [cls_scores], [cls_labels],
                weights=None,
                iou_thr=iou,
                skip_box_thr=conf,
                conf_type='avg'
            )
            final_boxes.extend(b_fused)
            final_scores.extend(s_fused)
            final_labels.extend(l_fused)

        if len(final_boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # 4) 还原到像素坐标
        final_boxes = np.array(final_boxes, dtype=np.float32)
        final_boxes[:, [0, 2]] *= w
        final_boxes[:, [1, 3]] *= h
        final_scores = np.array(final_scores, dtype=np.float32)
        final_labels = np.array(final_labels, dtype=np.float32)  # 与 det 保持同为 float 拼接

        fused = np.concatenate([final_boxes, final_scores[:, None], final_labels[:, None]], axis=1)
        return fused.astype(np.float32)

    def infer_image(self, image_path):
        """
        对单张图像进行滑窗推理
        """
        img = cv2.imread(image_path)
        H, W, _ = img.shape

        all_boxes, all_scores, all_classes = [], [], []

        # 遍历滑窗
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                # 保证 patch 不越界
                x1, y1 = min(x, W - self.patch_size), min(y, H - self.patch_size)
                patch = img[y1:y1 + self.patch_size, x1:x1 + self.patch_size]

                # 模型推理
                results = self.model.predict(
                    patch,
                    imgsz=self.patch_size,
                    conf=0.1,
                    iou=0.2,
                    agnostic_nms=False,
                    verbose=False
                )

                # 逐个结果处理
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        # [x1, y1, x2, y2]（相对小 patch 的坐标）
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()

                        # 坐标平移到大图
                        boxes[:, [0, 2]] += x1
                        boxes[:, [1, 3]] += y1

                        # 保存到全局列表
                        all_boxes.append(boxes)
                        all_scores.append(scores)
                        all_classes.append(classes)

        # 如果没有检测结果
        if not all_boxes:
            return None

        # 将不同切片的预测拼接成一个整体
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)

        # 拼接成 [x1, y1, x2, y2, score, cls]
        det = np.concatenate(
            [all_boxes, all_scores[:, None], all_classes[:, None]], axis=1
        ).astype(np.float32)

        # 使用 WBF 融合替代 NMS（改为类方法 self.wbf）
        det = self.wbf(det, (H, W))

        return det  # [x1, y1, x2, y2, conf, cls]