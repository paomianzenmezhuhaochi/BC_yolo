# mydataset.py
import os
import cv2
import torch
import numpy as np
from ultralytics.data.dataset import YOLODataset


class CropPadDataset(YOLODataset):
    """
    自定义数据集：仅保留 imgsz 这个参数。
    规则：
      - 若原图任一边 > imgsz：做中心裁剪到 imgsz（长边裁剪，短边保持 <= imgsz）
      - 若原图任一边 < imgsz：居中填充到 imgsz x imgsz（填充值固定 114）
      - 标签输入假设为归一化的 (cx, cy, w, h)，输出保持同格式（归一化到 imgsz）。
    返回字段：img, bboxes(归一化 cxcywh), cls, im_file, ori_shape, resized_shape
    与 YOLODataset 下游兼容。
    注意：不再调用父类 __getitem__，因此不会执行官方的 letterbox / 颜色/马赛克增强。
    """

    def __init__(self, *args, imgsz=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.imgsz = imgsz
        self._pad_value = 114  # 固定填充值

    @staticmethod
    def _xywhn_to_xyxy(boxes, w, h):
        """归一化 cx,cy,w,h -> 像素 x1,y1,x2,y2"""
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        cx = boxes[:, 0] * w
        cy = boxes[:, 1] * h
        bw = boxes[:, 2] * w
        bh = boxes[:, 3] * h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        return torch.stack([x1, y1, x2, y2], 1)

    @staticmethod
    def _xyxy_to_xywhn(boxes, size):
        """像素 x1,y1,x2,y2 -> 归一化 cx,cy,w,h (相对 size)"""
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w_box = (x2 - x1).clamp(min=1e-3)
        h_box = (y2 - y1).clamp(min=1e-3)
        cx = x1 + w_box / 2
        cy = y1 + h_box / 2
        return torch.stack([cx / size, cy / size, w_box / size, h_box / size], 1)

    def load_image_raw(self, index):
        path = self.im_files[index]
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        h0, w0 = im.shape[:2]
        return im, (h0, w0), path

    def __getitem__(self, index):
        # 直接读取原图与标签（不经过父类的 resize/augment）
        im, (h0, w0), path = self.load_image_raw(index)
        label_info = self.labels[index]
        cls = label_info["cls"]
        bboxes_n = label_info["bboxes"]  # 归一化 cxcywh
        if isinstance(cls, list):
            cls = torch.tensor(cls, dtype=torch.int64)
        cls = cls.view(-1).to(torch.int64)
        bboxes_n = bboxes_n.view(-1, 4).to(torch.float32)

        boxes_xyxy = self._xywhn_to_xyxy(bboxes_n, w0, h0)
        S = self.imgsz

        # 中心裁剪：若任一边 > S，则裁剪到 S（保持中心）
        crop_top = crop_left = 0
        new_h, new_w = h0, w0
        if h0 > S or w0 > S:
            crop_h = min(h0, S)
            crop_w = min(w0, S)
            crop_top = (h0 - crop_h) // 2 if h0 > S else 0
            crop_left = (w0 - crop_w) // 2 if w0 > S else 0
            im = im[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
            new_h, new_w = im.shape[:2]
            if boxes_xyxy.numel():
                boxes_xyxy[:, [0, 2]] -= crop_left
                boxes_xyxy[:, [1, 3]] -= crop_top

        # 过滤裁剪后无效框
        if boxes_xyxy.numel():
            keep = (boxes_xyxy[:, 2] > 0) & (boxes_xyxy[:, 3] > 0) & \
                   (boxes_xyxy[:, 0] < new_w) & (boxes_xyxy[:, 1] < new_h)
            boxes_xyxy = boxes_xyxy[keep]
            cls = cls[keep]
            if boxes_xyxy.numel():
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, new_w)
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, new_h)
                wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
                keep2 = (wh[:, 0] > 2) & (wh[:, 1] > 2)
                boxes_xyxy = boxes_xyxy[keep2]
                cls = cls[keep2]

        # 居中填充到 SxS
        pad_x = pad_y = 0
        if new_h < S or new_w < S:
            canvas = np.full((S, S, 3), self._pad_value, dtype=im.dtype)
            pad_y = (S - new_h) // 2
            pad_x = (S - new_w) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = im
            im = canvas
            if boxes_xyxy.numel():
                boxes_xyxy[:, [0, 2]] += pad_x
                boxes_xyxy[:, [1, 3]] += pad_y

        # 限幅并归一化
        if boxes_xyxy.numel():
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, S)
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, S)
        boxes_norm_final = self._xyxy_to_xywhn(boxes_xyxy, S)
        if boxes_norm_final.numel() == 0:
            boxes_norm_final = torch.zeros((0, 4), dtype=torch.float32)
            cls = torch.zeros((0,), dtype=torch.int64)

        # BGR->RGB -> Tensor
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float() / 255.0

        return {
            "img": img_tensor,
            "bboxes": boxes_norm_final,
            "cls": cls,
            "im_file": path,
            "ori_shape": (h0, w0),
            "resized_shape": (S, S)
        }
