import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import torch


class YoloRandomCropDataset(Dataset):
    def __init__(self, img_files, label_files, crop_size=1024, transforms=None):
        self.img_files = img_files
        self.label_files = label_files
        self.crop_size = crop_size
        self.nc = 1  # 类别数
        self.names = ['cancer']  # 类别名称

        self.labels = []
        for label_path in self.label_files:
            boxes = []
            if os.path.exists(label_path):
                for line in open(label_path):
                    cls, cx, cy, bw, bh = map(float, line.split())
                    boxes.append([cls, cx, cy, bw, bh])

            if boxes:
                bboxes = np.array([b[1:] for b in boxes], dtype=np.float32)
                cls = np.array([b[0] for b in boxes], dtype=np.int64)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.int64)

            self.labels.append({"bboxes": bboxes, "cls": cls})

        if transforms is None:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.RandomApply([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ], p=0.3),
                T.RandomApply([
                    T.GaussianBlur(kernel_size=3)
                ], p=0.2),
                T.ToTensor()
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_info = self.labels[index]

        # 读取图像并转换颜色空间
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image {img_path}")
            # 返回一个黑色图像作为fallback
            img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]

        # Step 1: resize 小图
        if w0 < self.crop_size or h0 < self.crop_size:
            scale_x = self.crop_size / w0
            scale_y = self.crop_size / h0
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            h, w = self.crop_size, self.crop_size
            x1, y1 = 0, 0
        else:
            scale_x = scale_y = 1.0
            h, w = h0, w0
            # Step 2: 随机裁剪
            x1 = random.randint(0, max(0, w - self.crop_size))
            y1 = random.randint(0, max(0, h - self.crop_size))
            img = img[y1:y1 + self.crop_size, x1:x1 + self.crop_size]
            h, w = self.crop_size, self.crop_size

        # Step 3: 修复标签处理
        labels_list = []
        if len(label_info["bboxes"]) > 0:
            for i in range(len(label_info["cls"])):
                cls = int(label_info["cls"][i])
                cx, cy, bw, bh = label_info["bboxes"][i]  # 正确获取坐标

                # 转换到原图坐标系
                cx_abs, cy_abs = cx * w0, cy * h0
                bw_abs, bh_abs = bw * w0, bh * h0

                # 缩放（如果有）
                cx_abs *= scale_x
                cy_abs *= scale_y
                bw_abs *= scale_x
                bh_abs *= scale_y

                # 计算边界框坐标
                x1_bbox = cx_abs - bw_abs / 2
                y1_bbox = cy_abs - bh_abs / 2
                x2_bbox = cx_abs + bw_abs / 2
                y2_bbox = cy_abs + bh_abs / 2

                # 应用裁剪
                x1_new = max(x1_bbox - x1, 0)
                y1_new = max(y1_bbox - y1, 0)
                x2_new = min(x2_bbox - x1, w)
                y2_new = min(y2_bbox - y1, h)

                # 计算保留面积比例
                inter_area = max(0, x2_new - x1_new) * max(0, y2_new - y1_new)
                orig_area = bw_abs * bh_abs

                if orig_area > 0 and inter_area / orig_area > 0.3:
                    # 计算新的归一化坐标
                    cx_new = (x1_new + x2_new) / 2 / w
                    cy_new = (y1_new + y2_new) / 2 / h
                    bw_new = (x2_new - x1_new) / w
                    bh_new = (y2_new - y1_new) / h

                    # 确保坐标有效
                    if 0 <= cx_new <= 1 and 0 <= cy_new <= 1 and bw_new > 0.01 and bh_new > 0.01:
                        labels_list.append([cls, cx_new, cy_new, bw_new, bh_new])

        # Step 4: 图像增强
        if self.transforms:
            img_tensor = self.transforms(img)  # 转换为 [0,1] 范围的Tensor
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Step 5: 转换为YOLOv8标准格式
        if labels_list:
            # 直接创建labels格式: [cls, cx, cy, bw, bh]
            # batch_idx会在collate_fn中添加
            labels = torch.tensor(labels_list, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'img': img_tensor,
            'labels': labels,  # 只包含 [cls, cx, cy, bw, bh]
            'im_file': img_path,
            'ori_shape': (h0, w0),
            'resized_shape': (h, w)
        }

    @staticmethod
    def collate_fn(batch):
        """简化的collate函数 - 第3种方案"""
        imgs = []
        labels = []
        im_files = []

        for i, item in enumerate(batch):
            imgs.append(item['img'])
            if len(item['labels']) > 0:
                # 添加批次索引: [batch_idx, cls, cx, cy, bw, bh]
                batch_idx = torch.full((len(item['labels']), 1), i, dtype=torch.float32)
                labels_with_idx = torch.cat([batch_idx, item['labels']], dim=1)
                labels.append(labels_with_idx)
            im_files.append(item['im_file'])

        imgs = torch.stack(imgs, dim=0)
        labels = torch.cat(labels, dim=0) if labels else torch.zeros((0, 6), dtype=torch.float32)

        # 创建YOLOv8期望的完整batch字典
        result = {
            'img': imgs,
            'labels': labels,
            'im_file': im_files
        }

        # 从labels中提取其他字段
        if len(labels) > 0:
            result['batch_idx'] = labels[:, 0]  # 第一列是batch_idx
            result['cls'] = labels[:, 1]  # 第二列是cls
            result['bboxes'] = labels[:, 2:]  # 剩余的是bboxes
        else:
            result['batch_idx'] = torch.zeros(0, dtype=torch.float32)
            result['cls'] = torch.zeros(0, dtype=torch.float32)
            result['bboxes'] = torch.zeros((0, 4), dtype=torch.float32)

        return result

    @staticmethod
    def add_noise(img_tensor):
        """在 ToTensor 之后加噪声 (img 是 C,H,W 的 Tensor)"""
        noise = torch.randn_like(img_tensor) * 0.05
        return torch.clamp(img_tensor + noise, 0., 1.)