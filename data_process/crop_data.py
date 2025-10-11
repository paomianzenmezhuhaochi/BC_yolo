img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
print(img.dtype)  # 如果是16位，应该显示为 uint16#data_process/crop_data.py
import os
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

# ======== 参数配置 ========
#自行调整路径处理训练、验证集
SOURCE_IMG_DIR = "../ISICDM2025_dataset/images/train"
SOURCE_LABEL_DIR = "../ISICDM2025_dataset/labels/train"
SAVE_IMG_DIR = "../crop_datasets/images/train"
SAVE_LABEL_DIR = "../crop_datasets/labels/train"
DEBUG_SAVE_PATH = "../debug"

CROP_SIZE = 1024
PAD_VALUE = 144  # 灰色填充
NUM_DEBUG = 8
SHOW_CROP = True  # 若为 True，随机抽取 NUM_DEBUG 张生成一张拼图(showCrop.jpg)
RANDOM_OFFSET = True  # 是否在锚点基础上添加随机偏移
OFFSET_RATIO = 0.5   # 随机偏移相对于 (CROP_SIZE/2) 的比例，0.25 表示最大偏移=窗口半边长的比例

# ========== 工具函数 ==========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_base_names(img_dir: str, label_dir: str) -> Tuple[set, set, List[str], List[str]]:
    img_paths = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.jpeg'))
    label_paths = glob.glob(os.path.join(label_dir, '*.txt'))
    img_bases = {os.path.splitext(os.path.basename(p))[0] for p in img_paths}
    label_bases = {os.path.splitext(os.path.basename(p))[0] for p in label_paths}
    return img_bases, label_bases, img_paths, label_paths

def check_mapping(img_bases: set, label_bases: set):
    only_img = sorted(list(img_bases - label_bases))
    only_label = sorted(list(label_bases - img_bases))
    if only_img:
        print(f"没有对应标签的图片({len(only_img)}):")
        for n in only_img[:50]:
            print('  ', n)
        if len(only_img) > 50:
            print('  ...')
    if only_label:
        print(f"没有对应图片的标签({len(only_label)}):")
        for n in only_label[:50]:
            print('  ', n)
        if len(only_label) > 50:
            print('  ...')
    if not only_img and not only_label:
        print("图片与标签一一匹配 ✅")

# YOLO标签解析
# line: class cx cy w h (全为归一化)

def read_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    if not os.path.exists(label_path):
        return []
    records = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                records.append((cls, cx, cy, w, h))
            except ValueError:
                continue
    return records

def write_yolo_labels(label_path: str, labels: List[Tuple[int, float, float, float, float]]):
    with open(label_path, 'w', encoding='utf-8') as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# 计算最大bbox尺寸(像素)

def max_bbox_dim(labels, img_w, img_h):
    max_dim = 0
    for _, _, _, w, h in labels:
        w_px = w * img_w
        h_px = h * img_h
        max_dim = max(max_dim, w_px, h_px)
    return max_dim

# 选择用于裁剪中心的bbox (选面积最大的)

def choose_anchor(labels):
    if not labels:
        return None
    best = None
    best_area = -1
    for rec in labels:
        _, _, _, w, h = rec
        area = w * h
        if area > best_area:
            best_area = area
            best = rec
    return best

# 主处理单张图片

def process_one(img_path: str, label_path: str, save_img_filename: str):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return False
    h, w = img.shape[:2]
    labels = read_yolo_labels(label_path)

    if not labels:
        # 没有标签也可选择跳过或直接复制，需求未明确，这里仍然进行中心裁剪(用图像中心)
        anchor_cx_px = w / 2
        anchor_cy_px = h / 2
    else:
        # 检测是否需要缩放
        max_dim = max_bbox_dim(labels, w, h)
        scale = 1.0
        if max_dim > CROP_SIZE:
            scale = CROP_SIZE / max_dim
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            # 归一化标签不用改
        anchor = choose_anchor(labels)
        if anchor is not None:
            _, cx_n, cy_n, _, _ = anchor
            anchor_cx_px = cx_n * w
            anchor_cy_px = cy_n * h
        else:
            anchor_cx_px = w / 2
            anchor_cy_px = h / 2

    # ---- 添加随机偏移 ----
    if RANDOM_OFFSET:
        # 最大理论偏移不超过窗口半边长，确保锚点仍留在裁剪窗口内（避免完全丢失主要目标）
        max_shift_x = (CROP_SIZE / 2 - 1) * OFFSET_RATIO
        max_shift_y = (CROP_SIZE / 2 - 1) * OFFSET_RATIO
        dx = random.uniform(-max_shift_x, max_shift_x)
        dy = random.uniform(-max_shift_y, max_shift_y)
        anchor_cx_px += dx
        anchor_cy_px += dy
        # （可选）如果想限制偏移后中心仍尽量落在原图边界内，可再做 clamp，这里允许越界以触发 padding。
    # ----------------------

    # 计算裁剪窗口左上角
    crop_left = int(round(anchor_cx_px - CROP_SIZE / 2))
    crop_top = int(round(anchor_cy_px - CROP_SIZE / 2))

    # 源与目标交集
    src_x1 = max(0, crop_left)
    src_y1 = max(0, crop_top)
    src_x2 = min(w, crop_left + CROP_SIZE)
    src_y2 = min(h, crop_top + CROP_SIZE)

    dst_x1 = src_x1 - crop_left
    dst_y1 = src_y1 - crop_top
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # 创建填充图
    cropped = np.full((CROP_SIZE, CROP_SIZE, 3), PAD_VALUE, dtype=img.dtype)
    if src_x2 > src_x1 and src_y2 > src_y1:
        cropped[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    # 更新标签
    new_labels = []
    for rec in labels:
        cls, cx_n, cy_n, w_n, h_n = rec
        # 原像素坐标
        bx_cx = cx_n * w
        bx_cy = cy_n * h
        bw = w_n * w
        bh = h_n * h
        bx1 = bx_cx - bw / 2
        bx2 = bx_cx + bw / 2
        by1 = bx_cy - bh / 2
        by2 = bx_cy + bh / 2
        # 与裁剪窗口的交集 (窗口坐标原图系)
        win_x1 = crop_left
        win_y1 = crop_top
        win_x2 = crop_left + CROP_SIZE
        win_y2 = crop_top + CROP_SIZE
        ix1 = max(bx1, win_x1)
        iy1 = max(by1, win_y1)
        ix2 = min(bx2, win_x2)
        iy2 = min(by2, win_y2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue  # 与裁剪无交集
        # 裁剪内坐标 (相对0,0=crop左上)
        new_cx_px = (ix1 + ix2) / 2 - win_x1
        new_cy_px = (iy1 + iy2) / 2 - win_y1
        new_w_px = ix2 - ix1
        new_h_px = iy2 - iy1
        # 归一化 (除以CROP_SIZE)
        new_cx = new_cx_px / CROP_SIZE
        new_cy = new_cy_px / CROP_SIZE
        new_w = new_w_px / CROP_SIZE
        new_h = new_h_px / CROP_SIZE
        # 过滤异常
        if new_w <= 0 or new_h <= 0:
            continue
        # 裁剪导致的数值轻微越界，clip
        new_cx = min(max(new_cx, 0.0), 1.0)
        new_cy = min(max(new_cy, 0.0), 1.0)
        new_w = min(max(new_w, 0.0), 1.0)
        new_h = min(max(new_h, 0.0), 1.0)
        new_labels.append((cls, new_cx, new_cy, new_w, new_h))

    # 若没有标签，且原本有标签 -> 可选择放弃；这里保留为空文件
    save_img_path = os.path.join(SAVE_IMG_DIR, save_img_filename)  # 保持原文件名与扩展
    base = os.path.splitext(save_img_filename)[0]
    save_label_path = os.path.join(SAVE_LABEL_DIR, base + '.txt')
    cv2.imwrite(save_img_path, cropped)
    write_yolo_labels(save_label_path, new_labels)
    return True

# 绘制调试可视化 (单张图上拼接多个样本)

def draw_montage(sample_files: List[str]):
    if not sample_files:
        return
    cols = 4 if len(sample_files) > 4 else len(sample_files)
    rows = int(np.ceil(len(sample_files) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, fname in enumerate(sample_files):
        img_path = os.path.join(SAVE_IMG_DIR, fname)
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(SAVE_LABEL_DIR, base + '.txt')
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        labels = read_yolo_labels(label_path)
        # 画框
        for cls, cx, cy, bw, bh in labels:
            cx_px = cx * w
            cy_px = cy * h
            bw_px = bw * w
            bh_px = bh * h
            x1 = int(cx_px - bw_px / 2)
            y1 = int(cy_px - bh_px / 2)
            x2 = int(cx_px + bw_px / 2)
            y2 = int(cy_px + bh_px / 2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            img = cv2.putText(img, str(cls), (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.title(base, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(DEBUG_SAVE_PATH, 'showCrop.jpg')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"调试拼图已保存: {out_path}")


def main():
    ensure_dir(SAVE_IMG_DIR)
    ensure_dir(SAVE_LABEL_DIR)
    ensure_dir(DEBUG_SAVE_PATH)

    img_bases, label_bases, img_paths, _ = list_base_names(SOURCE_IMG_DIR, SOURCE_LABEL_DIR)
    check_mapping(img_bases, label_bases)

    common_bases = sorted(list(img_bases & label_bases))
    total = len(common_bases)
    print(f"开始处理 {total} 对图像/标签 ...")

    processed_imgs = []  # 保存已处理的原始文件名(含扩展)
    for i, base in enumerate(common_bases, 1):
        # 找到真实存在的图像文件路径和扩展
        img_path = None
        img_filename = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            trial = os.path.join(SOURCE_IMG_DIR, base + ext)
            if os.path.exists(trial):
                img_path = trial
                img_filename = base + ext
                break
        if img_path is None:
            print(f"跳过: 找不到图像 {base}.*")
            continue
        label_path = os.path.join(SOURCE_LABEL_DIR, base + '.txt')
        ok = process_one(img_path, label_path, img_filename)
        if ok:
            processed_imgs.append(img_filename)
        if i % 50 == 0 or i == total:
            print(f"  进度: {i}/{total} ({i/total*100:.1f}%)")

    print(f"处理完成。成功处理 {len(processed_imgs)} 张。")

    if SHOW_CROP and processed_imgs:
        sample = random.sample(processed_imgs, min(NUM_DEBUG, len(processed_imgs)))
        print(f"生成调试拼图: {sample}")
        draw_montage(sample)
        print(f"调试图保存在: {DEBUG_SAVE_PATH}")


if __name__ == '__main__':
    main()
