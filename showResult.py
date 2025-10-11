import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
精简版批量可视化脚本 (只支持一种 JSON 格式)
------------------------------------------------
支持的唯一 JSON 标签格式: 根为列表，每个元素形如
[
  {
    "image_id": "ISICDM2025_test_100.png",  # 图像文件名 (需在 --test_dir 下存在 PNG)
    "category_id": 0,                        # 类别 (任意可转为字符串的值)
    "bbox": [x, y, width, height],           # 左上角 + 宽高 (全部像素坐标)
    "score": 0.97                            # (可选) 置信度 0~1，不显示时可忽略
  },
  ...
]

功能:
1. 读取测试图像目录 (--test_dir) 下的 PNG 图片。
2. 读取上述格式的 JSON (--json)。
3. 将同一 image_id 的多个目标聚合，绘制矩形框与类别 ID。
4. 每 batch_size (默认 5) 张图像拼成一张大图: batch1.jpg, batch2.jpg ...
5. 输出到 --save_dir，自动创建目录。

仅保留最小必要代码，去除所有其它 JSON 结构与归一化/中心格式推断。
"""

# ---------------- 参数解析 ---------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="批量检测结果可视化 (简化版, 仅支持列表 JSON)")
    parser.add_argument('--test_dir', type=str, default="ISICDM2025_images_for_test", help='包含待显示 PNG 图像的文件夹')
    parser.add_argument('--json', type=str, default="predictions3_2.json", help='标签 JSON 文件路径 (根为列表)')
    parser.add_argument('--save_dir', type=str, default="debug/try0", help='输出拼图目录')
    parser.add_argument('--batch_size', type=int, default=5, help='每张拼图包含的图像数量')
    parser.add_argument('--fig_size', type=float, default=4.0, help='单图子图基准大小（英寸）')
    parser.add_argument('--max_cols', type=int, default=5, help='拼图最大列数')
    parser.add_argument('--dpi', type=int, default=120, help='保存图片 DPI')
    parser.add_argument('--font_scale', type=float, default=2, help='标签字体缩放')
    parser.add_argument('--thickness', type=int, default=5, help='框线宽')
    parser.add_argument('--show_score', action='store_true', help='标签上同时显示 score')
    return parser.parse_args()

# ---------------- 读取与解析 ---------------- #

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('JSON 顶层必须是列表 (list)。')
    return data


def build_annotation_map(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for obj in records:
        if not isinstance(obj, dict):
            continue
        if 'image_id' not in obj or 'bbox' not in obj or 'category_id' not in obj:
            continue
        bbox = obj['bbox']
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x, y, w, h = bbox
        # 过滤非法尺寸
        if w <= 0 or h <= 0:
            continue
        ann = {
            'class': str(obj['category_id']),
            'bbox': [x, y, w, h],  # 绝对像素 (x,y,w,h)
            'score': float(obj.get('score')) if 'score' in obj else None
        }
        mapping.setdefault(obj['image_id'], []).append(ann)
    return mapping

# ---------------- 绘制 ---------------- #

COLOR_CACHE: Dict[str, Tuple[int, int, int]] = {}
# 固定颜色映射: 0红 1蓝 2绿 3粉 4黄 (其余类别用灰色或自动生成)
FIXED_COLORS: Dict[str, Tuple[int,int,int]] = {
    '0': (255, 0, 0),        # red
    '1': (0, 0, 255),        # blue
    '2': (0, 255, 0),        # green
    '3': (255, 105, 180),    # pink (hot pink)
    '4': (255, 255, 0),      # yellow
    '5': (255, 255, 255),    # white
    '6': (255, 165, 0),      # orange
}

def get_color(cls: str) -> Tuple[int, int, int]:
    # 若在固定映射中直接返回
    if cls in FIXED_COLORS:
        return FIXED_COLORS[cls]
    # 否则使用缓存或生成一个中性灰/随机色（这里采用浅灰）
    if cls not in COLOR_CACHE:
        COLOR_CACHE[cls] = (180, 180, 180)
    return COLOR_CACHE[cls]


def draw_bboxes(img: np.ndarray, anns: List[Dict[str, Any]], font_scale: float, thickness: int, show_score: bool) -> np.ndarray:
    if not anns:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for a in anns:
        cls = a['class']
        x, y, bw, bh = a['bbox']
        # 转角点
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + bw))
        y2 = int(round(y + bh))
        # clip
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = get_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = cls if not show_score or a.get('score') is None else f"{cls}:{a['score']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness // 2))
        ty1 = max(0, y1 - th - base)
        cv2.rectangle(out, (x1, ty1), (x1 + tw, ty1 + th + base), color, -1)
        cv2.putText(out, label, (x1, ty1 + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, thickness // 2), cv2.LINE_AA)
    return out

# ---------------- 主流程 ---------------- #

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        records = load_json_list(args.json)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return

    ann_map = build_annotation_map(records)
    if not ann_map:
        print("未解析到任何标注，退出。")
        return

    # 收集目录下 png
    images = [f for f in os.listdir(args.test_dir) if f.lower().endswith('.png')]
    images.sort()
    if not images:
        print("测试目录中没有 PNG 文件。")
        return

    batch_size = max(1, args.batch_size)
    max_cols = max(1, args.max_cols)
    total_batches = math.ceil(len(images) / batch_size)
    print(f"共 {len(images)} 张图片 -> {total_batches} 个批次 (batch_size={batch_size})")

    classes_seen = set()

    for b in range(total_batches):
        batch_imgs = images[b * batch_size:(b + 1) * batch_size]
        cols = min(len(batch_imgs), max_cols)
        rows = math.ceil(len(batch_imgs) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * args.fig_size, rows * args.fig_size))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        for idx, fname in enumerate(batch_imgs):
            r = idx // cols; c = idx % cols
            ax = axes[r, c]
            path = os.path.join(args.test_dir, fname)
            img = cv2.imread(path)
            if img is None:
                ax.set_title(f"{fname}\n读取失败", fontsize=9)
                ax.axis('off')
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            anns = ann_map.get(fname, [])
            if anns:
                for a in anns:
                    classes_seen.add(a['class'])
            drawn = draw_bboxes(img_rgb, anns, font_scale=args.font_scale, thickness=args.thickness, show_score=args.show_score)
            ax.imshow(drawn)
            ax.set_title(fname, fontsize=9)
            ax.axis('off')

        # 关闭空格子
        for ridx in range(rows):
            for cidx in range(cols):
                linear = ridx * cols + cidx
                if linear >= len(batch_imgs):
                    axes[ridx, cidx].axis('off')

        plt.tight_layout()
        out_file = os.path.join(args.save_dir, f"batch{b + 1}.jpg")
        plt.savefig(out_file, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"保存: {out_file}")

    if classes_seen:
        print(f"共检测到类别: {sorted(classes_seen)}")
        print("固定颜色映射 (RGB / HEX)：")
        for cls_name in sorted(classes_seen):
            r,g,b = get_color(cls_name)
            print(f"  {cls_name}: ({r},{g},{b})  #{r:02X}{g:02X}{b:02X}")
    else:
        print("未发现任何类别标注。")


if __name__ == '__main__':
    main()
