import os
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# 工具函数：YOLO格式解析与几何计算
# ------------------------------

def read_yolo_txt(path: str) -> List[Tuple[int, float, float, float, float, float]]:
    """
    读取YOLO标签/预测txt，按行解析为 (cls, conf, cx, cy, w, h)
    - GT标签一般为: cls cx cy w h (5列)，此时 conf 置为 1.0
    - 预测标签一般为: cls conf cx cy w h (6列)
    所有坐标均视为归一化到[0,1]的比例。
    """
    items = []
    if not os.path.isfile(path):
        return items
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 5:
                # GT: cls cx cy w h
                c = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                conf = 1.0
            elif len(parts) == 6:
                # Pred: cls conf cx cy w h
                c = int(float(parts[0]))
                conf = float(parts[1])
                cx, cy, w, h = map(float, parts[2:6])
            else:
                # 非法行，跳过
                continue
            items.append((c, conf, cx, cy, w, h))
    return items


def xywhn_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    将 (cx, cy, w, h) 归一化坐标转换为 (x1, y1, x2, y2) 归一化坐标。
    输入: boxes形状 [N,4]
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """
    计算两组框的IoU矩阵，a形状[N,4], b形状[M,4]，返回[N,M]
    坐标均为归一化空间中的xyxy。
    """
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]))

    # 扩展维度以便矢量化计算
    a = a_xyxy[:, None, :]  # [N,1,4]
    b = b_xyxy[None, :, :]  # [1,M,4]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, 1.0)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, 1.0)
    inter = inter_w * inter_h

    area_a = np.clip((a[..., 2] - a[..., 0]), 0.0, 1.0) * np.clip((a[..., 3] - a[..., 1]), 0.0, 1.0)
    area_b = np.clip((b[..., 2] - b[..., 0]), 0.0, 1.0) * np.clip((b[..., 3] - b[..., 1]), 0.0, 1.0)

    union = area_a + area_b - inter
    iou = inter / np.clip(union, 1e-9, None)
    return iou


# ------------------------------
# 评估核心：mAP、Precision、Recall
# ------------------------------

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    以11点插值(或更致密点)方式近似积分PR曲线，返回AP。
    这里使用与常见实现一致的插值：先对precision做向后包络，再对recall积分。
    """
    # 在首尾加锚点
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 使precision成为非增函数（右侧包络）
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 计算recall发生变化的位置并累加面积
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate_yolo(
    gt_dir: str,
    pred_dir: str,
    nc: int,
    names: List[str] = None,
    conf_thres_pr: float = 0.25,
    iou_thresholds: np.ndarray = None,
    save_confusion_path: str = None,
) -> Dict:
    """
    直接使用YOLO txt格式(gt: cls cx cy w h; pred: cls conf cx cy w h)进行评估。

    返回：
      {
        'metrics': {
            'mAP50': float,
            'mAP50-95': float,
            'precision': float,  # 在conf_thres_pr下
            'recall': float,     # 在conf_thres_pr下
        },
        'per_class': {
            class_id: {
                'ap50': float,
                'ap': float,  # 0.5:0.95 平均
                'precision': float,
                'recall': float,
            }
        },
        'confusion_matrix': np.ndarray  # [nc, nc+1] 最后一列为未检出(背景)
      }
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)  # 0.5:0.95

    # 1) 建立文件stem集合
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.txt')))
    gt_stems = {os.path.splitext(os.path.basename(p))[0] for p in gt_files}
    pred_stems = {os.path.splitext(os.path.basename(p))[0] for p in pred_files}
    all_stems = sorted(gt_stems | pred_stems)

    # 2) 容器：每类的预测置信度、以及在各IoU阈值下的TP/FP序列（与置信度同序）
    t_indices = list(range(len(iou_thresholds)))
    cls_confs: List[List[float]] = [[] for _ in range(nc)]
    cls_tp: List[List[List[int]]] = [[[] for _ in t_indices] for _ in range(nc)]
    cls_fp: List[List[List[int]]] = [[[] for _ in t_indices] for _ in range(nc)]
    npos_per_class = np.zeros(nc, dtype=int)

    # 混淆矩阵 (nc x (nc+1))，最后一列为未检出
    cm = np.zeros((nc, nc + 1), dtype=int)

    for stem in all_stems:
        gt_path = os.path.join(gt_dir, f'{stem}.txt')
        pred_path = os.path.join(pred_dir, f'{stem}.txt')
        gts = read_yolo_txt(gt_path)
        preds = read_yolo_txt(pred_path)

        # 分组统计GT
        g_by_cls: Dict[int, List[Tuple[int, float, float, float, float, float]]] = defaultdict(list)
        for it in gts:
            g_by_cls[it[0]].append(it)
        for c, lst in g_by_cls.items():
            if 0 <= c < nc:
                npos_per_class[c] += len(lst)

        # 逐类计算TP/FP（PR曲线与AP）
        p_by_cls: Dict[int, List[Tuple[int, float, float, float, float, float]]] = defaultdict(list)
        for it in preds:
            p_by_cls[it[0]].append(it)

        for c in set(list(g_by_cls.keys()) + list(p_by_cls.keys())):
            if c < 0 or c >= nc:
                continue
            g_arr = np.array([[gx, gy, gw, gh] for (_, _, gx, gy, gw, gh) in g_by_cls.get(c, [])], dtype=float)
            p_arr = np.array([[px, py, pw, ph] for (_, _, px, py, pw, ph) in p_by_cls.get(c, [])], dtype=float)
            p_conf = np.array([conf for (_, conf, *_rest) in p_by_cls.get(c, [])], dtype=float)

            # 按conf降序
            order = np.argsort(-p_conf) if p_conf.size else np.array([], dtype=int)
            p_arr = p_arr[order] if p_arr.size else p_arr.reshape(0, 4)
            p_conf = p_conf[order] if p_conf.size else p_conf

            # 将xywhn转xyxyn
            g_xyxy = xywhn_to_xyxy(g_arr) if g_arr.size else g_arr.reshape(0, 4)
            p_xyxy = xywhn_to_xyxy(p_arr) if p_arr.size else p_arr.reshape(0, 4)

            # IoU矩阵
            ious = box_iou_matrix(p_xyxy, g_xyxy)  # [P, G]
            G = g_xyxy.shape[0]
            P = p_xyxy.shape[0]
            gt_matched = np.zeros((len(iou_thresholds), G), dtype=bool)

            # 记录置信度
            if p_conf.size:
                cls_confs[c].extend(p_conf.tolist())

            # 为每个预测在各阈值记录TP/FP
            for pi in range(P):
                iou_row = ious[pi] if G > 0 else np.zeros((0,), dtype=float)
                best_gi = int(np.argmax(iou_row)) if G > 0 else -1
                best_iou = float(iou_row[best_gi]) if G > 0 else 0.0
                for ti, thr in enumerate(iou_thresholds):
                    if G > 0 and best_iou >= thr and not gt_matched[ti, best_gi]:
                        cls_tp[c][ti].append(1)
                        cls_fp[c][ti].append(0)
                        gt_matched[ti, best_gi] = True
                    else:
                        cls_tp[c][ti].append(0)
                        cls_fp[c][ti].append(1)

        # 全局混淆矩阵（IoU@0.5）：跨类别匹配，填充对角(正确)、非对角(误分类)、以及未检出列
        if len(gts) > 0:
            g_cls = np.array([it[0] for it in gts], dtype=int)
            g_xyxy_all = xywhn_to_xyxy(np.array([[gx, gy, gw, gh] for (_, _, gx, gy, gw, gh) in gts], dtype=float))
        else:
            g_cls = np.zeros((0,), dtype=int)
            g_xyxy_all = np.zeros((0, 4), dtype=float)

        if len(preds) > 0:
            p_cls = np.array([it[0] for it in preds], dtype=int)
            p_conf_all = np.array([it[1] for it in preds], dtype=float)
            p_xyxy_all = xywhn_to_xyxy(np.array([[px, py, pw, ph] for (_, _, px, py, pw, ph) in preds], dtype=float))
            # 按conf排序（高到低）
            order_all = np.argsort(-p_conf_all)
            p_cls = p_cls[order_all]
            p_xyxy_all = p_xyxy_all[order_all]
        else:
            p_cls = np.zeros((0,), dtype=int)
            p_xyxy_all = np.zeros((0, 4), dtype=float)

        if p_xyxy_all.size and g_xyxy_all.size:
            iou_all = box_iou_matrix(p_xyxy_all, g_xyxy_all)
            used_g = np.zeros((g_xyxy_all.shape[0],), dtype=bool)
            for pi in range(p_xyxy_all.shape[0]):
                gi = int(np.argmax(iou_all[pi]))
                iou = float(iou_all[pi, gi])
                if iou >= 0.5 and not used_g[gi]:
                    gt_c = int(g_cls[gi])
                    pr_c = int(p_cls[pi])
                    if 0 <= gt_c < nc and 0 <= pr_c < nc:
                        cm[gt_c, pr_c] += 1
                    used_g[gi] = True
            # 未匹配的GT -> 未检出列
            for gi, used in enumerate(used_g):
                if not used:
                    gt_c = int(g_cls[gi])
                    if 0 <= gt_c < nc:
                        cm[gt_c, -1] += 1
        elif g_xyxy_all.size and not p_xyxy_all.size:
            # 全未检出
            for gt_c in g_cls:
                if 0 <= gt_c < nc:
                    cm[int(gt_c), -1] += 1
        # 若只有预测没有GT，则这些为背景FP，不在当前矩阵中计入（如需可扩展出“背景行”）

    # 3) 计算每类在各阈值的AP，再平均（忽略无GT类别）
    ap50_per_class = np.zeros(nc, dtype=float)
    ap5095_per_class = np.zeros(nc, dtype=float)
    prec_conf = np.zeros(nc, dtype=float)
    rec_conf = np.zeros(nc, dtype=float)

    valid_cls = np.where(npos_per_class > 0)[0]

    for c in range(nc):
        if len(cls_confs[c]) == 0 or npos_per_class[c] == 0:
            # 没有该类预测或没有该类GT
            ap50_per_class[c] = 0.0
            ap5095_per_class[c] = 0.0
            prec_conf[c] = 0.0
            rec_conf[c] = 0.0
            continue

        confs = np.array(cls_confs[c], dtype=float)
        order = np.argsort(-confs)
        npos = int(npos_per_class[c])

        ap_t = []
        for ti, _thr in enumerate(iou_thresholds):
            tp = np.array(cls_tp[c][ti], dtype=float)[order]
            fp = np.array(cls_fp[c][ti], dtype=float)[order]

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(npos, 1)
            precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
            ap = compute_ap(recall, precision)
            ap_t.append(ap)

            if abs(_thr - 0.5) < 1e-6:
                # 计算在指定conf阈值下的P/R
                mask = confs[order] >= conf_thres_pr
                if mask.any():
                    tp_ = tp[mask]
                    fp_ = fp[mask]
                    tp_c = tp_.sum()
                    fp_c = fp_.sum()
                    prec_conf[c] = float(tp_c / max(tp_c + fp_c, 1e-9))
                    rec_conf[c] = float(tp_c / max(npos, 1))
                else:
                    prec_conf[c] = 0.0
                    rec_conf[c] = 0.0

        ap50_per_class[c] = ap_t[0]  # 第一个阈值是0.5
        ap5095_per_class[c] = float(np.mean(ap_t))

    # 汇总（仅在有GT的类别上平均）
    if len(valid_cls) > 0:
        mAP50 = float(np.mean(ap50_per_class[valid_cls]))
        mAP5095 = float(np.mean(ap5095_per_class[valid_cls]))
    else:
        mAP50 = 0.0
        mAP5095 = 0.0

    # 在conf_thres_pr下的总体P/R（IoU=0.5）
    total_tp = 0.0
    total_fp = 0.0
    total_pos = int(npos_per_class.sum())
    for c in range(nc):
        if len(cls_confs[c]) == 0:
            continue
        confs = np.array(cls_confs[c], dtype=float)
        order = np.argsort(-confs)
        tp = np.array(cls_tp[c][0], dtype=float)[order]
        fp = np.array(cls_fp[c][0], dtype=float)[order]
        mask = confs[order] >= conf_thres_pr
        total_tp += float(tp[mask].sum())
        total_fp += float(fp[mask].sum())
    precision_all = float(total_tp / max(total_tp + total_fp, 1e-9))
    recall_all = float(total_tp / max(total_pos, 1)) if total_pos > 0 else 0.0

    results = {
        'metrics': {
            'mAP50': mAP50,
            'mAP50-95': mAP5095,
            'precision': precision_all,
            'recall': recall_all,
        },
        'per_class': {
            int(c): {
                'ap50': float(ap50_per_class[c]),
                'ap': float(ap5095_per_class[c]),
                'precision': float(prec_conf[c]),
                'recall': float(rec_conf[c]),
            } for c in range(nc)
        },
        'confusion_matrix': cm,
    }

    # 可视化混淆矩阵
    if save_confusion_path is not None:
        os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
        class_names = names if names and len(names) == nc else [str(i) for i in range(nc)]
        fig_w = max(6, 0.6 * (nc + 1))
        fig_h = max(5, 0.6 * (nc + 1))
        plt.figure(figsize=(fig_w, fig_h))
        im = plt.imshow(cm, cmap='Blues')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(ticks=np.arange(nc + 1), labels=class_names + ['miss'], rotation=45, ha='right')
        plt.yticks(ticks=np.arange(nc), labels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix (IoU@0.5)')
        # 在格子中标数字
        for i in range(nc):
            for j in range(nc + 1):
                val = int(cm[i, j])
                if val > 0:
                    plt.text(j, i, str(val), ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(save_confusion_path, dpi=200)
        plt.close()

    return results


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO-format predictions against YOLO-format GT (no COCO conversion).')
    parser.add_argument('--gt-dir', type=str, required=True, help='路径：GT标签目录（YOLO txt: cls cx cy w h）')
    parser.add_argument('--pred-dir', type=str, required=True, help='路径：预测标签目录（YOLO txt: cls conf cx cy w h）')
    parser.add_argument('--nc', type=int, required=True, help='类别数')
    parser.add_argument('--names', type=str, default='', help='以逗号分隔的类别名，可选')
    parser.add_argument('--conf', type=float, default=0.25, help='用于报告总体P/R的置信度阈值（IoU=0.5）')
    parser.add_argument('--save-cm', type=str, default='runs/metrics/confusion_matrix.png', help='混淆矩阵保存路径')
    args = parser.parse_args()

    names = [s.strip() for s in args.names.split(',')] if args.names else None

    results = evaluate_yolo(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        nc=args.nc,
        names=names,
        conf_thres_pr=args.conf,
        iou_thresholds=np.arange(0.5, 0.96, 0.05),
        save_confusion_path=args.save_cm,
    )

    print('\n==== Summary ====')
    print(f"mAP@0.5     : {results['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 : {results['metrics']['mAP50-95']:.4f}")
    print(f"Precision@{args.conf:.2f} : {results['metrics']['precision']:.4f}")
    print(f"Recall@{args.conf:.2f}    : {results['metrics']['recall']:.4f}")

    if names and len(names) == args.nc:
        print('\nPer-class (name : AP@0.5 / AP@0.5:0.95 / P / R)')
        for i, n in enumerate(names):
            d = results['per_class'][i]
            print(f"{n:>12s} : {d['ap50']:.4f} / {d['ap']:.4f} / {d['precision']:.4f} / {d['recall']:.4f}")
    else:
        print('\nPer-class (id : AP@0.5 / AP@0.5:0.95 / P / R)')
        for i in range(args.nc):
            d = results['per_class'][i]
            print(f"{i:>3d} : {d['ap50']:.4f} / {d['ap']:.4f} / {d['precision']:.4f} / {d['recall']:.4f}")


if __name__ == '__main__':
    main()
