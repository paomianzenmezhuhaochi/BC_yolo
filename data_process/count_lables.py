import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def analyze_labels(label_root, num_classes, class_names=None, save_fig=True):
    stats = defaultdict(int)

    # 统计 train/val
    for split in ["train", "val"]:
        split_dir = os.path.join(label_root, split)
        if not os.path.exists(split_dir):
            print(f"{split_dir} 不存在，跳过")
            continue

        count = Counter()
        n_files = 0

        for file in os.listdir(split_dir):
            if not file.endswith(".txt"):
                continue
            n_files += 1
            with open(os.path.join(split_dir, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    cls = int(line.strip().split()[0])
                    count[cls] += 1

        print(f"\n数据集划分: {split} (共 {n_files} 张图片)")
        for i in range(num_classes):
            name = class_names[i] if class_names else str(i)
            print(f"  类别 {i} ({name}): {count[i]} 个标注")
            stats[i] += count[i]

    print("\n=== 全部数据集类别分布 ===")
    for i in range(num_classes):
        name = class_names[i] if class_names else str(i)
        print(f"类别 {i} ({name}): {stats[i]} 个标注")

    # 画柱状图
    labels = [class_names[i] if class_names else str(i) for i in range(num_classes)]
    values = [stats[i] for i in range(num_classes)]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="skyblue", edgecolor="black")
    plt.xlabel("category")
    plt.ylabel("number")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_fig:
        debug_dir = os.path.join(PROJECT_ROOT, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, "class_distribution1.png") #TODO:注意修改不然覆盖了
        plt.savefig(save_path)
        print(f"\n柱状图已保存到 {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_root = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "labels")
    num_classes = 7
    class_names = [
        "BI-RADS-0", "BI-RADS-1", "BI-RADS-2",
        "BI-RADS-3", "BI-RADS-4", "BI-RADS-5", "BI-RADS-6"
    ]

    analyze_labels(labels_root, num_classes, class_names, save_fig=True)
