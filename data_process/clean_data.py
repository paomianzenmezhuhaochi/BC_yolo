import os

# 删除多余的图片和标签

# 获取项目根目录（假设 data_process 和 ISICDM2025_dataset 同级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置
img_dir = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "images", "train")
label_dir = os.path.join(PROJECT_ROOT, "ISICDM2025_dataset", "labels", "train")  #TODO：手动更改以处理验证集

print("\n处理训练集...")
# 获取图片和标签的基础名集合
img_set = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith('.png'))
label_set = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith('.txt'))
# 有标签但无图片
label_no_img = label_set - img_set
# 有图片但无标签
img_no_label = img_set - label_set

print(f"有标签但无图片的数量: {len(label_no_img)}")
print("有标签但无图片的文件名和所有类别:")
for name in sorted(label_no_img):
    txt_path = os.path.join(label_dir, name + ".txt")
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if parts:
                    print(f"{name}.txt 第{idx+1}行: 类别标签 {parts[0]}")
                else:
                    print(f"{name}.txt 第{idx+1}行: 空行")
        else:
            print(f"{name}.txt: 空文件")
        # 删除标签文件
        os.remove(txt_path)
        print(f"已删除标签: {name}.txt")
    except Exception as e:
        print(f"读取或删除失败: {name}.txt, 错误: {e}")

print(f"有图片但无标签的数量: {len(img_no_label)}")
print("有图片但无标签的文件名:")
for name in sorted(img_no_label):
    img_path = os.path.join(img_dir, name + ".png")
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
            print(f"已删除图片: {name}.png")
        except Exception as e:
            print(f"删除失败: {name}.png, 错误: {e}")
    else:
        print(f"图片不存在: {name}.png")
