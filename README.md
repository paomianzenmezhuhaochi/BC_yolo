# 使用yolo进行病灶检测
## 1、环境配置
### 1.1、本地环境配置
```bash
pip install -r requirements.txt
```
### 1.2、colab 运行更推荐 
 见 [Colab Notebook](./train_colab.ipynb)
## 2、项目结构介绍
```text
BC_yolo/
├── configs/                  # 模型配置文件
├── data_process/             # 数据处理相关脚本
├── ground_truth.json         # 测试集集真值文件
├── predict_patch.py          # Patch-based 推理脚本
├── predict_test.py           # basic推理脚本
├── predict_test0.json        # 推理结果
├── predictions.json          # 推理结果
├── requirements.txt          # 项目依赖
├── runs/                     # 训练输出目录
├── scripts/                  # 训练脚本目录
├── showResult.py             # 可视化结果脚本
├── train_colab.ipynb         # Colab 训练流程 notebook
├── ultralytics/              # Ultralytics YOLO 源码或定制目录
├── README.md                 # 项目说明文件
```

### 主要文件和目录说明

- `predict_patch.py`：基于 Patch 的目标检测/分割推理脚本。
- `predict_test.py`：直接调用yolo模型推理，预测测试集结果。
- `showResult.py`：推理结果的可视化与展示。
- `train_colab.ipynb`：Colab 环境下的训练流程 notebook。
- `configs/`：存放模型、训练等配置文件。
- `data_process/`：数据预处理、增强等相关代码。
- `ultralytics/`：Ultralytics YOLO 源码或二次开发部分。
- `runs/`：训练/推理的输出文件夹（如权重、日志等）。
- `requirements.txt`：项目依赖包列表。
- `ground_truth.json`、`predictions.json` 等：数据集真值及推理结果。



