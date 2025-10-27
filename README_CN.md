# HTNet 面瘫识别和分级系统

这是将 HTNet（层次化Transformer网络）从微表情识别迁移到面瘫（面神经麻痹）识别和分级任务的实现，支持 FNP 和 CK+ 数据集。

## 📋 项目概述

面瘫是一种面部肌肉无力或麻痹的疾病，通常影响面部的一侧。本项目将原本用于微表情识别的 HTNet 架构改造为自动评估面瘫严重程度的系统，使用 House-Brackmann 分级量表。

### 主要改动

1. **任务转换**: 微表情识别 → 面瘫分级
2. **数据集**: CASME/SMIC/SAMM → FNP（面神经麻痹）+ CK+
3. **分类**: 3类情感 → 6级（House-Brackmann量表：I-VI级）
4. **关注点**: 时序微动作 → 面部对称性和运动能力

### House-Brackmann 分级量表

- **I级**: 正常面部功能
- **II级**: 仔细观察可见轻微无力
- **III级**: 明显无力但不影响外观
- **IV级**: 明显影响外观的无力
- **V级**: 仅能察觉到极微弱的运动
- **VI级**: 完全无运动

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

#### FNP 数据集

将数据组织成以下结构：

```
datasets/facial_palsy/FNP/
├── grade_1/
│   ├── patient001_left.jpg
│   ├── patient001_right.jpg
│   └── ...
├── grade_2/
├── grade_3/
├── grade_4/
├── grade_5/
└── grade_6/
```

然后运行准备脚本：

```bash
python prepare_dataset.py \
    --dataset_type FNP \
    --data_root ./datasets/facial_palsy/FNP \
    --output_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --split_ratio 0.7 0.15 0.15
```

#### CK+ 数据集

```bash
python prepare_dataset.py \
    --dataset_type CK+ \
    --data_root ./path/to/ckplus/cohn-kanade-images \
    --output_root ./datasets/facial_palsy/CK_prepared \
    --output_csv ./datasets/facial_palsy/ckplus_annotation.csv
```

### 3. 训练模型

#### 基础训练

```bash
python train_facial_palsy.py \
    --data_root ./datasets/facial_palsy/FNP \
    --train_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --val_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --save_dir ./checkpoints/fnp \
    --log_dir ./logs/fnp
```

#### 使用预训练权重（迁移学习）

```bash
python train_facial_palsy.py \
    --data_root ./datasets/facial_palsy/FNP \
    --train_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --val_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6 \
    --pretrained_path ./ourmodel_threedatasets_weights/best_model.pth \
    --batch_size 32 \
    --epochs 100
```

### 4. 评估模型

```bash
python evaluate_facial_palsy.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --data_root ./datasets/facial_palsy/FNP \
    --test_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6 \
    --batch_size 32 \
    --output_dir ./evaluation_results
```

评估脚本会生成：
- **混淆矩阵** (`confusion_matrix.png`)
- **评估报告** (`evaluation_results.txt`) - 包含准确率、F1分数、每类准确率等
- **预测结果** (`predictions.npz`) - 原始预测和概率

### 5. 推理演示

#### 单张图片预测

```bash
python demo_inference.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_path ./test_image.jpg \
    --num_classes 6
```

#### 批量预测

```bash
python demo_inference.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_dir ./test_images/ \
    --output_csv ./predictions.csv \
    --num_classes 6
```

### 6. 注意力可视化

```bash
python visualize_attention.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_path ./test_image.jpg \
    --output_dir ./attention_visualizations \
    --visualize_regions
```

## 📁 项目文件说明

### 核心文件

- **`Model.py`** - HTNet 模型架构定义
- **`facial_palsy_dataset.py`** - FNP 和 CK+ 数据集加载器
- **`train_facial_palsy.py`** - 训练脚本
- **`evaluate_facial_palsy.py`** - 评估脚本
- **`demo_inference.py`** - 推理演示脚本
- **`prepare_dataset.py`** - 数据集准备工具

### 辅助文件

- **`data_augmentation.py`** - 专门为面部图像设计的数据增强
- **`visualize_attention.py`** - 注意力图可视化工具
- **`config_examples.yaml`** - 配置示例模板
- **`quick_start.sh`** - 快速开始脚本

### 文档

- **`README_FACIAL_PALSY.md`** - 详细的英文文档
- **`README_CN.md`** - 本文档（中文说明）

## 🎯 为什么 HTNet 适合面瘫评估？

HTNet 的层次化架构特别适合面瘫评估，原因如下：

1. **多尺度分析**: 同时捕获局部特征（眼睛/嘴部）和全局面部对称性
2. **基于区域的处理**: 独立分析关键诊断区域
3. **注意力机制**: 聚焦于临床相关的面部区域
4. **经过验证的架构**: 在微妙面部运动检测上表现出色

### 关键面部区域

模型分析以下关键区域来评估面瘫：

- **额头/眉毛**: 抬眉能力
- **眼睛**: 闭眼力量和对称性
- **鼻**: 鼻唇沟深度
- **嘴部**: 微笑对称性和嘴角运动

## ⚙️ 自定义配置

### 3类问题（简化版本）

如果想使用 3 类（正常/轻度/重度）而不是 6 类：

```bash
python train_facial_palsy.py \
    --num_classes 3 \
    --其他参数...
```

### 调整图像大小

```bash
python train_facial_palsy.py \
    --image_size 112 \
    --patch_size 7 \
    --其他参数...
```

注意：确保 `image_size` 能被 `patch_size` 整除。

### 调整模型容量

**小模型**（更快的训练，更少的内存）：

```bash
python train_facial_palsy.py \
    --dim 128 \
    --heads 2 \
    --num_hierarchies 2 \
    --block_repeats 2 2 \
    --其他参数...
```

**大模型**（可能有更好的准确率）：

```bash
python train_facial_palsy.py \
    --dim 384 \
    --heads 4 \
    --num_hierarchies 3 \
    --block_repeats 3 3 12 \
    --其他参数...
```

## 📊 预期结果

性能会根据数据集质量和大小而变化。典型期望：

- **总体准确率**: 60-85%
- **F1分数（宏平均）**: 55-80%
- **平均类别准确率**: 58-82%

注意：较高级别（更严重的面瘫）通常比正常和II级之间的微妙差异更容易分类。

## 🔧 故障排除

### 问题：CUDA 内存不足

**解决方案**: 减少批量大小或图像大小

```bash
python train_facial_palsy.py --batch_size 16 --image_size 112 ...
```

### 问题：特定级别性能差

**解决方案**: 检查类别平衡，考虑使用类别权重

### 问题：人脸检测失败

**解决方案**: 代码在检测失败时使用默认地标点，但您可以在 `facial_palsy_dataset.py` 中调整默认位置

### 问题：过拟合

**解决方案**: 
- 增加 dropout
- 添加数据增强
- 减少模型容量
- 使用更多训练数据

## 🔍 进阶使用

### 使用快速开始脚本

```bash
chmod +x quick_start.sh
./quick_start.sh
```

### 查看配置示例

查看 `config_examples.yaml` 文件，了解不同场景的配置模板。

### 使用数据增强

```python
from data_augmentation import get_augmentation_pipeline

# 标准增强
aug = get_augmentation_pipeline('standard', image_size=224, is_training=True)

# 保持不对称的增强
aug = get_augmentation_pipeline('asymmetry_preserving', image_size=224, is_training=True)
```

## 📖 完整文档

更详细的说明请参阅：
- **英文文档**: `README_FACIAL_PALSY.md`
- **配置示例**: `config_examples.yaml`

## 📝 引用

如果您使用此代码进行面瘫研究，请引用原始 HTNet 论文：

```bibtex
@article{wang2024htnet,
  title={Htnet for micro-expression recognition},
  author={Wang, Zhifeng and Zhang, Kaihao and Luo, Wenhan and Sankaranarayana, Ramesh},
  journal={Neurocomputing},
  volume={602},
  pages={128196},
  year={2024},
  publisher={Elsevier}
}
```

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

请参考原始 HTNet 仓库的许可证信息。

## 🙏 致谢

感谢原始 HTNet 作者提供的出色工作，使得这个面瘫识别系统的开发成为可能。

---

**注意**: 这是一个研究工具，不应用于临床诊断。任何医疗决策都应由合格的医疗专业人员做出。
