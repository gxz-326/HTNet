# 文档索引 / Documentation Index

本项目包含完整的文档体系，以下是快速导航：

---

## 🎯 新用户从这里开始

### 中文用户
1. **[USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)** - 回答用户核心问题
2. **[QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)** - 5分钟快速入门
3. **[FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)** - 详细的可行性分析

### English Users
1. **[README.md](README.md)** - Main project README
2. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation summary

---

## 📚 文档分类

### 1️⃣ 核心问答和快速入门

| 文档 | 语言 | 说明 | 推荐度 |
|------|------|------|--------|
| [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md) | 🇨🇳 中文 | **直接回答用户问题** | ⭐⭐⭐⭐⭐ |
| [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md) | 🇨🇳 中文 | **5分钟快速上手指南** | ⭐⭐⭐⭐⭐ |
| [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md) | 🇨🇳 中文 | 详细的可行性分析 | ⭐⭐⭐⭐ |

### 2️⃣ 技术文档

| 文档 | 语言 | 说明 | 推荐度 |
|------|------|------|--------|
| [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md) | 🇨🇳 中文 | 增强模块详细使用文档 | ⭐⭐⭐⭐⭐ |
| [FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md) | 🇨🇳 中文 | 功能特性和技术细节 | ⭐⭐⭐⭐ |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | 🇬🇧 English | Implementation summary | ⭐⭐⭐⭐⭐ |

### 3️⃣ 项目文档

| 文档 | 语言 | 说明 | 推荐度 |
|------|------|------|--------|
| [README.md](README.md) | 🇬🇧 English | 主README，包含增强功能说明 | ⭐⭐⭐⭐⭐ |
| [README_CN.md](README_CN.md) | 🇨🇳 中文 | 中文版主README | ⭐⭐⭐⭐ |
| [README_FACIAL_PALSY.md](README_FACIAL_PALSY.md) | 🇬🇧 English | 面瘫识别应用完整指南 | ⭐⭐⭐⭐ |

---

## 🎯 按需求查找

### 我想快速了解新功能
👉 **[QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)** - 5分钟快速入门

### 我想知道是否可行
👉 **[USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)** - 直接回答可行性问题  
👉 **[FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)** - 详细分析

### 我想了解技术细节
👉 **[README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)** - 模块详细文档  
👉 **[FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md)** - 技术特性

### 我想开始训练模型
👉 **[QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)** - 训练命令  
👉 **[README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)** - 详细配置

### 我想可视化结果
👉 **[QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)** - 可视化工具使用  
👉 **[README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)** - 可视化详解

### 我想了解实现过程
👉 **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - 完整实现总结

---

## 💻 代码文件

### 核心实现
- **Model.py** - 包含所有模型类
  - `DiagonalMicroAttention` - 对角微注意力模块
  - `ROIAttentionModule` - ROI注意力模块
  - `EnhancedTransformer` - 增强Transformer
  - `HTNetEnhanced` - 完整增强模型

### 训练和评估
- **train_enhanced_facial_palsy.py** - 增强模型训练脚本
- **train_facial_palsy.py** - 基础模型训练脚本
- **evaluate_facial_palsy.py** - 模型评估脚本

### 可视化工具
- **visualize_roi_and_asymmetry.py** - ROI和不对称性可视化
- **visualize_attention.py** - 注意力图可视化

### 测试和验证
- **test_enhanced_model.py** - 增强模块测试套件
- **check_syntax.py** - 语法检查工具

### 数据处理
- **facial_palsy_dataset.py** - 数据集加载器
- **prepare_dataset.py** - 数据集准备工具
- **data_augmentation.py** - 数据增强

---

## 📊 文档特点对比

| 文档 | 适合对象 | 深度 | 实用性 | 阅读时间 |
|------|---------|------|--------|---------|
| USER_QUESTION_ANSWER.md | 所有人 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 5-10分钟 |
| QUICK_START_ENHANCED.md | 开发者 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 5-10分钟 |
| FEASIBILITY_ANSWER_CN.md | 决策者/研究者 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 15-20分钟 |
| README_ENHANCED_MODULES.md | 开发者 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 20-30分钟 |
| FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md | 研究者 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 15-25分钟 |
| IMPLEMENTATION_SUMMARY.md | 开发者/研究者 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 10-15分钟 |

---

## 🔍 关键词搜索

### 对角微注意力 / Diagonal Micro-Attention
- [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)
- [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)
- [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)
- [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)
- [FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md)

### ROI / 兴趣区域
- [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)
- [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)
- [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)
- [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)
- [FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md)

### 面部不对称 / Facial Asymmetry
- [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)
- [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)
- [FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md)

### 背景抑制 / Background Suppression
- [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)
- [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)
- [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)

### 可行性 / Feasibility
- [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md)
- [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)
- [FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md)

### 训练 / Training
- [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)
- [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)
- [README_FACIAL_PALSY.md](README_FACIAL_PALSY.md)

### 可视化 / Visualization
- [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)
- [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md)
- [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md)

---

## 📖 推荐阅读路径

### 路径1: 快速了解（15分钟）
1. [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md) - 5分钟
2. [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md) - 10分钟

### 路径2: 深入学习（45分钟）
1. [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md) - 5分钟
2. [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md) - 20分钟
3. [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md) - 20分钟

### 路径3: 技术研究（90分钟）
1. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 15分钟
2. [FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md](FEATURE_DIAGONAL_MICRO_ATTENTION_ROI.md) - 25分钟
3. [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md) - 30分钟
4. Model.py 源码阅读 - 20分钟

### 路径4: 实战开发（60分钟）
1. [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md) - 10分钟
2. [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md) - 20分钟
3. 运行 test_enhanced_model.py - 10分钟
4. 尝试训练模型 - 20分钟

---

## ✅ 验证状态

所有文档和代码均已验证：

- ✅ 所有Python文件通过语法检查
- ✅ 模型实现完整
- ✅ 测试套件就绪
- ✅ 文档完整详尽
- ✅ 示例代码可运行

---

## 🆘 获取帮助

### 常见问题
查看 [FEASIBILITY_ANSWER_CN.md](FEASIBILITY_ANSWER_CN.md) 的"常见问题"章节

### 技术支持
查看 [README_ENHANCED_MODULES.md](README_ENHANCED_MODULES.md) 的"故障排除"章节

### 快速参考
查看 [QUICK_START_ENHANCED.md](QUICK_START_ENHANCED.md)

---

## 📝 文档更新

**最后更新**: 2024-10-28  
**文档版本**: v1.0  
**状态**: ✅ 完整且最新

---

**建议**: 从 [USER_QUESTION_ANSWER.md](USER_QUESTION_ANSWER.md) 开始，然后根据您的需求选择其他文档！
