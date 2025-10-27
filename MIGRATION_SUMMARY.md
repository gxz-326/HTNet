# HTNet Migration Summary: Micro-expression → Facial Palsy Recognition

## Overview

This document summarizes the migration of HTNet from micro-expression recognition to facial palsy (facial nerve paralysis) recognition and grading for FNP and CK+ datasets.

## Migration Date
October 27, 2024

## Original System
- **Task**: Micro-expression recognition
- **Datasets**: CASME II, SMIC, SAMM
- **Classes**: 3 (positive, negative, surprise)
- **Focus**: Temporal micro-movements in facial expressions
- **Key files**: `main_HTNet.py`, `Model.py`

## Migrated System
- **Task**: Facial palsy recognition and grading
- **Datasets**: FNP (Facial Nerve Palsy), CK+
- **Classes**: 6 (House-Brackmann scale I-VI) or 3 (simplified)
- **Focus**: Facial symmetry and movement capability
- **Key files**: See "New Files Created" section below

## Key Changes

### 1. Task Adaptation
- **From**: Emotion classification (3 classes)
- **To**: Medical grading (6 classes - House-Brackmann scale)
- **Rationale**: HTNet's hierarchical architecture is well-suited for analyzing facial asymmetry and movement deficits

### 2. Dataset Support
- **New datasets**: FNP and CK+
- **Dataset structure**: Grade-based folders (grade_0 to grade_5)
- **Annotations**: CSV files with image_path, grade, side, split columns
- **Preprocessing**: Maintained facial landmark detection using MTCNN

### 3. Model Architecture
- **Core architecture**: Preserved HTNet's hierarchical transformer design
- **Modifications**: 
  - Adjusted `num_classes` parameter (3 → 6)
  - Maintained multi-scale facial region analysis
  - Kept attention mechanisms for diagnostic region focus

### 4. Data Processing
- **Facial regions**: Still analyzes eyes, nose, and mouth regions
- **Key difference**: Preserved left-right asymmetry (critical for palsy assessment)
- **Augmentation**: Custom augmentations that don't flip or distort asymmetry

## New Files Created

### Core Implementation Files

1. **`facial_palsy_dataset.py`** (9.8 KB)
   - `FacialPalsyDataset` class for FNP dataset
   - `CKPlusFacialPalsyDataset` class for CK+ dataset
   - Facial landmark detection and region extraction
   - Hierarchical input creation

2. **`train_facial_palsy.py`** (11.9 KB)
   - Complete training pipeline
   - Command-line interface with argparse
   - Metrics computation (accuracy, F1, per-class accuracy)
   - Model checkpointing and logging
   - Learning rate scheduling

3. **`evaluate_facial_palsy.py`** (8.8 KB)
   - Comprehensive evaluation metrics
   - Confusion matrix generation and visualization
   - Classification report
   - Results export (text, images, npz)

4. **`demo_inference.py`** (9.9 KB)
   - `FacialPalsyPredictor` class for easy inference
   - Single image prediction
   - Batch prediction
   - Visualization with grade overlay

### Utility Files

5. **`prepare_dataset.py`** (7.3 KB)
   - Dataset preparation for FNP
   - CK+ dataset conversion
   - Train/val/test splitting with stratification
   - 3-class to 6-class conversion

6. **`data_augmentation.py`** (9.8 KB)
   - `FacialPalsyAugmentation` - standard augmentation
   - `AsymmetryPreservingAugmentation` - preserves left-right asymmetry
   - `MixUp` - smooth decision boundaries
   - `CutOut` and `RandomErasing` - robustness to occlusions

7. **`visualize_attention.py`** (10.5 KB)
   - `AttentionVisualizer` class
   - Attention map extraction from HTNet layers
   - Hierarchical attention visualization
   - Facial region overlay

### Documentation Files

8. **`README_FACIAL_PALSY.md`** (detailed English documentation)
   - Complete usage guide
   - Installation instructions
   - Training/evaluation examples
   - Troubleshooting section
   - Configuration options

9. **`README_CN.md`** (Chinese documentation)
   - Full Chinese translation
   - Quick start guide
   - Detailed examples
   - Troubleshooting in Chinese

10. **`config_examples.yaml`**
    - 7 configuration templates
    - Different scenarios (6-class, 3-class, transfer learning, etc.)
    - Model size variations (small, standard, large)
    - Combined dataset configuration

11. **`quick_start.sh`**
    - End-to-end workflow script
    - From dataset preparation to inference
    - Automated pipeline example

12. **`MIGRATION_SUMMARY.md`** (this file)
    - Migration overview
    - Changes documentation
    - File listing

### Modified Files

13. **`README.md`**
    - Added facial palsy section
    - Quick start instructions
    - Links to detailed documentation
    - Feature highlights

14. **`requirements.txt`**
    - Added seaborn for visualizations
    - Cleaned up duplicate Pillow entries
    - Maintained all original dependencies

## Architecture Compatibility

### Why HTNet Works for Facial Palsy

1. **Hierarchical Processing**
   - Low-level: Fine-grained features in local regions (eyes, mouth)
   - High-level: Coarse-grained features capturing facial asymmetry
   - Perfect for multi-scale palsy assessment

2. **Region-based Analysis**
   - Original: 4-5 facial regions (eyes, mouth, nose)
   - Maintained: Same regions, different interpretation
   - Eyes: Closure ability and symmetry
   - Mouth: Smile symmetry and movement range

3. **Attention Mechanism**
   - Focuses on diagnostically relevant regions
   - Learns which facial areas are important for grading
   - Provides interpretability through attention visualization

## Usage Examples

### Basic Training
```bash
python train_facial_palsy.py \
    --data_root ./datasets/facial_palsy/FNP \
    --train_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --val_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6 \
    --batch_size 32 \
    --epochs 100
```

### Transfer Learning
```bash
python train_facial_palsy.py \
    --pretrained_path ./ourmodel_threedatasets_weights/best_model.pth \
    --data_root ./datasets/facial_palsy/FNP \
    --num_classes 6 \
    --epochs 80 \
    --learning_rate 0.00005
```

### Evaluation
```bash
python evaluate_facial_palsy.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --data_root ./datasets/facial_palsy/FNP \
    --test_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6
```

### Inference
```bash
# Single image
python demo_inference.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_path ./test_image.jpg

# Batch processing
python demo_inference.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_dir ./test_images/ \
    --output_csv ./predictions.csv
```

### Visualization
```bash
python visualize_attention.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_path ./test_image.jpg \
    --output_dir ./attention_visualizations \
    --visualize_regions
```

## Expected Performance

Based on typical medical image classification tasks:

- **Overall Accuracy**: 60-85%
- **F1 Score (Macro)**: 55-80%
- **Mean Class Accuracy**: 58-82%

Performance varies based on:
- Dataset size and quality
- Class balance
- Image quality and consistency
- Model configuration

## Advantages of This Migration

1. **Minimal Code Changes**: Preserved core HTNet architecture
2. **Modular Design**: Easy to switch between datasets
3. **Comprehensive Tools**: Full pipeline from data prep to visualization
4. **Transfer Learning**: Can leverage micro-expression pretrained weights
5. **Clinical Relevance**: Follows standard House-Brackmann grading
6. **Interpretability**: Attention visualization for understanding decisions

## Limitations and Considerations

1. **Dataset Requirements**: Needs well-organized, labeled facial palsy images
2. **Asymmetry Preservation**: Augmentations must not flip images horizontally
3. **Medical Context**: This is a research tool, not for clinical diagnosis
4. **Class Imbalance**: May require weighted loss or data balancing
5. **Cross-dataset**: CK+ originally for expressions, needs proper annotation

## Future Enhancements

Potential improvements mentioned in documentation:

1. Data augmentation expansion
2. Multi-task learning (grade + affected side)
3. Temporal analysis using video sequences
4. Attention visualization improvements
5. Ensemble methods
6. Cross-dataset validation studies

## Backward Compatibility

- **Original micro-expression code**: Fully preserved
- **Can still run**: `python main_HTNet.py --train True`
- **Model.py**: Unchanged, supports both tasks
- **Datasets**: Original datasets remain accessible

## Dependencies

All dependencies maintained, with additions:
- seaborn (for confusion matrix visualization)
- All original packages preserved

## Testing

All Python files compile successfully:
```bash
python -m py_compile facial_palsy_dataset.py
python -m py_compile train_facial_palsy.py
python -m py_compile evaluate_facial_palsy.py
python -m py_compile demo_inference.py
python -m py_compile prepare_dataset.py
python -m py_compile data_augmentation.py
python -m py_compile visualize_attention.py
```

## Conclusion

Successfully migrated HTNet from micro-expression recognition to facial palsy grading while:
- Preserving the original codebase
- Maintaining architectural advantages
- Adding comprehensive tools and documentation
- Supporting multiple datasets (FNP, CK+)
- Providing both English and Chinese documentation
- Including visualization and interpretation tools

The migration leverages HTNet's strengths in hierarchical facial analysis while adapting it to the specific requirements of medical facial palsy assessment.
