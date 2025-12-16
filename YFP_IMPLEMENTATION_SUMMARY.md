# YFP Facial Palsy Detection - Implementation Summary

This document summarizes the implementation of facial palsy detection using the YFP (YouTube Facial Palsy) dataset with comprehensive evaluation metrics.

## Overview

The system has been enhanced to support **binary classification for facial palsy detection**:
- **Class 0**: Normal (no facial palsy)
- **Class 1**: Facial Palsy

The implementation uses the HTNet (Hierarchical Transformer Network) architecture with Leave-One-Subject-Out (LOSO) cross-validation for robust evaluation.

## New Files Created

### 1. Core Implementation Files

#### `yfp_dataset.py`
Dataset classes for YFP facial palsy data.

**Classes:**
- `YFPFacialPalsyDataset`: Dataset class for raw facial images
- `YFPOpticalFlowDataset`: Dataset class with optical flow computation between onset and apex frames

**Features:**
- Automatic face detection and alignment using MTCNN
- Fallback mechanisms for failed face detections
- Support for both raw images and optical flow features
- Subject-based data splitting for LOSO cross-validation

#### `evaluation_metrics.py`
Comprehensive evaluation metrics for facial palsy detection.

**Classes:**
- `FacialPalsyMetrics`: Complete metrics computation and visualization

**Metrics Computed:**
- **Classification Metrics**: Accuracy, Precision, Recall/Sensitivity, Specificity, F1 Score, NPV
- **Statistical Metrics**: Matthews Correlation Coefficient (MCC), Cohen's Kappa, Balanced Accuracy
- **Error Rates**: False Positive Rate (FPR), False Negative Rate (FNR)
- **ROC Analysis**: AUC-ROC with curve visualization
- **Confusion Matrix**: With TP, TN, FP, FN counts

**Visualization Methods:**
- `plot_confusion_matrix()`: Publication-quality confusion matrix
- `plot_roc_curve()`: ROC curve with AUC score
- `save_metrics_to_file()`: Export metrics to text file
- `print_metrics()`: Formatted console output

#### `train_yfp_palsy_detection.py`
Main training and evaluation script.

**Features:**
- LOSO (Leave-One-Subject-Out) cross-validation
- Automatic subject-based data splitting
- Per-subject and overall evaluation
- Model weight saving per subject
- Comprehensive result logging
- Support for both raw images and optical flow

**Key Functions:**
- `train_epoch()`: Single epoch training
- `evaluate()`: Model evaluation with metrics
- `train_subject_loso()`: Complete LOSO cross-validation
- `main()`: Entry point with argument parsing

**Command Line Arguments:**
```
Dataset Parameters:
  --csv_file              Path to CSV with annotations
  --data_root             Root directory for images
  --use_optical_flow      Use optical flow features (true/false)

Model Parameters:
  --image_size            Input image size (default: 28)
  --patch_size            Patch size (default: 7)
  --dim                   Transformer dimension (default: 256)
  --heads                 Attention heads (default: 3)
  --num_hierarchies       Hierarchical levels (default: 3)
  --block_repeats         Block repeats per level (default: [2,2,10])

Training Parameters:
  --train                 Train or evaluate only (true/false)
  --batch_size            Batch size (default: 32)
  --epochs                Number of epochs (default: 100)
  --learning_rate         Learning rate (default: 0.00005)
  --num_workers           Data loading workers (default: 4)
  --print_freq            Print frequency (default: 10)

Output Parameters:
  --weights_dir           Directory for model weights (default: ./yfp_weights)
  --results_dir           Directory for results (default: ./yfp_results)
```

### 2. Utility and Testing Files

#### `test_metrics.py`
Standalone test script for evaluation metrics module.

**Features:**
- Tests metrics with sample data
- Demonstrates perfect classification scenario
- Demonstrates random classification scenario
- Generates sample visualizations
- Validates metric computations

**Output Files:**
- `test_metrics_output.txt`
- `test_confusion_matrix.png`
- `test_roc_curve.png`

#### `create_sample_yfp_dataset.py`
Helper script to create synthetic YFP dataset for testing.

**Features:**
- Generates synthetic face images
- Creates normal and palsy variants
- Simulates facial asymmetry for palsy cases
- Generates both raw image and optical flow CSV files
- Creates proper directory structure

**Usage:**
```bash
python create_sample_yfp_dataset.py \
    --output_dir ./datasets/YFP_sample \
    --num_subjects 10 \
    --images_per_subject 5
```

### 3. Documentation Files

#### `YFP_README.md`
Comprehensive documentation for YFP facial palsy detection.

**Contents:**
- System overview and features
- Dataset format specifications
- Directory structure
- Installation instructions
- Detailed usage examples
- Command-line arguments reference
- Evaluation metrics explanation
- Output files description
- Model architecture details
- Performance tips
- Troubleshooting guide

#### `QUICKSTART_YFP.md`
Quick start guide for rapid setup.

**Contents:**
- 5-step quick start process
- Dataset preparation (two options)
- Installation commands
- Training commands
- Results interpretation
- Common parameter adjustments
- Troubleshooting quick tips

#### `YFP_IMPLEMENTATION_SUMMARY.md` (this file)
Summary of the implementation for developers.

### 4. Template and Configuration Files

#### `yfp_dataset_template.csv`
Template for raw image dataset CSV.

**Columns:**
- `subject_id`: Unique subject identifier
- `image_path`: Relative path to image
- `label`: 0 (normal) or 1 (palsy)

#### `yfp_optical_flow_template.csv`
Template for optical flow dataset CSV.

**Columns:**
- `subject_id`: Unique subject identifier
- `onset_frame`: Path to neutral/onset frame
- `apex_frame`: Path to peak/apex frame
- `label`: 0 (normal) or 1 (palsy)

### 5. Modified Files

#### `requirements.txt`
Updated to include `seaborn>=0.11.0` for enhanced visualizations and removed duplicate Pillow entries.

#### `README.md`
Updated with:
- New section highlighting YFP facial palsy detection feature
- Quick start example for YFP
- Links to detailed documentation
- Key features list

## Output Structure

### During Training

For each LOSO fold (subject):
```
================================================================================
LOSO Fold X/N - Test Subject: subject_XXX
================================================================================
Train samples: NNN, Test samples: NN

Training...
Epoch [1/100] Train Loss: X.XXXX, Train Acc: X.XXXX | Test Loss: X.XXXX, Test Acc: X.XXXX
...
Best Test Accuracy: X.XXXX at Epoch XX

--- Results for Subject subject_XXX ---
============================================================
FACIAL PALSY DETECTION - EVALUATION METRICS
============================================================

--- Classification Metrics ---
Accuracy:           X.XXXX
Balanced Accuracy:  X.XXXX
Precision:          X.XXXX
Recall/Sensitivity: X.XXXX
Specificity:        X.XXXX
F1 Score:           X.XXXX
...
```

### Output Files Generated

#### In `weights_dir` (default: `./yfp_weights/`)
- `subject_001.pth` - Model weights for subject 1
- `subject_002.pth` - Model weights for subject 2
- ... (one per subject)

#### In `results_dir` (default: `./yfp_results/`)

**Per-Subject Results:**
- `confusion_matrix_subject_001.png`
- `confusion_matrix_subject_002.png`
- `roc_curve_subject_001.png`
- `roc_curve_subject_002.png`
- ... (one pair per subject)

**Overall Results:**
- `confusion_matrix_overall.png` - Overall confusion matrix
- `roc_curve_overall.png` - Overall ROC curve
- `overall_metrics.txt` - Detailed metrics in text format
- `per_subject_summary.csv` - CSV with metrics for each subject

## Evaluation Metrics Details

### Classification Metrics

1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness of predictions

2. **Precision**: TP / (TP + FP)
   - Proportion of positive predictions that are correct
   - Important when false positives are costly

3. **Recall (Sensitivity)**: TP / (TP + FN)
   - Proportion of actual positives correctly identified
   - Critical for medical diagnosis (don't miss palsy cases)

4. **Specificity**: TN / (TN + FP)
   - Proportion of actual negatives correctly identified
   - Important to avoid over-diagnosis

5. **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   - Good overall performance metric

6. **NPV (Negative Predictive Value)**: TN / (TN + FN)
   - Reliability of negative predictions

7. **Balanced Accuracy**: (Sensitivity + Specificity) / 2
   - Average performance across both classes
   - Better than accuracy for imbalanced datasets

8. **AUC-ROC**: Area Under the ROC Curve
   - Overall discriminative ability
   - Values: 0.5 (random) to 1.0 (perfect)

### Statistical Metrics

1. **Matthews Correlation Coefficient (MCC)**: Range [-1, 1]
   - Correlation between predictions and ground truth
   - More informative than accuracy for imbalanced datasets

2. **Cohen's Kappa**: Agreement between predictions and ground truth
   - Accounts for chance agreement

## Usage Examples

### Example 1: Train with Optical Flow (Recommended)

```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_optical_flow_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow true \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.00005
```

### Example 2: Train with Raw Images

```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow false \
    --batch_size 32 \
    --epochs 100
```

### Example 3: Evaluate Pre-trained Models

```bash
python train_yfp_palsy_detection.py \
    --train false \
    --csv_file ./datasets/yfp_dataset.csv \
    --data_root ./datasets/YFP \
    --weights_dir ./yfp_weights \
    --results_dir ./yfp_results
```

### Example 4: Create Sample Dataset and Test

```bash
# Create sample dataset
python create_sample_yfp_dataset.py \
    --output_dir ./datasets/YFP_sample \
    --num_subjects 10 \
    --images_per_subject 5

# Train on sample dataset
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/YFP_sample/yfp_optical_flow_dataset.csv \
    --data_root ./datasets/YFP_sample \
    --use_optical_flow true \
    --epochs 20 \
    --batch_size 8
```

## Key Design Decisions

### 1. Leave-One-Subject-Out (LOSO) Cross-Validation
- **Why**: Ensures model generalizes to unseen subjects
- **How**: Each subject used as test set once
- **Benefit**: Prevents overfitting to specific individuals

### 2. Binary Classification (Normal vs. Palsy)
- **Why**: Clear, actionable clinical decision
- **How**: Modified HTNet to output 2 classes instead of 3
- **Benefit**: Simpler interpretation and deployment

### 3. Optical Flow Support
- **Why**: Captures motion and asymmetry better than static images
- **How**: Compute Farneback optical flow between onset and apex frames
- **Benefit**: Enhanced performance on subtle asymmetries

### 4. Comprehensive Metrics
- **Why**: Single accuracy metric insufficient for medical applications
- **How**: Implemented FacialPalsyMetrics class with 15+ metrics
- **Benefit**: Complete performance characterization

### 5. Automatic Face Detection with Fallback
- **Why**: Real-world images vary in quality
- **How**: MTCNN detection with resize fallback
- **Benefit**: Robust to challenging images

## Performance Expectations

### Good Performance Benchmarks
For a well-prepared YFP dataset, expect:
- **Accuracy**: > 80%
- **F1 Score**: > 75%
- **AUC-ROC**: > 85%
- **Sensitivity**: > 80% (critical for medical applications)
- **Specificity**: > 80%

### Factors Affecting Performance
1. **Dataset Quality**: Clear images with visible faces
2. **Dataset Size**: More subjects → better generalization
3. **Class Balance**: Roughly equal normal and palsy cases
4. **Label Quality**: Accurate ground truth labels
5. **Feature Type**: Optical flow generally outperforms raw images

## Integration with Existing Code

The implementation is designed as an **addition** to the existing micro-expression recognition system:

- **Original files**: Unchanged (main_HTNet.py, confusion_matrix.py, Alex_for_three_datasets.py, etc.)
- **Shared components**: Model.py (HTNet architecture) used by both systems
- **Separate workflows**: YFP detection uses its own scripts and doesn't interfere with existing functionality

## Future Extensions

Potential improvements and extensions:

1. **Multi-grade Classification**: Implement House-Brackmann grading (I-VI)
2. **Ensemble Methods**: Combine multiple models for better performance
3. **Data Augmentation**: Random crops, rotations, brightness adjustments
4. **Attention Visualization**: Visualize which facial regions the model focuses on
5. **Real-time Detection**: Optimize for video stream processing
6. **Transfer Learning**: Pre-train on larger datasets
7. **Multi-modal Features**: Combine optical flow with landmark-based features

## Testing and Validation

### Unit Tests
- `test_metrics.py`: Validates metric computations

### Integration Tests
- `create_sample_yfp_dataset.py`: Creates test data
- Can run end-to-end training on sample data

### Validation Checklist
- ✅ Dataset loading works correctly
- ✅ Face detection handles failures gracefully
- ✅ Optical flow computation works
- ✅ LOSO cross-validation splits correctly
- ✅ Metrics computed accurately
- ✅ Visualizations generated properly
- ✅ Results saved to correct locations

## Troubleshooting

### Common Issues and Solutions

1. **CUDA out of memory**
   - Solution: Reduce batch_size or image_size

2. **Low accuracy (< 60%)**
   - Check: Label correctness, class balance, face visibility
   - Try: Optical flow, more epochs, different learning rate

3. **Face detection failures**
   - System has automatic fallback
   - Check: Image quality and face visibility

4. **Training very slow**
   - Reduce: image_size, num_workers
   - Use: GPU if available

5. **Import errors**
   - Run: `pip install -r requirements.txt`

## Dependencies

All dependencies listed in `requirements.txt`:
- torch, torchvision (deep learning)
- numpy, pandas (data processing)
- opencv-python (image processing and optical flow)
- facenet-pytorch (face detection)
- scikit-learn (metrics)
- matplotlib, seaborn (visualization)
- einops (tensor operations)
- Pillow (image I/O)

## Conclusion

This implementation provides a complete, production-ready system for facial palsy detection using the YFP dataset with state-of-the-art deep learning and comprehensive evaluation. The modular design allows easy customization and extension while maintaining code quality and documentation.
