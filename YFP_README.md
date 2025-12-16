# YFP Facial Palsy Detection System

This system implements facial palsy detection using the HTNet (Hierarchical Transformer Network) architecture on the YouTube Facial Palsy (YFP) dataset. It performs binary classification to detect the presence of facial palsy.

## Overview

The system provides:
- **Binary Classification**: Distinguishes between Normal (0) and Facial Palsy (1) cases
- **Leave-One-Subject-Out (LOSO) Cross-Validation**: Ensures robust evaluation
- **Comprehensive Evaluation Metrics**: Including accuracy, precision, recall, F1-score, AUC-ROC, and more
- **Optical Flow Support**: Can use optical flow features for enhanced motion detection
- **Visualization**: Generates confusion matrices and ROC curves

## Dataset Format

### For Raw Images

Create a CSV file with the following columns:
- `subject_id`: Unique identifier for each subject
- `image_path`: Relative path to the image from data_root
- `label`: 0 for Normal, 1 for Facial Palsy

Example (`yfp_dataset_template.csv`):
```csv
subject_id,image_path,label
subject_001,subject_001/img_001.jpg,0
subject_002,subject_002/img_001.jpg,1
```

### For Optical Flow (Onset-Apex Pairs)

Create a CSV file with the following columns:
- `subject_id`: Unique identifier for each subject
- `onset_frame`: Path to the onset (neutral) frame
- `apex_frame`: Path to the apex (peak expression) frame
- `label`: 0 for Normal, 1 for Facial Palsy

Example (`yfp_dataset_optical_flow_template.csv`):
```csv
subject_id,onset_frame,apex_frame,label
subject_001,subject_001/onset_001.jpg,subject_001/apex_001.jpg,0
subject_002,subject_002/onset_001.jpg,subject_002/apex_001.jpg,1
```

## Directory Structure

```
project/
├── datasets/
│   ├── YFP/                          # Image data directory
│   │   ├── subject_001/
│   │   │   ├── img_001.jpg
│   │   │   └── ...
│   │   └── subject_002/
│   │       └── ...
│   └── yfp_dataset.csv              # Dataset annotation file
├── yfp_weights/                      # Saved model weights (created automatically)
├── yfp_results/                      # Results and metrics (created automatically)
├── train_yfp_palsy_detection.py     # Main training script
├── yfp_dataset.py                    # Dataset classes
├── evaluation_metrics.py             # Comprehensive metrics
└── Model.py                          # HTNet model architecture
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- torch
- torchvision
- numpy
- pandas
- opencv-python
- scikit-learn
- matplotlib
- seaborn
- facenet-pytorch
- einops

## Usage

### Training

Train the model with LOSO cross-validation:

```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow false \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.00005
```

### Training with Optical Flow

For better performance, use optical flow features:

```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_optical_flow_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow true \
    --batch_size 32 \
    --epochs 100
```

### Evaluation Only

Evaluate pre-trained models:

```bash
python train_yfp_palsy_detection.py \
    --train false \
    --csv_file ./datasets/yfp_dataset.csv \
    --data_root ./datasets/YFP \
    --weights_dir ./yfp_weights \
    --results_dir ./yfp_results
```

## Command Line Arguments

### Dataset Parameters
- `--csv_file`: Path to CSV file with dataset annotations (default: `./datasets/yfp_dataset.csv`)
- `--data_root`: Root directory containing the images (default: `./datasets/YFP`)
- `--use_optical_flow`: Whether to use optical flow features (default: `true`)

### Model Parameters
- `--image_size`: Input image size (default: `28`)
- `--patch_size`: Patch size for vision transformer (default: `7`)
- `--dim`: Dimension of transformer (default: `256`)
- `--heads`: Number of attention heads (default: `3`)
- `--num_hierarchies`: Number of hierarchical levels (default: `3`)
- `--block_repeats`: Number of transformer blocks at each hierarchy (default: `2 2 10`)

### Training Parameters
- `--train`: Train or evaluate only (default: `false`)
- `--batch_size`: Batch size for training (default: `32`)
- `--epochs`: Number of training epochs (default: `100`)
- `--learning_rate`: Learning rate (default: `0.00005`)
- `--num_workers`: Number of data loading workers (default: `4`)
- `--print_freq`: Print frequency during training (default: `10`)

### Output Parameters
- `--weights_dir`: Directory to save model weights (default: `./yfp_weights`)
- `--results_dir`: Directory to save results and metrics (default: `./yfp_results`)

## Evaluation Metrics

The system computes comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Average of recall for each class
- **Precision**: Proportion of positive predictions that are correct
- **Recall/Sensitivity**: Proportion of actual positives correctly identified
- **Specificity**: Proportion of actual negatives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **NPV (Negative Predictive Value)**: Proportion of negative predictions that are correct
- **AUC-ROC**: Area Under the ROC Curve

### Statistical Metrics
- **Matthews Correlation Coefficient (MCC)**: Correlation between predictions and ground truth
- **Cohen's Kappa**: Inter-rater agreement statistic

### Error Rates
- **False Positive Rate (FPR)**: Proportion of negatives incorrectly classified as positive
- **False Negative Rate (FNR)**: Proportion of positives incorrectly classified as negative

### Confusion Matrix
- **True Positive (TP)**: Correctly identified palsy cases
- **True Negative (TN)**: Correctly identified normal cases
- **False Positive (FP)**: Normal cases incorrectly identified as palsy
- **False Negative (FN)**: Palsy cases incorrectly identified as normal

## Output Files

After training/evaluation, the following files are generated in the `results_dir`:

1. **Per-Subject Results**:
   - `confusion_matrix_subject_XXX.png`: Confusion matrix for each subject
   - `roc_curve_subject_XXX.png`: ROC curve for each subject

2. **Overall Results**:
   - `confusion_matrix_overall.png`: Overall confusion matrix across all subjects
   - `roc_curve_overall.png`: Overall ROC curve
   - `overall_metrics.txt`: Detailed metrics in text format
   - `per_subject_summary.csv`: Summary of metrics for each subject

3. **Model Weights**:
   - `subject_XXX.pth`: Trained model weights for each subject (in `weights_dir`)

## Example Output

```
================================================================================
YFP FACIAL PALSY DETECTION - Leave-One-Subject-Out Cross-Validation
================================================================================

Device: cuda

Loading dataset from: ./datasets/yfp_dataset.csv
Using optical flow features

Total subjects: 50
Total samples: 1000

================================================================================
LOSO Fold 1/50 - Test Subject: subject_001
================================================================================
Train samples: 980, Test samples: 20

Training...
Epoch [1/100] Train Loss: 0.6523, Train Acc: 0.6234 | Test Loss: 0.5821, Test Acc: 0.7000
...
Best Test Accuracy: 0.8500 at Epoch 45

--- Results for Subject subject_001 ---
============================================================
FACIAL PALSY DETECTION - EVALUATION METRICS
============================================================

--- Classification Metrics ---
Accuracy:           0.8500
Balanced Accuracy:  0.8333
Precision:          0.8750
Recall/Sensitivity: 0.7000
Specificity:        0.9667
F1 Score:           0.7778
NPV:                0.8529
AUC-ROC:            0.9100

--- Statistical Metrics ---
Matthews Corr Coef: 0.6892
Cohen's Kappa:      0.6667
...
```

## Features

### 1. Face Detection and Alignment
- Automatic face detection using MTCNN
- Face alignment and normalization
- Fallback mechanisms for failed detections

### 2. Optical Flow Analysis
- Farneback optical flow computation between onset and apex frames
- Enhanced motion feature extraction
- Multi-channel flow representation (u, v, magnitude)

### 3. Leave-One-Subject-Out Cross-Validation
- Ensures model generalization to unseen subjects
- Prevents data leakage
- Provides robust performance estimates

### 4. Visualization
- Confusion matrices for each subject and overall
- ROC curves with AUC scores
- High-quality publication-ready figures

## Model Architecture

The system uses HTNet (Hierarchical Transformer Network):
- **Hierarchical Structure**: Multi-scale feature extraction
- **Transformer Blocks**: Self-attention mechanisms for spatial relationships
- **Patch-based Processing**: Divides face into patches for detailed analysis
- **Aggregation Layers**: Combines features across different scales

### Default Architecture
- Image size: 28×28
- Patch size: 7×7
- Transformer dimension: 256
- Attention heads: 3
- Hierarchical levels: 3
- Block repeats: [2, 2, 10]

## Tips for Best Performance

1. **Use Optical Flow**: Optical flow features generally provide better performance for detecting subtle facial asymmetries

2. **Data Augmentation**: Consider adding data augmentation for small datasets:
   - Random horizontal flips (be careful with left/right asymmetry)
   - Slight rotations
   - Brightness/contrast adjustments

3. **Hyperparameter Tuning**: Adjust based on your dataset:
   - Learning rate: Try values between 1e-5 and 1e-4
   - Batch size: Larger batches (32-64) for stable training
   - Epochs: Monitor validation performance to avoid overfitting

4. **Class Imbalance**: If your dataset is imbalanced:
   - Use weighted loss functions
   - Consider oversampling minority class
   - Focus on balanced accuracy and F1 score

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `image_size`
- Reduce model `dim` parameter

### Poor Performance
- Check data quality and labels
- Ensure faces are properly detected and aligned
- Try using optical flow features
- Increase number of training epochs
- Adjust learning rate

### Face Detection Failures
- The system has fallback mechanisms
- Consider preprocessing images for better face detection
- Adjust MTCNN parameters if needed

## Citation

If you use this system in your research, please cite:

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

## License

Please refer to the original HTNet paper and repository for license information.

## Contact

For questions or issues, please open an issue in the repository.
