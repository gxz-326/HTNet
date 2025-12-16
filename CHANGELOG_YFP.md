# Changelog - YFP Facial Palsy Detection Implementation

## Version 2.0.0 - YFP Facial Palsy Detection (2024-12-16)

### 🎉 Major New Feature: Facial Palsy Detection

Added complete facial palsy detection system for YouTube Facial Palsy (YFP) dataset with comprehensive evaluation metrics.

### ✨ New Files Added

#### Core Implementation
- **yfp_dataset.py** - Dataset classes for YFP facial palsy data
  - `YFPFacialPalsyDataset`: Raw image dataset loader
  - `YFPOpticalFlowDataset`: Optical flow dataset with onset/apex frames
  - Automatic face detection using MTCNN
  - Robust fallback mechanisms for failed detections

- **evaluation_metrics.py** - Comprehensive metrics module
  - `FacialPalsyMetrics`: Complete evaluation metrics class
  - 15+ evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC, etc.)
  - Confusion matrix visualization
  - ROC curve plotting
  - Statistical metrics (MCC, Cohen's Kappa)
  - Export to text files and images

- **train_yfp_palsy_detection.py** - Main training script
  - Leave-One-Subject-Out (LOSO) cross-validation
  - Binary classification (Normal vs. Palsy)
  - Per-subject and overall evaluation
  - Comprehensive result logging
  - Support for both raw images and optical flow

#### Utilities
- **test_metrics.py** - Test script for evaluation metrics
  - Validates metric computations
  - Demonstrates usage with sample data
  - Generates sample visualizations

- **create_sample_yfp_dataset.py** - Sample dataset generator
  - Creates synthetic face images for testing
  - Generates both normal and palsy variants
  - Creates proper directory structure and CSV files

#### Documentation
- **YFP_README.md** - Comprehensive documentation
  - Complete system overview
  - Dataset format specifications
  - Installation and usage instructions
  - Command-line arguments reference
  - Evaluation metrics explanation
  - Troubleshooting guide

- **QUICKSTART_YFP.md** - Quick start guide
  - 5-step setup process
  - Common usage examples
  - Performance tips

- **YFP_IMPLEMENTATION_SUMMARY.md** - Developer documentation
  - Technical implementation details
  - Design decisions
  - File structure overview

- **CHANGELOG_YFP.md** - This file

#### Templates
- **yfp_dataset_template.csv** - Template for raw image dataset
- **yfp_optical_flow_template.csv** - Template for optical flow dataset

### 🔧 Modified Files

#### requirements.txt
- Added `seaborn>=0.11.0` for enhanced visualizations
- Removed duplicate Pillow entries
- Cleaned up version specifications

#### README.md
- Added prominent YFP Facial Palsy Detection section
- Quick start example for YFP
- Links to detailed documentation
- Feature highlights with emoji

#### .gitignore
- Added YFP output directories (yfp_weights/, yfp_results/)
- Added test output files
- Added comprehensive Python, IDE, and environment patterns
- Better organization with comments

### 🎯 Key Features

#### Binary Classification
- **Normal (Class 0)**: Healthy facial movement
- **Palsy (Class 1)**: Facial palsy detected
- HTNet model adapted for 2-class output

#### Leave-One-Subject-Out Cross-Validation
- Each subject used as test set once
- Prevents overfitting to specific individuals
- Ensures model generalization

#### Optical Flow Support
- Computes Farneback optical flow between onset and apex frames
- Captures subtle motion and asymmetry
- Enhanced performance for palsy detection
- Three-channel flow representation (u, v, magnitude)

#### Comprehensive Evaluation Metrics

**Classification Metrics:**
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- NPV (Negative Predictive Value)
- Balanced Accuracy

**Statistical Metrics:**
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa

**Error Analysis:**
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Confusion Matrix with TP, TN, FP, FN

**ROC Analysis:**
- AUC-ROC computation
- ROC curve visualization

#### Visualization
- Publication-quality confusion matrices
- ROC curves with AUC scores
- Per-subject and overall results
- High-resolution PNG outputs (300 DPI)

#### Robust Face Detection
- MTCNN-based face detection
- Automatic face alignment
- Fallback to image resizing if detection fails
- GPU acceleration support

### 📊 Output Structure

The system generates organized outputs:

```
yfp_weights/
├── subject_001.pth
├── subject_002.pth
└── ...

yfp_results/
├── confusion_matrix_overall.png
├── roc_curve_overall.png
├── overall_metrics.txt
├── per_subject_summary.csv
├── confusion_matrix_subject_001.png
├── roc_curve_subject_001.png
└── ...
```

### 🚀 Usage Examples

#### Train with Optical Flow (Recommended)
```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_optical_flow_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow true \
    --epochs 100
```

#### Evaluate Pre-trained Models
```bash
python train_yfp_palsy_detection.py \
    --train false \
    --weights_dir ./yfp_weights \
    --results_dir ./yfp_results
```

#### Create and Test Sample Dataset
```bash
python create_sample_yfp_dataset.py --num_subjects 10
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/YFP_sample/yfp_optical_flow_dataset.csv \
    --data_root ./datasets/YFP_sample \
    --epochs 20
```

### 📈 Performance Expectations

For well-prepared datasets, expect:
- **Accuracy**: > 80%
- **F1 Score**: > 75%
- **AUC-ROC**: > 85%
- **Sensitivity**: > 80%
- **Specificity**: > 80%

### 🔬 Technical Details

#### Model Architecture
- Based on HTNet (Hierarchical Transformer Network)
- Modified for binary classification
- Default configuration:
  - Image size: 28×28
  - Patch size: 7×7
  - Transformer dimension: 256
  - Attention heads: 3
  - Hierarchical levels: 3
  - Block repeats: [2, 2, 10]

#### Dataset Format

**Raw Images CSV:**
```csv
subject_id,image_path,label
subject_001,subject_001/img_001.jpg,0
subject_002,subject_002/img_001.jpg,1
```

**Optical Flow CSV:**
```csv
subject_id,onset_frame,apex_frame,label
subject_001,subject_001/onset.jpg,subject_001/apex.jpg,0
subject_002,subject_002/onset.jpg,subject_002/apex.jpg,1
```

### 🐛 Bug Fixes

None (initial implementation)

### ⚠️ Breaking Changes

None - This is an additive feature that doesn't affect existing functionality

### 🔄 Backward Compatibility

- All original files remain unchanged
- Original micro-expression recognition functionality preserved
- New YFP functionality is completely separate
- Shared components (Model.py) work with both systems

### 📝 Notes

- The implementation uses Python 3.7+ features
- GPU acceleration recommended for faster training
- Minimum 4GB GPU memory recommended for default settings
- CPU training supported but slower
- LOSO cross-validation may take several hours depending on dataset size

### 🎓 Citation

If using this implementation, please cite the original HTNet paper:

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

### 🔮 Future Enhancements

Planned improvements:
1. Multi-grade classification (House-Brackmann I-VI)
2. Ensemble methods for improved accuracy
3. Attention visualization for interpretability
4. Real-time video processing
5. Data augmentation strategies
6. Transfer learning from larger datasets
7. Multi-modal feature fusion

### 🙏 Acknowledgments

- Original HTNet architecture by Wang et al.
- MTCNN face detection (facenet-pytorch)
- scikit-learn metrics implementation
- OpenCV optical flow algorithms

---

## Previous Versions

### Version 1.0.0 - Original HTNet Implementation
- Micro-expression recognition on SAMM, SMIC, CASME II/III
- Three-class classification (positive, negative, surprise)
- HTNet architecture implementation
- Confusion matrix visualization
