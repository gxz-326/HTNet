# Changelog

## [Unreleased] - 2024-10-27

### Added - Facial Palsy Recognition Module

#### Core Implementation
- **facial_palsy_dataset.py**: Dataset loaders for FNP and CK+ datasets with facial landmark detection
- **train_facial_palsy.py**: Complete training pipeline with CLI, metrics, and checkpointing
- **evaluate_facial_palsy.py**: Comprehensive evaluation with confusion matrix and detailed metrics
- **demo_inference.py**: Real-time inference tool for single images and batch processing
- **prepare_dataset.py**: Dataset preparation utilities for FNP and CK+ datasets
- **data_augmentation.py**: Specialized augmentation preserving facial asymmetry
- **visualize_attention.py**: Attention map visualization and interpretation tools

#### Documentation
- **README_FACIAL_PALSY.md**: Comprehensive English documentation for facial palsy recognition
- **README_CN.md**: Complete Chinese documentation and user guide
- **MIGRATION_SUMMARY.md**: Detailed migration documentation from micro-expression to facial palsy
- **config_examples.yaml**: Configuration templates for different scenarios
- **quick_start.sh**: Automated workflow script for quick start
- **CHANGELOG.md**: This file

### Changed
- **README.md**: Added facial palsy recognition section with quick start guide
- **requirements.txt**: Added seaborn dependency, cleaned up duplicate entries

### Features
- Support for 6-class House-Brackmann grading scale
- Optional 3-class simplified grading (Normal/Mild/Severe)
- Transfer learning from micro-expression pretrained models
- Asymmetry-preserving data augmentation
- Attention visualization for model interpretability
- Comprehensive evaluation metrics (accuracy, F1, per-class metrics)
- Batch inference capabilities
- Facial region visualization
- Multiple configuration templates

### Technical Details
- Maintained HTNet hierarchical transformer architecture
- Preserved original micro-expression recognition code
- Modular design for easy dataset switching
- CLI interface with extensive configuration options
- Checkpoint management with metadata
- GPU/CPU support with automatic device selection

### Datasets Supported
- **FNP (Facial Nerve Palsy)**: Primary dataset for facial palsy grading
- **CK+ (Extended Cohn-Kanade)**: Adapted for facial palsy task
- Flexible CSV-based annotation system
- Automatic train/val/test splitting with stratification

### Model Configurations
- Standard: 224x224 input, dim=256, heads=3, hierarchies=3
- Large: 224x224 input, dim=384, heads=4, hierarchies=3 (higher capacity)
- Small: 112x112 input, dim=128, heads=2, hierarchies=2 (faster, lightweight)
- Transfer learning: Fine-tune from micro-expression weights

### Performance Expectations
- Overall Accuracy: 60-85%
- F1 Score (Macro): 55-80%
- Mean Class Accuracy: 58-82%
- Performance varies based on dataset quality and size

### Visualization Tools
- Attention heatmaps at multiple hierarchical levels
- Facial landmark and region visualization
- Confusion matrix with custom colormaps
- Prediction confidence overlay

### Use Cases
- Automated facial palsy grading
- Clinical decision support (research only)
- Facial asymmetry analysis
- Medical image analysis research
- Rehabilitation progress tracking

### Backward Compatibility
- All original micro-expression code preserved
- Can still run original training: `python main_HTNet.py --train True`
- Model.py unchanged, supports both tasks
- Original datasets remain accessible

### Known Limitations
- Requires well-organized, labeled datasets
- Not for clinical diagnosis (research tool only)
- May require class balancing for imbalanced datasets
- CK+ requires manual annotation for palsy grades
- Horizontal flipping not recommended (breaks asymmetry)

### Future Work
- Enhanced data augmentation techniques
- Multi-task learning (grade + affected side)
- Video sequence analysis for temporal patterns
- Ensemble methods for improved accuracy
- Cross-dataset validation studies
- Mobile/edge deployment optimization

---

## Previous Versions

### Original Implementation
- HTNet for micro-expression recognition
- CASME II, SMIC, SAMM datasets
- 3-class emotion classification (positive, negative, surprise)
- LOSO (Leave-One-Subject-Out) protocol
- State-of-the-art performance on micro-expression benchmarks
