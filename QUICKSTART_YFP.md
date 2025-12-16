# Quick Start Guide - YFP Facial Palsy Detection

This guide will help you quickly set up and run facial palsy detection on the YFP dataset.

## Step 1: Prepare Your Dataset

### Option A: Using Raw Images

1. Create a directory structure:
```
datasets/
└── YFP/
    ├── subject_001/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── subject_002/
    │   └── ...
    └── ...
```

2. Create a CSV file (`datasets/yfp_dataset.csv`):
```csv
subject_id,image_path,label
subject_001,subject_001/image_001.jpg,0
subject_001,subject_001/image_002.jpg,0
subject_002,subject_002/image_001.jpg,1
```

Where:
- `label = 0` → Normal (no facial palsy)
- `label = 1` → Facial Palsy

### Option B: Using Optical Flow (Recommended)

1. Create the same directory structure as above

2. Create a CSV file (`datasets/yfp_optical_flow_dataset.csv`):
```csv
subject_id,onset_frame,apex_frame,label
subject_001,subject_001/neutral_001.jpg,subject_001/peak_001.jpg,0
subject_002,subject_002/neutral_001.jpg,subject_002/peak_001.jpg,1
```

Where:
- `onset_frame` → Neutral/rest facial expression
- `apex_frame` → Peak/maximum facial expression
- `label` → 0 for normal, 1 for palsy

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Train the Model

### For Raw Images:
```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow false \
    --epochs 100
```

### For Optical Flow (Recommended):
```bash
python train_yfp_palsy_detection.py \
    --train true \
    --csv_file ./datasets/yfp_optical_flow_dataset.csv \
    --data_root ./datasets/YFP \
    --use_optical_flow true \
    --epochs 100
```

## Step 4: View Results

After training, check the `yfp_results/` directory for:

1. **Overall Performance**:
   - `overall_metrics.txt` - Detailed metrics
   - `confusion_matrix_overall.png` - Confusion matrix
   - `roc_curve_overall.png` - ROC curve
   - `per_subject_summary.csv` - Per-subject results

2. **Per-Subject Results**:
   - `confusion_matrix_subject_XXX.png`
   - `roc_curve_subject_XXX.png`

## Step 5: Evaluate Pre-trained Models

To evaluate without training:

```bash
python train_yfp_palsy_detection.py \
    --train false \
    --csv_file ./datasets/yfp_dataset.csv \
    --data_root ./datasets/YFP \
    --weights_dir ./yfp_weights
```

## Understanding the Results

### Key Metrics to Look At:

1. **Accuracy**: Overall correctness (should be > 0.80 for good performance)
2. **Sensitivity (Recall)**: How well it detects actual palsy cases (higher is better)
3. **Specificity**: How well it identifies normal cases (higher is better)
4. **F1 Score**: Balance between precision and recall (should be > 0.75)
5. **AUC-ROC**: Overall discriminative ability (should be > 0.85)

### Example Good Results:
```
Accuracy:           0.8750
Balanced Accuracy:  0.8667
Precision:          0.8500
Recall/Sensitivity: 0.8947
Specificity:        0.8387
F1 Score:           0.8718
AUC-ROC:            0.9200
```

## Testing the Metrics System

Test the evaluation metrics with sample data:

```bash
python test_metrics.py
```

This will generate:
- `test_confusion_matrix.png`
- `test_roc_curve.png`
- `test_metrics_output.txt`

## Common Parameters

### Adjust for Your Hardware:

**If you have limited GPU memory:**
```bash
--batch_size 16 \
--image_size 24
```

**If you have powerful GPU:**
```bash
--batch_size 64 \
--image_size 32 \
--dim 384
```

### Adjust for Your Dataset Size:

**Small dataset (< 500 samples):**
```bash
--epochs 150 \
--learning_rate 0.00003
```

**Large dataset (> 2000 samples):**
```bash
--epochs 50 \
--learning_rate 0.0001
```

## Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce `--batch_size` to 16 or 8

### Problem: Low accuracy (< 60%)
**Solutions**:
1. Check if labels are correct in CSV file
2. Ensure images contain visible faces
3. Try using optical flow features
4. Increase training epochs
5. Check class balance in dataset

### Problem: Face detection failures
**Solution**: Images are automatically resized if face detection fails. Ensure images are clear and faces are visible.

### Problem: Training is very slow
**Solutions**:
1. Reduce `--image_size` to 24
2. Reduce `--num_workers` if CPU is bottleneck
3. Use GPU if available

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and model dimensions
2. **Data Augmentation**: Add more diverse training examples
3. **Ensemble Models**: Combine multiple models for better performance
4. **Fine-tuning**: Adjust based on specific characteristics of your dataset

## Getting Help

- Check `YFP_README.md` for detailed documentation
- Review example outputs in `yfp_results/`
- Examine the generated metrics files

## Citation

If you use this system, please cite the HTNet paper:
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
