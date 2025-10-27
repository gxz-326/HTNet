# HTNet for Facial Palsy Recognition and Grading

This is an adaptation of HTNet (Hierarchical Transformer Network) for facial palsy (facial nerve paralysis) recognition and grading tasks using FNP and CK+ datasets.

## Overview

Facial palsy is a condition characterized by facial muscle weakness or paralysis, typically affecting one side of the face. This project adapts the HTNet architecture, originally designed for micro-expression recognition, to automatically assess facial palsy severity using the House-Brackmann grading scale.

### Key Adaptations

1. **Task**: Micro-expression recognition → Facial palsy grading
2. **Datasets**: CASME/SMIC/SAMM → FNP (Facial Nerve Palsy) + CK+
3. **Classes**: 3 emotions → 6 grades (House-Brackmann scale: Grade I-VI)
4. **Focus**: Temporal micro-movements → Facial symmetry and movement capability

### House-Brackmann Grading Scale

- **Grade I**: Normal facial function
- **Grade II**: Slight weakness on close inspection
- **Grade III**: Obvious weakness but not disfiguring
- **Grade IV**: Obviously disfiguring weakness
- **Grade V**: Only barely perceptible motion
- **Grade VI**: No movement

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Additional requirements for facial palsy module:
```bash
pip install scikit-learn matplotlib seaborn
```

### Python Dependencies

- Python >= 3.7
- PyTorch >= 1.8.0
- torchvision
- opencv-python
- pandas
- numpy
- facenet-pytorch (for facial landmark detection)
- scikit-learn
- matplotlib
- seaborn

## Dataset Preparation

### 1. FNP Dataset

Organize your FNP dataset in the following structure:

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

Then prepare the dataset:

```bash
python prepare_dataset.py \
    --dataset_type FNP \
    --data_root ./datasets/facial_palsy/FNP \
    --output_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --split_ratio 0.7 0.15 0.15
```

### 2. CK+ Dataset

For CK+ dataset (originally for facial expressions):

```bash
python prepare_dataset.py \
    --dataset_type CK+ \
    --data_root ./path/to/ckplus/cohn-kanade-images \
    --output_root ./datasets/facial_palsy/CK_prepared \
    --output_csv ./datasets/facial_palsy/ckplus_annotation.csv \
    --grade_mapping ./ckplus_grade_mapping.csv
```

**Note**: Since CK+ is originally for facial expressions, you need to provide a grade mapping file or manually annotate the images for facial palsy grades.

### 3. Convert 3-class to 6-class

If you have 3-class annotations (Normal/Mild/Severe), convert them:

```bash
python prepare_dataset.py \
    --dataset_type convert \
    --data_root ./existing_3class_annotation.csv \
    --output_csv ./converted_6class_annotation.csv
```

## Training

### Basic Training

Train on FNP dataset:

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

### Training on CK+ Dataset

```bash
python train_facial_palsy.py \
    --data_root ./datasets/facial_palsy/CK_prepared \
    --train_csv ./datasets/facial_palsy/ckplus_annotation.csv \
    --val_csv ./datasets/facial_palsy/ckplus_annotation.csv \
    --dataset_type CK+ \
    --num_classes 6 \
    --batch_size 32 \
    --epochs 100 \
    --save_dir ./checkpoints/ckplus
```

### Transfer Learning from Micro-expression Model

Use pre-trained weights from the original HTNet:

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

### Advanced Training Options

```bash
python train_facial_palsy.py \
    --data_root ./datasets/facial_palsy/FNP \
    --dataset_type FNP \
    --num_classes 6 \
    --image_size 224 \
    --patch_size 7 \
    --dim 256 \
    --heads 3 \
    --num_hierarchies 3 \
    --block_repeats 2 2 10 \
    --batch_size 32 \
    --epochs 200 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --save_interval 10
```

## Evaluation

Evaluate trained model on test set:

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

### Evaluation Outputs

The evaluation script generates:

1. **Confusion matrix** (`confusion_matrix.png`): Visual representation of predictions vs ground truth
2. **Evaluation report** (`evaluation_results.txt`): Detailed metrics including:
   - Overall accuracy
   - F1 scores (macro and weighted)
   - Per-class accuracy
   - Classification report
3. **Predictions** (`predictions.npz`): Raw predictions and probabilities for further analysis

## Model Architecture

The HTNet architecture is well-suited for facial palsy assessment because:

1. **Hierarchical Processing**: Captures both local facial features (eyes, mouth) and global facial symmetry
2. **Multi-scale Analysis**: Processes facial regions at different scales
3. **Attention Mechanism**: Focuses on diagnostically relevant facial areas
4. **Region-based Approach**: Analyzes key facial regions independently (eyes, mouth) and their interactions

### Key Facial Regions for Palsy Assessment

- **Forehead/Eyebrows**: Ability to raise eyebrows
- **Eyes**: Eye closure strength and symmetry
- **Nose**: Nasolabial fold depth
- **Mouth**: Smile symmetry and mouth corner movement

## Customization

### Modify for 3-class Problem

If you want to use 3 classes (Normal/Mild/Severe) instead of 6:

```bash
python train_facial_palsy.py \
    --num_classes 3 \
    --other_args...
```

Update the Fusionmodel in `Model.py` accordingly if using region fusion.

### Change Image Size

For different input resolutions:

```bash
python train_facial_palsy.py \
    --image_size 112 \
    --patch_size 7 \
    --other_args...
```

Make sure `image_size` is divisible by `patch_size`.

### Adjust Model Capacity

For smaller models (faster training, less memory):

```bash
python train_facial_palsy.py \
    --dim 128 \
    --heads 2 \
    --num_hierarchies 2 \
    --block_repeats 2 2 \
    --other_args...
```

For larger models (potentially better accuracy):

```bash
python train_facial_palsy.py \
    --dim 384 \
    --heads 4 \
    --num_hierarchies 3 \
    --block_repeats 3 3 12 \
    --other_args...
```

## Expected Results

Performance will vary based on dataset quality and size. Typical expectations:

- **Overall Accuracy**: 60-85%
- **F1 Score (Macro)**: 55-80%
- **Mean Class Accuracy**: 58-82%

Higher grades (more severe palsy) are generally easier to classify than subtle differences between normal and Grade II.

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or image size

```bash
python train_facial_palsy.py --batch_size 16 --image_size 112 ...
```

### Issue: Poor Performance on Specific Grades

**Solution**: Check class balance and consider class weights

```python
# In train_facial_palsy.py, modify:
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Issue: Face Detection Fails

**Solution**: The code uses default landmarks if detection fails, but you can adjust the default positions in `facial_palsy_dataset.py`

### Issue: Overfitting

**Solution**: 
- Increase dropout
- Add data augmentation
- Reduce model capacity
- Use more training data

## Citation

If you use this code for facial palsy research, please cite the original HTNet paper:

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

Please refer to the original HTNet repository for license information.

## Contact

For issues specific to the facial palsy adaptation, please open an issue in the repository.

## Future Improvements

Potential enhancements:

1. **Data Augmentation**: Add facial transformations, lighting variations
2. **Multi-task Learning**: Simultaneously predict grade and affected side
3. **Temporal Analysis**: Use video sequences instead of single frames
4. **Attention Visualization**: Visualize which facial regions contribute to predictions
5. **Ensemble Methods**: Combine multiple models for better accuracy
6. **Cross-dataset Validation**: Validate on multiple datasets for generalization
