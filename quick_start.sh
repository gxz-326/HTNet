#!/bin/bash

# Quick Start Script for Facial Palsy Recognition using HTNet
# This script demonstrates the complete workflow from dataset preparation to evaluation

echo "=========================================="
echo "HTNet for Facial Palsy Recognition"
echo "Quick Start Guide"
echo "=========================================="
echo ""

# Step 1: Dataset Preparation
echo "Step 1: Preparing FNP Dataset"
echo "------------------------------------------"
echo "Organizing dataset structure..."

# Example: Prepare FNP dataset
python prepare_dataset.py \
    --dataset_type FNP \
    --data_root ./datasets/facial_palsy/FNP \
    --output_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --split_ratio 0.7 0.15 0.15

echo ""
echo "Dataset preparation completed!"
echo ""

# Step 2: Training
echo "Step 2: Training HTNet Model"
echo "------------------------------------------"
echo "Starting training with default parameters..."
echo ""

python train_facial_palsy.py \
    --data_root ./datasets/facial_palsy/FNP \
    --train_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --val_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6 \
    --image_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --save_dir ./checkpoints/fnp \
    --log_dir ./logs/fnp

echo ""
echo "Training completed!"
echo ""

# Step 3: Evaluation
echo "Step 3: Evaluating Model"
echo "------------------------------------------"
echo "Running evaluation on test set..."
echo ""

python evaluate_facial_palsy.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --data_root ./datasets/facial_palsy/FNP \
    --test_csv ./datasets/facial_palsy/fnp_annotation.csv \
    --dataset_type FNP \
    --num_classes 6 \
    --batch_size 32 \
    --output_dir ./evaluation_results

echo ""
echo "Evaluation completed!"
echo ""

# Step 4: Demo Inference
echo "Step 4: Running Demo Inference"
echo "------------------------------------------"
echo "Testing on sample images..."
echo ""

python demo_inference.py \
    --model_path ./checkpoints/fnp/best_model.pth \
    --image_dir ./datasets/facial_palsy/test_samples \
    --output_csv ./predictions.csv \
    --num_classes 6

echo ""
echo "=========================================="
echo "Quick start completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Model checkpoint: ./checkpoints/fnp/best_model.pth"
echo "  - Training logs: ./logs/fnp/"
echo "  - Evaluation results: ./evaluation_results/"
echo "  - Predictions: ./predictions.csv"
echo ""
echo "For more options, see README_FACIAL_PALSY.md"
echo ""
