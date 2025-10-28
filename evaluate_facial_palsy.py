import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from Model import HTNet
from facial_palsy_dataset import FacialPalsyDataset, CKPlusFacialPalsyDataset


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Facial Palsy Grading')
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def evaluate_model(config):
    """Evaluate the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    print(f'Loading {config.dataset_type} dataset...')
    if config.dataset_type == 'FNP':
        test_dataset = FacialPalsyDataset(
            data_root=config.data_root,
            csv_file=config.test_csv,
            dataset_type='FNP',
            image_size=config.image_size,
            split='test'
        )
    elif config.dataset_type == 'CK+':
        test_dataset = CKPlusFacialPalsyDataset(
            data_root=config.data_root,
            image_size=config.image_size,
            split='test'
        )
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f'Test samples: {len(test_dataset)}\n')
    
    print('Loading model...')
    model = HTNet(
        image_size=config.image_size,
        patch_size=config.patch_size,
        dim=config.dim,
        heads=config.heads,
        num_hierarchies=config.num_hierarchies,
        block_repeats=config.block_repeats,
        num_classes=config.num_classes,
        use_micro_attention=config.use_micro_attention,
        use_roi_module=config.use_roi_module,
        num_roi_regions=config.num_roi_regions
    )
    
    checkpoint = torch.load(config.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print('\nEvaluating model...')
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print('\n' + '='*60)
    print('Evaluation Results')
    print('='*60)
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'F1 Score (Macro): {f1_macro:.4f}')
    print(f'F1 Score (Weighted): {f1_weighted:.4f}')
    
    per_class_acc = []
    for i in range(config.num_classes):
        mask = all_labels == i
        if np.sum(mask) > 0:
            class_acc = np.sum(all_preds[mask] == i) / np.sum(mask)
            per_class_acc.append(class_acc)
            print(f'Grade {i} Accuracy: {class_acc:.4f} (n={np.sum(mask)})')
        else:
            per_class_acc.append(0.0)
            print(f'Grade {i} Accuracy: N/A (no samples)')
    
    mean_class_acc = np.mean([acc for acc in per_class_acc if acc > 0])
    print(f'\nMean Class Accuracy: {mean_class_acc:.4f}')
    print('='*60 + '\n')
    
    class_names = [f'Grade {i}' for i in range(config.num_classes)]
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, 
                                target_names=class_names, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix:')
    print(cm)
    
    os.makedirs(config.output_dir, exist_ok=True)
    cm_path = os.path.join(config.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    results_path = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('Facial Palsy Grading - Evaluation Results\n')
        f.write('='*60 + '\n\n')
        f.write(f'Dataset: {config.dataset_type}\n')
        f.write(f'Model: {config.model_path}\n')
        f.write(f'Test samples: {len(test_dataset)}\n\n')
        f.write(f'Overall Accuracy: {accuracy:.4f}\n')
        f.write(f'F1 Score (Macro): {f1_macro:.4f}\n')
        f.write(f'F1 Score (Weighted): {f1_weighted:.4f}\n')
        f.write(f'Mean Class Accuracy: {mean_class_acc:.4f}\n\n')
        
        for i, acc in enumerate(per_class_acc):
            n_samples = np.sum(all_labels == i)
            f.write(f'Grade {i} Accuracy: {acc:.4f} (n={n_samples})\n')
        
        f.write('\n' + '='*60 + '\n')
        f.write('Classification Report:\n')
        f.write('='*60 + '\n')
        f.write(classification_report(all_labels, all_preds, 
                                     target_names=class_names, digits=4))
        
        f.write('\n' + '='*60 + '\n')
        f.write('Confusion Matrix:\n')
        f.write('='*60 + '\n')
        f.write(str(cm))
    
    print(f'\nResults saved to {results_path}')
    
    predictions_path = os.path.join(config.output_dir, 'predictions.npz')
    np.savez(predictions_path, 
             predictions=all_preds,
             labels=all_labels,
             probabilities=all_probs)
    print(f'Predictions saved to {predictions_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate HTNet for Facial Palsy Recognition')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./datasets/facial_palsy',
                       help='Root directory of the dataset')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='CSV file for test data (optional)')
    parser.add_argument('--dataset_type', type=str, default='FNP', choices=['FNP', 'CK+'],
                       help='Dataset type: FNP or CK+')
    
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=7,
                       help='Patch size for HTNet')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of facial palsy grades')
    
    parser.add_argument('--dim', type=int, default=256,
                       help='Dimension of the model')
    parser.add_argument('--heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--num_hierarchies', type=int, default=3,
                       help='Number of hierarchies in HTNet')
    parser.add_argument('--block_repeats', type=int, nargs='+', default=[2, 2, 10],
                       help='Number of transformer blocks at each hierarchy')
    
    parser.add_argument('--use_micro_attention', action='store_true',
                       help='Enable Diagonal Micro-Attention for facial asymmetry detection')
    parser.add_argument('--use_roi_module', action='store_true',
                       help='Enable ROI module for facial region focusing')
    parser.add_argument('--num_roi_regions', type=int, default=5,
                       help='Number of ROI regions to detect')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    config = parser.parse_args()
    
    config.block_repeats = tuple(config.block_repeats)
    
    print('\n' + '='*60)
    print('HTNet for Facial Palsy Recognition - Evaluation')
    print('='*60)
    print('\nConfiguration:')
    for key, value in vars(config).items():
        print(f'  {key}: {value}')
    print('='*60 + '\n')
    
    evaluate_model(config)


if __name__ == '__main__':
    main()
