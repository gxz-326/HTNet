import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

from Model import HTNet
from facial_palsy_dataset import FacialPalsyDataset, CKPlusFacialPalsyDataset


def visualize_roi_and_asymmetry(model, dataloader, device, output_dir, num_samples=5):
    """
    Visualize the ROI maps and facial asymmetry detection
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, attention_info = model(inputs, return_attention_maps=True)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                if samples_processed >= num_samples:
                    break
                
                img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                fig.suptitle(f'Sample {samples_processed + 1}: True Grade={true_label}, Predicted Grade={pred_label}', 
                           fontsize=16, fontweight='bold')
                
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Original Image', fontsize=12)
                axes[0, 0].axis('off')
                
                if attention_info['roi_mask'] is not None:
                    roi_mask = attention_info['roi_mask'][i, 0].cpu().numpy()
                    axes[0, 1].imshow(roi_mask, cmap='hot')
                    axes[0, 1].set_title('ROI Mask (Face Focus)', fontsize=12)
                    axes[0, 1].axis('off')
                    
                    overlay = img.copy()
                    roi_mask_resized = cv2.resize(roi_mask, (img.shape[1], img.shape[0]))
                    roi_colored = plt.cm.hot(roi_mask_resized)[:, :, :3]
                    overlay = 0.6 * overlay + 0.4 * roi_colored
                    axes[0, 2].imshow(overlay)
                    axes[0, 2].set_title('ROI Overlay on Image', fontsize=12)
                    axes[0, 2].axis('off')
                else:
                    axes[0, 1].text(0.5, 0.5, 'ROI Module Not Enabled', 
                                  ha='center', va='center', fontsize=12)
                    axes[0, 1].axis('off')
                    axes[0, 2].axis('off')
                
                if attention_info['roi_maps'] is not None:
                    roi_maps = attention_info['roi_maps'][i].cpu().numpy()
                    
                    axes[0, 3].imshow(np.mean(roi_maps, axis=0), cmap='viridis')
                    axes[0, 3].set_title('Average ROI Regions', fontsize=12)
                    axes[0, 3].axis('off')
                    
                    for region_idx in range(min(4, roi_maps.shape[0])):
                        row = 1
                        col = region_idx
                        axes[row, col].imshow(roi_maps[region_idx], cmap='viridis')
                        axes[row, col].set_title(f'ROI Region {region_idx + 1}', fontsize=10)
                        axes[row, col].axis('off')
                else:
                    axes[0, 3].text(0.5, 0.5, 'ROI Maps Not Available', 
                                  ha='center', va='center', fontsize=12)
                    axes[0, 3].axis('off')
                    for col in range(4):
                        axes[1, col].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(output_dir, f'sample_{samples_processed + 1}_visualization.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f'Saved visualization for sample {samples_processed + 1} to {save_path}')
                samples_processed += 1


def analyze_facial_asymmetry(model, dataloader, device, output_dir):
    """
    Analyze facial asymmetry patterns across different grades
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    grade_asymmetry_scores = {i: [] for i in range(6)}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            b, c, h, w = inputs.shape
            left_half = inputs[:, :, :, :w//2]
            right_half = inputs[:, :, :, w//2:]
            right_half_flipped = torch.flip(right_half, [3])
            
            min_width = min(left_half.shape[3], right_half_flipped.shape[3])
            left_half = left_half[:, :, :, :min_width]
            right_half_flipped = right_half_flipped[:, :, :, :min_width]
            
            asymmetry = torch.abs(left_half - right_half_flipped).mean(dim=[1, 2, 3])
            
            for i in range(len(labels)):
                grade = labels[i].item()
                asym_score = asymmetry[i].item()
                grade_asymmetry_scores[grade].append(asym_score)
    
    grades = []
    avg_asymmetry = []
    
    for grade in sorted(grade_asymmetry_scores.keys()):
        if len(grade_asymmetry_scores[grade]) > 0:
            grades.append(f'Grade {grade}')
            avg_asymmetry.append(np.mean(grade_asymmetry_scores[grade]))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(grades, avg_asymmetry, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Facial Palsy Grade', fontsize=12, fontweight='bold')
    plt.ylabel('Average Asymmetry Score', fontsize=12, fontweight='bold')
    plt.title('Facial Asymmetry Analysis by Grade', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, avg_asymmetry):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    save_path = os.path.join(output_dir, 'asymmetry_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Asymmetry analysis saved to {save_path}')
    
    report_path = os.path.join(output_dir, 'asymmetry_report.txt')
    with open(report_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('Facial Asymmetry Analysis Report\n')
        f.write('='*60 + '\n\n')
        
        for grade in sorted(grade_asymmetry_scores.keys()):
            if len(grade_asymmetry_scores[grade]) > 0:
                scores = grade_asymmetry_scores[grade]
                f.write(f'Grade {grade}:\n')
                f.write(f'  Samples: {len(scores)}\n')
                f.write(f'  Mean Asymmetry: {np.mean(scores):.4f}\n')
                f.write(f'  Std Asymmetry: {np.std(scores):.4f}\n')
                f.write(f'  Min Asymmetry: {np.min(scores):.4f}\n')
                f.write(f'  Max Asymmetry: {np.max(scores):.4f}\n\n')
    
    print(f'Asymmetry report saved to {report_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Diagonal Micro-Attention and ROI Module for Facial Palsy Detection'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./datasets/facial_palsy',
                       help='Root directory of the dataset')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='CSV file for test data')
    parser.add_argument('--dataset_type', type=str, default='FNP', choices=['FNP', 'CK+'],
                       help='Dataset type')
    
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=7,
                       help='Patch size')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of classes')
    
    parser.add_argument('--dim', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--num_hierarchies', type=int, default=3,
                       help='Number of hierarchies')
    parser.add_argument('--block_repeats', type=int, nargs='+', default=[2, 2, 10],
                       help='Block repeats')
    
    parser.add_argument('--use_micro_attention', action='store_true',
                       help='Enable Diagonal Micro-Attention')
    parser.add_argument('--use_roi_module', action='store_true',
                       help='Enable ROI module')
    parser.add_argument('--num_roi_regions', type=int, default=5,
                       help='Number of ROI regions')
    
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    config = parser.parse_args()
    config.block_repeats = tuple(config.block_repeats)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    print(f'Loading {config.dataset_type} dataset...')
    if config.dataset_type == 'FNP':
        dataset = FacialPalsyDataset(
            data_root=config.data_root,
            csv_file=config.test_csv,
            dataset_type='FNP',
            image_size=config.image_size,
            split='test'
        )
    elif config.dataset_type == 'CK+':
        dataset = CKPlusFacialPalsyDataset(
            data_root=config.data_root,
            image_size=config.image_size,
            split='test'
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f'\nGenerating visualizations...')
    visualize_roi_and_asymmetry(model, dataloader, device, config.output_dir, config.num_samples)
    
    print(f'\nAnalyzing facial asymmetry patterns...')
    analyze_facial_asymmetry(model, dataloader, device, config.output_dir)
    
    print(f'\nAll visualizations saved to {config.output_dir}')


if __name__ == '__main__':
    main()
