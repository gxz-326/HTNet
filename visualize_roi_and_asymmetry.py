import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F

from Model import HTNetEnhanced
from facial_palsy_dataset import FacialPalsyDataset, CKPlusFacialPalsyDataset


def load_model(model_path, config):
    """Load the trained HTNetEnhanced model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HTNetEnhanced(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        dim=config['dim'],
        heads=config['heads'],
        num_hierarchies=config['num_hierarchies'],
        block_repeats=config['block_repeats'],
        num_classes=config['num_classes'],
        use_diagonal_attn=True,
        use_roi=True
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device


def visualize_roi_and_asymmetry(model, image_tensor, original_image, device, save_path=None):
    """
    Visualize ROI regions and facial asymmetry detection
    
    Args:
        model: HTNetEnhanced model
        image_tensor: Input tensor [1, C, H, W]
        original_image: Original image for display [H, W, C]
        device: torch device
        save_path: Path to save the visualization
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output, roi_maps_list = model(image_tensor, return_roi_maps=True)
        
        pred_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title(f'Original Image\nPredicted Grade: {pred_class+1}\nConfidence: {confidence:.2%}', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    h, w = original_image.shape[:2]
    mid = w // 2
    
    left_side = original_image[:, :mid]
    right_side = original_image[:, mid:]
    right_flipped = cv2.flip(right_side, 1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(left_side)
    ax2.set_title('Left Side', fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(right_side)
    ax3.set_title('Right Side', fontsize=12)
    ax3.axis('off')
    
    if left_side.shape == right_flipped.shape:
        diff = np.abs(left_side.astype(float) - right_flipped.astype(float))
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
        
        ax4 = fig.add_subplot(gs[0, 3])
        im = ax4.imshow(diff_normalized, cmap='hot')
        ax4.set_title('Asymmetry Heatmap\n(Left vs Right-Flipped)', fontsize=12)
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046)
    
    if roi_maps_list and len(roi_maps_list) > 0:
        roi_maps = roi_maps_list[0].cpu().numpy()[0]
        num_roi = min(roi_maps.shape[0], 5)
        
        roi_titles = ['ROI 1: Forehead/Eyebrows', 'ROI 2: Left Eye', 'ROI 3: Nose', 
                      'ROI 4: Right Eye', 'ROI 5: Mouth']
        
        for i in range(num_roi):
            row = 1 + i // 4
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            
            roi_map = roi_maps[i]
            roi_resized = cv2.resize(roi_map, (w, h))
            
            overlay = original_image.copy()
            heatmap = cv2.applyColorMap((roi_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
            
            ax.imshow(overlay)
            ax.set_title(roi_titles[i] if i < len(roi_titles) else f'ROI {i+1}', fontsize=10)
            ax.axis('off')
        
        ax_combined = fig.add_subplot(gs[2, :2])
        roi_combined = np.sum(roi_maps, axis=0)
        roi_combined = roi_combined / roi_combined.max() if roi_combined.max() > 0 else roi_combined
        roi_combined_resized = cv2.resize(roi_combined, (w, h))
        
        overlay_combined = original_image.copy()
        heatmap_combined = cv2.applyColorMap((roi_combined_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_combined = cv2.cvtColor(heatmap_combined, cv2.COLOR_BGR2RGB)
        overlay_combined = cv2.addWeighted(overlay_combined, 0.5, heatmap_combined, 0.5, 0)
        
        ax_combined.imshow(overlay_combined)
        ax_combined.set_title('Combined ROI Attention Map', fontsize=12, fontweight='bold')
        ax_combined.axis('off')
        
        ax_mask = fig.add_subplot(gs[2, 2:])
        roi_mask = (roi_combined_resized > 0.3).astype(np.uint8) * 255
        masked_image = original_image.copy()
        for c in range(3):
            masked_image[:, :, c] = masked_image[:, :, c] * (roi_mask / 255.0)
        
        ax_mask.imshow(masked_image)
        ax_mask.set_title('Focused Facial Regions\n(Background Suppressed)', fontsize=12, fontweight='bold')
        ax_mask.axis('off')
    
    plt.suptitle('HTNetEnhanced: Diagonal Micro-Attention & ROI Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Visualization saved to: {save_path}')
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize ROI and Asymmetry Detection')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./datasets/facial_palsy',
                       help='Root directory of the dataset')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='CSV file for test data')
    parser.add_argument('--dataset_type', type=str, default='FNP', choices=['FNP', 'CK+'],
                       help='Dataset type')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations/roi_asymmetry',
                       help='Directory to save visualizations')
    
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_hierarchies', type=int, default=3)
    parser.add_argument('--block_repeats', type=int, nargs='+', default=[2, 2, 10])
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = {
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'num_classes': args.num_classes,
        'dim': args.dim,
        'heads': args.heads,
        'num_hierarchies': args.num_hierarchies,
        'block_repeats': tuple(args.block_repeats)
    }
    
    print('Loading model...')
    model, device = load_model(args.model_path, config)
    print('Model loaded successfully!')
    
    print(f'\nLoading {args.dataset_type} dataset...')
    if args.dataset_type == 'FNP':
        dataset = FacialPalsyDataset(
            data_root=args.data_root,
            csv_file=args.test_csv,
            dataset_type='FNP',
            image_size=args.image_size,
            split='test'
        )
    else:
        dataset = CKPlusFacialPalsyDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            split='test'
        )
    
    print(f'Total samples: {len(dataset)}')
    
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f'\nGenerating visualizations for {num_samples} samples...')
    
    for idx, sample_idx in enumerate(indices):
        image_tensor, label = dataset[sample_idx]
        
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        
        image_batch = image_tensor.unsqueeze(0)
        
        save_path = os.path.join(args.output_dir, f'sample_{idx+1}_grade_{label+1}.png')
        
        print(f'Processing sample {idx+1}/{num_samples} (True Grade: {label+1})...')
        visualize_roi_and_asymmetry(model, image_batch, image_np, device, save_path)
    
    print(f'\n✓ All visualizations saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
