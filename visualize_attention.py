import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Model import HTNet
from facial_palsy_dataset import FacialPalsyDataset


class AttentionVisualizer:
    """
    Visualize attention maps from HTNet for facial palsy assessment
    """
    def __init__(self, model_path, num_classes=6, image_size=224, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        self.model = HTNet(
            image_size=image_size,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract attention maps"""
        def get_attention_hook(name):
            def hook(module, input, output):
                self.attention_maps.append({
                    'name': name,
                    'output': output.detach()
                })
            return hook
        
        for name, module in self.model.named_modules():
            if 'Attention' in module.__class__.__name__:
                module.register_forward_hook(get_attention_hook(name))
    
    def extract_attention(self, image_path):
        """
        Extract attention maps for an image
        
        Args:
            image_path: Path to input image
        
        Returns:
            attention_maps: List of attention maps
            prediction: Model prediction
            confidence: Prediction confidence
        """
        self.attention_maps = []
        
        dataset = FacialPalsyDataset(
            data_root=os.path.dirname(image_path),
            image_size=self.image_size,
            split='test'
        )
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return self.attention_maps, predicted.item(), confidence.item(), image
    
    def visualize_attention_map(self, attention_map, original_image, save_path=None):
        """
        Visualize a single attention map overlaid on original image
        
        Args:
            attention_map: Attention tensor
            original_image: Original image (H, W, 3)
            save_path: Path to save visualization
        """
        if len(attention_map.shape) == 4:
            attention_map = attention_map[0].mean(0)
        
        attention_map = attention_map.cpu().numpy()
        
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        attention_map_resized = cv2.resize(
            attention_map,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        cmap = plt.cm.jet
        heatmap = cmap(attention_map_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Visualization saved to {save_path}')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_hierarchical_attention(self, image_path, output_dir):
        """
        Visualize attention at different hierarchical levels
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        attention_maps, prediction, confidence, original_image = self.extract_attention(image_path)
        
        print(f'Prediction: Grade {prediction} (Confidence: {confidence:.2%})')
        print(f'Found {len(attention_maps)} attention layers')
        
        for i, attention_data in enumerate(attention_maps):
            name = attention_data['name']
            attention = attention_data['output']
            
            save_path = os.path.join(output_dir, f'attention_layer_{i}_{name.replace(".", "_")}.png')
            self.visualize_attention_map(attention, original_image, save_path)
        
        self._create_summary_visualization(attention_maps, original_image, prediction, confidence, output_dir)
    
    def _create_summary_visualization(self, attention_maps, original_image, prediction, confidence, output_dir):
        """
        Create a summary visualization showing all attention levels
        """
        n_maps = min(len(attention_maps), 6)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Original\nGrade {prediction} ({confidence:.2%})')
        axes[0, 0].axis('off')
        
        for i in range(n_maps):
            row = (i + 1) // 4
            col = (i + 1) % 4
            
            attention = attention_maps[i]['output']
            if len(attention.shape) == 4:
                attention = attention[0].mean(0)
            
            attention_np = attention.cpu().numpy()
            attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
            
            axes[row, col].imshow(attention_np, cmap='jet')
            axes[row, col].set_title(f'Layer {i+1}')
            axes[row, col].axis('off')
        
        for i in range(n_maps + 1, 8):
            row = i // 4
            col = i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, 'attention_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f'Summary visualization saved to {summary_path}')
        plt.close()


def visualize_facial_regions(image_path, output_path=None):
    """
    Visualize the facial regions used for analysis
    """
    from facenet_pytorch import MTCNN
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mtcnn = MTCNN(margin=0, select_largest=True, post_process=False,
                  device='cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, landmarks = mtcnn.detect(image_rgb, landmarks=True)
    
    if landmarks is None:
        print("No face detected!")
        return
    
    landmarks = landmarks[0].astype(int)
    
    region_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    result = image.copy()
    
    for i, (landmark, name, color) in enumerate(zip(landmarks, region_names, colors)):
        x, y = landmark
        
        cv2.circle(result, (x, y), 5, color, -1)
        
        cv2.rectangle(result, (x-14, y-14), (x+14, y+14), color, 2)
        
        cv2.putText(result, name, (x-20, y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f'Facial regions visualization saved to {output_path}')
    else:
        cv2.imshow('Facial Regions', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize HTNet Attention for Facial Palsy')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./attention_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of facial palsy grades')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--visualize_regions', action='store_true',
                       help='Also visualize facial regions')
    
    args = parser.parse_args()
    
    print('\n' + '='*60)
    print('HTNet Attention Visualization for Facial Palsy')
    print('='*60 + '\n')
    
    visualizer = AttentionVisualizer(
        model_path=args.model_path,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=args.device
    )
    
    print(f'Processing image: {args.image_path}')
    visualizer.visualize_hierarchical_attention(args.image_path, args.output_dir)
    
    if args.visualize_regions:
        regions_path = os.path.join(args.output_dir, 'facial_regions.png')
        print('\nVisualizing facial regions...')
        visualize_facial_regions(args.image_path, regions_path)
    
    print('\n' + '='*60)
    print('Visualization completed!')
    print(f'Results saved in: {args.output_dir}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
