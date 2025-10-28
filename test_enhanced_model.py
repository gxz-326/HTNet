"""
Test script for HTNetEnhanced model
Tests the diagonal micro-attention and ROI modules
"""

import torch
import numpy as np
from Model import HTNetEnhanced, DiagonalMicroAttention, ROIAttentionModule


def test_diagonal_micro_attention():
    """Test the DiagonalMicroAttention module"""
    print("="*60)
    print("Testing DiagonalMicroAttention Module")
    print("="*60)
    
    batch_size = 2
    channels = 128
    height, width = 16, 16
    
    module = DiagonalMicroAttention(dim=channels, heads=4, dropout=0.1)
    
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    output = module(x)
    
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    
    left_side = x[:, :, :, :width//2]
    right_side = x[:, :, :, width//2:]
    
    print(f"\nLeft side shape: {left_side.shape}")
    print(f"Right side shape: {right_side.shape}")
    
    asymmetry_weight = module.detect_left_right_asymmetry(x)
    print(f"Asymmetry weight shape: {asymmetry_weight.shape}")
    print(f"Asymmetry weight range: [{asymmetry_weight.min():.4f}, {asymmetry_weight.max():.4f}]")
    
    print("✓ DiagonalMicroAttention test passed!\n")
    return True


def test_roi_attention_module():
    """Test the ROIAttentionModule"""
    print("="*60)
    print("Testing ROIAttentionModule")
    print("="*60)
    
    batch_size = 2
    channels = 256
    height, width = 32, 32
    num_roi_regions = 5
    
    module = ROIAttentionModule(dim=channels, num_roi_regions=num_roi_regions)
    
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    output, roi_maps = module(x)
    
    print(f"Output shape: {output.shape}")
    print(f"ROI maps shape: {roi_maps.shape}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    assert roi_maps.shape == (batch_size, num_roi_regions, height, width), "ROI maps shape mismatch!"
    
    print(f"\nNumber of ROI regions: {num_roi_regions}")
    for i in range(num_roi_regions):
        roi_mean = roi_maps[:, i].mean().item()
        roi_max = roi_maps[:, i].max().item()
        print(f"ROI {i+1} - Mean: {roi_mean:.4f}, Max: {roi_max:.4f}")
    
    print("✓ ROIAttentionModule test passed!\n")
    return True


def test_htnet_enhanced():
    """Test the full HTNetEnhanced model"""
    print("="*60)
    print("Testing HTNetEnhanced Model")
    print("="*60)
    
    batch_size = 2
    image_size = 224
    num_classes = 6
    
    model = HTNetEnhanced(
        image_size=image_size,
        patch_size=7,
        num_classes=num_classes,
        dim=128,
        heads=4,
        num_hierarchies=3,
        block_repeats=(2, 2, 4),
        use_diagonal_attn=True,
        use_roi=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    print(f"\nInput shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape (without ROI maps): {output.shape}")
    assert output.shape == (batch_size, num_classes), "Output shape mismatch!"
    
    output, roi_maps = model(x, return_roi_maps=True)
    print(f"Output shape (with ROI maps): {output.shape}")
    print(f"Number of ROI map sets: {len(roi_maps) if roi_maps else 0}")
    
    if roi_maps and len(roi_maps) > 0:
        print(f"First ROI map set shape: {roi_maps[0].shape}")
    
    print("\n✓ HTNetEnhanced test passed!\n")
    return True


def test_model_modes():
    """Test model with different module combinations"""
    print("="*60)
    print("Testing Different Module Combinations")
    print("="*60)
    
    batch_size = 2
    image_size = 224
    num_classes = 6
    
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    configs = [
        {"use_diagonal_attn": True, "use_roi": True, "name": "Both modules"},
        {"use_diagonal_attn": True, "use_roi": False, "name": "Diagonal attention only"},
        {"use_diagonal_attn": False, "use_roi": True, "name": "ROI only"},
        {"use_diagonal_attn": False, "use_roi": False, "name": "No enhancement"},
    ]
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        model = HTNetEnhanced(
            image_size=image_size,
            patch_size=7,
            num_classes=num_classes,
            dim=128,
            heads=4,
            num_hierarchies=3,
            block_repeats=(2, 2, 4),
            use_diagonal_attn=config["use_diagonal_attn"],
            use_roi=config["use_roi"]
        )
        
        output, roi_maps = model(x, return_roi_maps=True)
        
        print(f"  Output shape: {output.shape}")
        print(f"  ROI maps available: {roi_maps is not None and len(roi_maps) > 0}")
        
        assert output.shape == (batch_size, num_classes), f"Output shape mismatch for {config['name']}!"
    
    print("\n✓ All module combination tests passed!\n")
    return True


def test_backward_pass():
    """Test backward pass (gradient flow)"""
    print("="*60)
    print("Testing Backward Pass")
    print("="*60)
    
    batch_size = 2
    image_size = 224
    num_classes = 6
    
    model = HTNetEnhanced(
        image_size=image_size,
        patch_size=7,
        num_classes=num_classes,
        dim=128,
        heads=4,
        num_hierarchies=3,
        block_repeats=(2, 2, 4),
        use_diagonal_attn=True,
        use_roi=True
    )
    
    x = torch.randn(batch_size, 3, image_size, image_size, requires_grad=True)
    target = torch.randint(0, num_classes, (batch_size,))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    
    output = model(x)
    loss = criterion(output, target)
    
    print(f"\nLoss: {loss.item():.4f}")
    
    loss.backward()
    
    has_grad = x.grad is not None
    print(f"Input has gradients: {has_grad}")
    
    param_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Parameters with gradients: {param_with_grad}/{total_params}")
    
    assert has_grad, "No gradients computed!"
    assert param_with_grad == total_params, "Not all parameters have gradients!"
    
    print("✓ Backward pass test passed!\n")
    return True


def main():
    print("\n" + "="*60)
    print("HTNetEnhanced Model Testing Suite")
    print("="*60 + "\n")
    
    try:
        test_diagonal_micro_attention()
        test_roi_attention_module()
        test_htnet_enhanced()
        test_model_modes()
        test_backward_pass()
        
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe HTNetEnhanced model with diagonal micro-attention")
        print("and ROI modules is working correctly!")
        print("\nYou can now use it for training with:")
        print("  python train_enhanced_facial_palsy.py [args]")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
