#!/usr/bin/env python3
"""
Test script to verify Diagonal Micro-Attention and ROI modules
"""

import torch
import torch.nn as nn
from Model import HTNet, DiagonalMicroAttention, FacialROIModule


def test_diagonal_micro_attention():
    """Test the Diagonal Micro-Attention module"""
    print("Testing Diagonal Micro-Attention Module...")
    
    batch_size = 2
    channels = 256
    height = 16
    width = 16
    heads = 4
    
    attention = DiagonalMicroAttention(dim=channels, heads=heads, dropout=0.1)
    
    x = torch.randn(batch_size, channels, height, width)
    
    output = attention(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Diagonal Micro-Attention module works correctly!\n")
    
    return True


def test_facial_roi_module():
    """Test the Facial ROI Module"""
    print("Testing Facial ROI Module...")
    
    batch_size = 2
    channels = 256
    height = 32
    width = 32
    num_roi_regions = 5
    
    roi_module = FacialROIModule(dim=channels, num_roi_regions=num_roi_regions)
    
    x = torch.randn(batch_size, channels, height, width)
    
    output, roi_maps, roi_mask = roi_module(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert roi_maps.shape == (batch_size, num_roi_regions, height, width), \
        f"ROI maps shape mismatch: {roi_maps.shape}"
    assert roi_mask.shape == (batch_size, 1, height, width), \
        f"ROI mask shape mismatch: {roi_mask.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ ROI maps shape: {roi_maps.shape}")
    print(f"✓ ROI mask shape: {roi_mask.shape}")
    print(f"✓ Facial ROI module works correctly!\n")
    
    return True


def test_htnet_with_modules():
    """Test HTNet with both modules enabled"""
    print("Testing HTNet with Diagonal Micro-Attention and ROI Module...")
    
    batch_size = 2
    image_size = 224
    patch_size = 7
    num_classes = 6
    
    # Test with both modules enabled
    model = HTNet(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 4),
        use_micro_attention=True,
        use_roi_module=True,
        num_roi_regions=5
    )
    
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    # Test normal forward pass
    output = model(x)
    assert output.shape == (batch_size, num_classes), \
        f"Output shape mismatch: {output.shape} vs ({batch_size}, {num_classes})"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Test with attention maps
    output, attention_info = model(x, return_attention_maps=True)
    assert 'roi_maps' in attention_info, "ROI maps not returned"
    assert 'roi_mask' in attention_info, "ROI mask not returned"
    
    print(f"✓ Attention info keys: {attention_info.keys()}")
    if attention_info['roi_maps'] is not None:
        print(f"✓ ROI maps shape: {attention_info['roi_maps'].shape}")
    if attention_info['roi_mask'] is not None:
        print(f"✓ ROI mask shape: {attention_info['roi_mask'].shape}")
    
    print(f"✓ HTNet with modules works correctly!\n")
    
    return True


def test_htnet_without_modules():
    """Test HTNet with modules disabled (backward compatibility)"""
    print("Testing HTNet without modules (backward compatibility)...")
    
    batch_size = 2
    image_size = 224
    patch_size = 7
    num_classes = 6
    
    model = HTNet(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 4),
        use_micro_attention=False,
        use_roi_module=False
    )
    
    x = torch.randn(batch_size, 3, image_size, image_size)
    output = model(x)
    
    assert output.shape == (batch_size, num_classes), \
        f"Output shape mismatch: {output.shape} vs ({batch_size}, {num_classes})"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Backward compatibility maintained!\n")
    
    return True


def test_parameter_count():
    """Compare parameter counts"""
    print("Comparing parameter counts...")
    
    config = {
        'image_size': 224,
        'patch_size': 7,
        'num_classes': 6,
        'dim': 256,
        'heads': 3,
        'num_hierarchies': 3,
        'block_repeats': (2, 2, 4)
    }
    
    # Model without modules
    model_basic = HTNet(**config, use_micro_attention=False, use_roi_module=False)
    params_basic = sum(p.numel() for p in model_basic.parameters())
    
    # Model with modules
    model_enhanced = HTNet(**config, use_micro_attention=True, use_roi_module=True)
    params_enhanced = sum(p.numel() for p in model_enhanced.parameters())
    
    print(f"✓ Basic model parameters: {params_basic:,}")
    print(f"✓ Enhanced model parameters: {params_enhanced:,}")
    print(f"✓ Additional parameters: {params_enhanced - params_basic:,} " +
          f"({(params_enhanced - params_basic) / params_basic * 100:.1f}% increase)")
    print()
    
    return True


def main():
    print("="*60)
    print("Testing Diagonal Micro-Attention and ROI Modules")
    print("="*60)
    print()
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")
        
        # Run tests
        test_diagonal_micro_attention()
        test_facial_roi_module()
        test_htnet_with_modules()
        test_htnet_without_modules()
        test_parameter_count()
        
        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print("="*60)
        print(f"✗ Test failed with error:")
        print(f"  {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
