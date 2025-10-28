#!/usr/bin/env python3
"""
Example script demonstrating the usage of Diagonal Micro-Attention 
and ROI modules for facial asymmetry detection.

对角微注意力和ROI模块用于面部不对称检测的示例脚本。
"""

import torch
import torch.nn as nn
from Model import HTNet
import numpy as np


def example_basic_usage():
    """
    Example 1: Basic usage with both modules enabled
    示例1：启用两个模块的基本用法
    """
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create model with asymmetry detection modules
    # 创建带有不对称检测模块的模型
    model = HTNet(
        image_size=224,
        patch_size=7,
        num_classes=6,              # House-Brackmann grades I-VI
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 10),
        use_micro_attention=True,   # Enable diagonal micro-attention
        use_roi_module=True,         # Enable ROI module
        num_roi_regions=5,           # 5 facial regions
        dropout=0.1
    )
    
    # Create dummy input (batch of 2 images)
    # 创建虚拟输入（2张图像的批次）
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    # 前向传播
    model.eval()
    with torch.no_grad():
        # Standard prediction
        # 标准预测
        output = model(images)
        predictions = torch.softmax(output, dim=1)
        predicted_grades = torch.argmax(predictions, dim=1)
        
        print(f"\nInput shape: {images.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Predicted grades: {predicted_grades.tolist()}")
        print(f"Confidence scores: {predictions[0].tolist()}")
    
    print("\n✓ Basic usage completed successfully!\n")


def example_with_visualization():
    """
    Example 2: Get attention maps for visualization
    示例2：获取注意力图用于可视化
    """
    print("="*60)
    print("Example 2: Visualization with Attention Maps")
    print("="*60)
    
    model = HTNet(
        image_size=224,
        patch_size=7,
        num_classes=6,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 10),
        use_micro_attention=True,
        use_roi_module=True,
        num_roi_regions=5
    )
    
    images = torch.randn(1, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        # Get predictions with attention maps
        # 获取带有注意力图的预测
        output, attention_info = model(images, return_attention_maps=True)
        
        predicted_grade = torch.argmax(output, dim=1).item()
        
        print(f"\nPredicted grade: {predicted_grade}")
        print(f"\nAttention info available:")
        
        if attention_info['roi_maps'] is not None:
            roi_maps = attention_info['roi_maps']
            print(f"  - ROI maps shape: {roi_maps.shape}")
            print(f"  - Number of regions: {roi_maps.shape[1]}")
            
            # Analyze each ROI region
            # 分析每个ROI区域
            for i in range(roi_maps.shape[1]):
                region_importance = roi_maps[0, i].mean().item()
                print(f"    Region {i} importance: {region_importance:.4f}")
        
        if attention_info['roi_mask'] is not None:
            roi_mask = attention_info['roi_mask']
            print(f"  - ROI mask shape: {roi_mask.shape}")
            facial_focus = roi_mask[0].mean().item()
            print(f"    Average facial focus: {facial_focus:.4f}")
    
    print("\n✓ Visualization example completed!\n")


def example_asymmetry_analysis():
    """
    Example 3: Manual asymmetry analysis
    示例3：手动不对称分析
    """
    print("="*60)
    print("Example 3: Manual Asymmetry Analysis")
    print("="*60)
    
    # Simulate a face image with asymmetry
    # 模拟一个带有不对称的面部图像
    image = torch.randn(1, 3, 224, 224)
    
    # Add artificial asymmetry (left side brighter than right)
    # 添加人工不对称（左侧比右侧更亮）
    image[:, :, :, :112] += 0.5  # Left half
    
    # Split into left and right halves
    # 分为左右两半
    left_half = image[:, :, :, :112]
    right_half = image[:, :, :, 112:]
    right_flipped = torch.flip(right_half, [3])
    
    # Compute asymmetry score
    # 计算不对称分数
    asymmetry = torch.abs(left_half - right_flipped).mean().item()
    
    print(f"\nAsymmetry score: {asymmetry:.4f}")
    print("\nInterpretation:")
    if asymmetry < 0.1:
        print("  - Low asymmetry (likely Grade I-II)")
    elif asymmetry < 0.3:
        print("  - Moderate asymmetry (likely Grade III-IV)")
    else:
        print("  - High asymmetry (likely Grade V-VI)")
    
    print("\n✓ Asymmetry analysis completed!\n")


def example_training_setup():
    """
    Example 4: Training setup with modules
    示例4：带模块的训练设置
    """
    print("="*60)
    print("Example 4: Training Setup")
    print("="*60)
    
    # Model configuration
    # 模型配置
    model = HTNet(
        image_size=224,
        patch_size=7,
        num_classes=6,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 10),
        use_micro_attention=True,
        use_roi_module=True,
        num_roi_regions=5,
        dropout=0.1
    )
    
    # Training setup
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Dummy training step
    # 虚拟训练步骤
    model.train()
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 6, (4,))
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"\nTraining configuration:")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Learning rate: {optimizer.param_groups[0]['lr']}")
    
    print("\n✓ Training setup example completed!\n")


def example_comparison():
    """
    Example 5: Compare models with and without modules
    示例5：比较有无模块的模型
    """
    print("="*60)
    print("Example 5: Model Comparison")
    print("="*60)
    
    # Model without modules
    # 无模块的模型
    model_basic = HTNet(
        image_size=224,
        patch_size=7,
        num_classes=6,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 10),
        use_micro_attention=False,
        use_roi_module=False
    )
    
    # Model with modules
    # 有模块的模型
    model_enhanced = HTNet(
        image_size=224,
        patch_size=7,
        num_classes=6,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(2, 2, 10),
        use_micro_attention=True,
        use_roi_module=True,
        num_roi_regions=5
    )
    
    params_basic = sum(p.numel() for p in model_basic.parameters())
    params_enhanced = sum(p.numel() for p in model_enhanced.parameters())
    
    print(f"\nBasic Model:")
    print(f"  - Parameters: {params_basic:,}")
    print(f"  - Features: Standard attention")
    
    print(f"\nEnhanced Model:")
    print(f"  - Parameters: {params_enhanced:,}")
    print(f"  - Additional parameters: {params_enhanced - params_basic:,} (+{(params_enhanced - params_basic) / params_basic * 100:.1f}%)")
    print(f"  - Features: Diagonal micro-attention + ROI module")
    print(f"  - Benefits: Better asymmetry detection, background suppression")
    
    print("\n✓ Model comparison completed!\n")


def main():
    """
    Run all examples
    运行所有示例
    """
    print("\n" + "="*60)
    print("Diagonal Micro-Attention & ROI Module Examples")
    print("对角微注意力与ROI模块示例")
    print("="*60 + "\n")
    
    try:
        example_basic_usage()
        example_with_visualization()
        example_asymmetry_analysis()
        example_training_setup()
        example_comparison()
        
        print("="*60)
        print("✓ All examples completed successfully!")
        print("✓ 所有示例成功完成！")
        print("="*60)
        print("\nNext steps:")
        print("  1. Prepare your dataset using prepare_dataset.py")
        print("  2. Train with: python train_facial_palsy.py --use_micro_attention --use_roi_module")
        print("  3. Visualize results: python visualize_asymmetry_roi.py")
        print("\n下一步：")
        print("  1. 使用 prepare_dataset.py 准备数据集")
        print("  2. 训练: python train_facial_palsy.py --use_micro_attention --use_roi_module")
        print("  3. 可视化结果: python visualize_asymmetry_roi.py")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
