import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import json

from Model import HTNet
from facial_palsy_dataset import FacialPalsyDataset, CKPlusFacialPalsyDataset


def reset_weights(m):
    """Reset the weights for network to avoid weight leakage"""
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def compute_metrics(y_true, y_pred, num_classes):
    """
    Compute evaluation metrics for facial palsy grading
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    
    per_class_acc = []
    for i in range(num_classes):
        mask = np.array(y_true) == i
        if np.sum(mask) > 0:
            class_acc = np.sum((np.array(y_pred)[mask]) == i) / np.sum(mask)
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    mean_class_acc = np.mean(per_class_acc)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mean_class_accuracy': mean_class_acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, num_classes):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    
    return val_loss, metrics, all_preds, all_labels


def train_model(config):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    print(f'\nLoading {config.dataset_type} dataset...')
    
    if config.dataset_type == 'FNP':
        train_dataset = FacialPalsyDataset(
            data_root=config.data_root,
            csv_file=config.train_csv,
            dataset_type='FNP',
            image_size=config.image_size,
            split='train'
        )
        val_dataset = FacialPalsyDataset(
            data_root=config.data_root,
            csv_file=config.val_csv,
            dataset_type='FNP',
            image_size=config.image_size,
            split='val'
        )
    elif config.dataset_type == 'CK+':
        train_dataset = CKPlusFacialPalsyDataset(
            data_root=config.data_root,
            image_size=config.image_size,
            split='train'
        )
        val_dataset = CKPlusFacialPalsyDataset(
            data_root=config.data_root,
            image_size=config.image_size,
            split='val'
        )
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    print('\nInitializing HTNet model...')
    model = HTNet(
        image_size=config.image_size,
        patch_size=config.patch_size,
        dim=config.dim,
        heads=config.heads,
        num_hierarchies=config.num_hierarchies,
        block_repeats=config.block_repeats,
        num_classes=config.num_classes
    )
    
    model = model.to(device)
    
    if config.pretrained_path and os.path.exists(config.pretrained_path):
        print(f'Loading pretrained weights from {config.pretrained_path}')
        model.load_state_dict(torch.load(config.pretrained_path), strict=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    best_val_acc = 0.0
    best_f1 = 0.0
    train_history = []
    
    print(f'\nStarting training for {config.epochs} epochs...')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Batch size: {config.batch_size}')
    
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device, config.num_classes
        )
        
        scheduler.step(val_metrics['accuracy'])
        
        epoch_time = time.time() - epoch_start
        
        print(f'\nEpoch [{epoch}/{config.epochs}] - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_metrics["accuracy"]:.4f}')
        print(f'Val F1 (macro): {val_metrics["f1_macro"]:.4f} | Mean Class Acc: {val_metrics["mean_class_accuracy"]:.4f}')
        
        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_mean_class_acc': val_metrics['mean_class_accuracy']
        }
        train_history.append(history_entry)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_f1 = val_metrics['f1_macro']
            
            save_path = os.path.join(config.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_f1,
                'config': vars(config)
            }, save_path)
            print(f'✓ Best model saved with Val Acc: {best_val_acc:.4f}, F1: {best_f1:.4f}')
        
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy']
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Training completed in {total_time/60:.2f} minutes')
    print(f'Best Val Accuracy: {best_val_acc:.4f}')
    print(f'Best Val F1 (macro): {best_f1:.4f}')
    print(f'{"="*60}\n')
    
    history_path = os.path.join(config.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    return model, best_val_acc, best_f1


def main():
    parser = argparse.ArgumentParser(description='Train HTNet for Facial Palsy Recognition and Grading')
    
    parser.add_argument('--data_root', type=str, default='./datasets/facial_palsy',
                       help='Root directory of the dataset')
    parser.add_argument('--train_csv', type=str, default=None,
                       help='CSV file for training data (optional)')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='CSV file for validation data (optional)')
    parser.add_argument('--dataset_type', type=str, default='FNP', choices=['FNP', 'CK+'],
                       help='Dataset type: FNP or CK+')
    
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=7,
                       help='Patch size for HTNet')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of facial palsy grades (default: 6 for House-Brackmann scale)')
    
    parser.add_argument('--dim', type=int, default=256,
                       help='Dimension of the model')
    parser.add_argument('--heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--num_hierarchies', type=int, default=3,
                       help='Number of hierarchies in HTNet')
    parser.add_argument('--block_repeats', type=int, nargs='+', default=[2, 2, 10],
                       help='Number of transformer blocks at each hierarchy')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model weights')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save training logs')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    config = parser.parse_args()
    
    config.block_repeats = tuple(config.block_repeats)
    
    print('\n' + '='*60)
    print('HTNet for Facial Palsy Recognition and Grading')
    print('='*60)
    print('\nConfiguration:')
    for key, value in vars(config).items():
        print(f'  {key}: {value}')
    print('='*60 + '\n')
    
    train_model(config)


if __name__ == '__main__':
    main()
