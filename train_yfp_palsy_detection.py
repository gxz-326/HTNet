import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from distutils.util import strtobool

from Model import HTNet
from yfp_dataset import YFPFacialPalsyDataset, YFPOpticalFlowDataset
from evaluation_metrics import FacialPalsyMetrics


def reset_weights(m):
    """Reset the weights for network to avoid weight leakage"""
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    num_correct = 0
    num_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        num_correct += (pred == target).sum().item()
        num_samples += target.size(0)
    
    avg_loss = train_loss / num_samples
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device, metrics_tracker):
    """Evaluate the model"""
    model.eval()
    test_loss = 0.0
    num_correct = 0
    num_samples = 0
    
    metrics_tracker.reset()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            num_correct += (pred == target).sum().item()
            num_samples += target.size(0)
            
            # Get probabilities for positive class (palsy)
            probs = torch.softmax(output, dim=1)[:, 1]
            
            # Update metrics
            metrics_tracker.update(pred, target, probs)
    
    avg_loss = test_loss / num_samples
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy


def train_subject_loso(config):
    """
    Train using Leave-One-Subject-Out (LOSO) cross-validation.
    Each subject is used as test set once while others are used for training.
    """
    print("="*80)
    print("YFP FACIAL PALSY DETECTION - Leave-One-Subject-Out Cross-Validation")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from: {config.csv_file}")
    if config.use_optical_flow:
        dataset = YFPOpticalFlowDataset(
            csv_file=config.csv_file,
            data_root=config.data_root,
            image_size=config.image_size
        )
        print("Using optical flow features")
    else:
        dataset = YFPFacialPalsyDataset(
            csv_file=config.csv_file,
            data_root=config.data_root,
            image_size=config.image_size,
            use_optical_flow=False
        )
        print("Using raw images")
    
    # Get unique subjects
    subject_ids = dataset.get_subject_ids()
    print(f"\nTotal subjects: {len(subject_ids)}")
    print(f"Total samples: {len(dataset)}")
    
    # Create output directory
    if config.train:
        os.makedirs(config.weights_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
    
    # LOSO Cross-validation
    all_subjects_metrics = []
    global_metrics = FacialPalsyMetrics()
    
    for subject_idx, test_subject in enumerate(subject_ids):
        print(f"\n{'='*80}")
        print(f"LOSO Fold {subject_idx + 1}/{len(subject_ids)} - Test Subject: {test_subject}")
        print(f"{'='*80}")
        
        # Split data into train and test
        train_indices = []
        test_indices = []
        
        for idx in range(len(dataset)):
            subject_id = dataset.data_frame.iloc[idx]['subject_id']
            if subject_id == test_subject:
                test_indices.append(idx)
            else:
                train_indices.append(idx)
        
        print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        
        # Create data loaders
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        test_loader = DataLoader(
            test_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # Initialize model
        model = HTNet(
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.dim,
            heads=config.heads,
            num_hierarchies=config.num_hierarchies,
            block_repeats=tuple(config.block_repeats),
            num_classes=2  # Binary classification: Normal vs Palsy
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Weight path for this subject
        weight_path = os.path.join(config.weights_dir, f'subject_{test_subject}.pth')
        
        # Load or train
        if config.train:
            print("\nTraining...")
            best_test_acc = 0.0
            best_epoch = 0
            
            for epoch in range(1, config.epochs + 1):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                
                # Evaluate
                metrics_tracker = FacialPalsyMetrics()
                test_loss, test_acc = evaluate(
                    model, test_loader, criterion, device, metrics_tracker
                )
                
                # Save best model
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), weight_path)
                
                # Print progress
                if epoch % config.print_freq == 0 or epoch == 1:
                    print(f"Epoch [{epoch}/{config.epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            print(f"\nBest Test Accuracy: {best_test_acc:.4f} at Epoch {best_epoch}")
        else:
            print(f"\nLoading weights from: {weight_path}")
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path))
            else:
                print(f"Warning: Weight file not found. Using random initialization.")
        
        # Final evaluation with best model
        if config.train and os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
        
        metrics_tracker = FacialPalsyMetrics()
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, metrics_tracker
        )
        
        # Compute and print metrics for this subject
        print(f"\n--- Results for Subject {test_subject} ---")
        subject_metrics = metrics_tracker.print_metrics()
        all_subjects_metrics.append(subject_metrics)
        
        # Update global metrics
        global_metrics.update(
            np.array(metrics_tracker.all_preds),
            np.array(metrics_tracker.all_labels),
            np.array(metrics_tracker.all_probs) if len(metrics_tracker.all_probs) > 0 else None
        )
        
        # Save confusion matrix for this subject
        cm_path = os.path.join(config.results_dir, f'confusion_matrix_subject_{test_subject}.png')
        metrics_tracker.plot_confusion_matrix(save_path=cm_path)
        
        # Save ROC curve for this subject
        if len(metrics_tracker.all_probs) > 0:
            roc_path = os.path.join(config.results_dir, f'roc_curve_subject_{test_subject}.png')
            metrics_tracker.plot_roc_curve(save_path=roc_path)
    
    # Print overall results across all subjects
    print("\n" + "="*80)
    print("OVERALL RESULTS ACROSS ALL SUBJECTS (LOSO)")
    print("="*80)
    overall_metrics = global_metrics.print_metrics()
    
    # Save overall confusion matrix
    cm_path = os.path.join(config.results_dir, 'confusion_matrix_overall.png')
    global_metrics.plot_confusion_matrix(save_path=cm_path)
    
    # Save overall ROC curve
    if len(global_metrics.all_probs) > 0:
        roc_path = os.path.join(config.results_dir, 'roc_curve_overall.png')
        global_metrics.plot_roc_curve(save_path=roc_path)
    
    # Save overall metrics to file
    metrics_file = os.path.join(config.results_dir, 'overall_metrics.txt')
    global_metrics.save_metrics_to_file(metrics_file)
    
    # Save per-subject metrics summary
    summary_file = os.path.join(config.results_dir, 'per_subject_summary.csv')
    summary_df = pd.DataFrame(all_subjects_metrics)
    summary_df.insert(0, 'subject_id', subject_ids)
    summary_df.to_csv(summary_file, index=False)
    print(f"\nPer-subject summary saved to: {summary_file}")
    
    # Print average metrics across subjects
    print("\n" + "="*80)
    print("AVERAGE METRICS ACROSS SUBJECTS")
    print("="*80)
    for key in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'balanced_accuracy']:
        if key in all_subjects_metrics[0]:
            avg_value = np.mean([m[key] for m in all_subjects_metrics])
            std_value = np.std([m[key] for m in all_subjects_metrics])
            print(f"{key.replace('_', ' ').title():20s}: {avg_value:.4f} ± {std_value:.4f}")
    
    if 'auc_roc' in all_subjects_metrics[0] and all_subjects_metrics[0]['auc_roc'] is not None:
        auc_values = [m['auc_roc'] for m in all_subjects_metrics if m['auc_roc'] is not None]
        if auc_values:
            avg_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
            print(f"{'AUC-ROC':20s}: {avg_auc:.4f} ± {std_auc:.4f}")
    
    print("="*80)


def main(config):
    """Main training function"""
    start_time = time.time()
    
    # Train with LOSO cross-validation
    train_subject_loso(config)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YFP Facial Palsy Detection with HTNet')
    
    # Dataset parameters
    parser.add_argument('--csv_file', type=str, default='./datasets/yfp_dataset.csv',
                       help='Path to CSV file with dataset annotations')
    parser.add_argument('--data_root', type=str, default='./datasets/YFP',
                       help='Root directory containing the images')
    parser.add_argument('--use_optical_flow', type=strtobool, default='true',
                       help='Whether to use optical flow features')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=28,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=7,
                       help='Patch size for vision transformer')
    parser.add_argument('--dim', type=int, default=256,
                       help='Dimension of transformer')
    parser.add_argument('--heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--num_hierarchies', type=int, default=3,
                       help='Number of hierarchical levels')
    parser.add_argument('--block_repeats', type=int, nargs='+', default=[2, 2, 10],
                       help='Number of transformer blocks at each hierarchy')
    
    # Training parameters
    parser.add_argument('--train', type=strtobool, default='false',
                       help='Train or evaluate only')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--print_freq', type=int, default=10,
                       help='Print frequency during training')
    
    # Output parameters
    parser.add_argument('--weights_dir', type=str, default='./yfp_weights',
                       help='Directory to save model weights')
    parser.add_argument('--results_dir', type=str, default='./yfp_results',
                       help='Directory to save results and metrics')
    
    config = parser.parse_args()
    
    # Print configuration
    print("\nConfiguration:")
    for arg in vars(config):
        print(f"  {arg}: {getattr(config, arg)}")
    print()
    
    main(config)
