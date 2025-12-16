import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)
import torch
import os


class FacialPalsyMetrics:
    """
    Comprehensive evaluation metrics for facial palsy detection.
    Supports binary classification (palsy vs. normal).
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, predictions, labels, probabilities=None):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Predicted class labels (0 or 1)
            labels: True class labels (0 or 1)
            probabilities: Predicted probabilities for positive class (optional)
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()
        
        self.all_preds.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())
        if probabilities is not None:
            self.all_probs.extend(probabilities.flatten())
    
    def compute_metrics(self):
        """
        Compute all evaluation metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)  # Also known as sensitivity
        f1 = f1_score(labels, preds, zero_division=0)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Negative Predictive Value (NPV)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # False Positive Rate (FPR)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate (FNR)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(labels, preds)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, preds)
        
        # Balanced Accuracy
        balanced_acc = (recall + specificity) / 2
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': recall,  # Same as recall
            'specificity': specificity,
            'f1_score': f1,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'mcc': mcc,
            'kappa': kappa,
            'balanced_accuracy': balanced_acc,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'total_samples': len(labels),
            'positive_samples': int(np.sum(labels)),
            'negative_samples': int(len(labels) - np.sum(labels))
        }
        
        # AUC-ROC if probabilities are available
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            try:
                auc_roc = roc_auc_score(labels, probs)
                metrics['auc_roc'] = auc_roc
            except:
                metrics['auc_roc'] = None
        
        return metrics
    
    def print_metrics(self):
        """Print all metrics in a formatted way"""
        metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print("FACIAL PALSY DETECTION - EVALUATION METRICS")
        print("="*60)
        
        print("\n--- Classification Metrics ---")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"Precision:          {metrics['precision']:.4f}")
        print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
        print(f"Specificity:        {metrics['specificity']:.4f}")
        print(f"F1 Score:           {metrics['f1_score']:.4f}")
        print(f"NPV:                {metrics['npv']:.4f}")
        
        if 'auc_roc' in metrics and metrics['auc_roc'] is not None:
            print(f"AUC-ROC:            {metrics['auc_roc']:.4f}")
        
        print("\n--- Statistical Metrics ---")
        print(f"Matthews Corr Coef: {metrics['mcc']:.4f}")
        print(f"Cohen's Kappa:      {metrics['kappa']:.4f}")
        
        print("\n--- Error Rates ---")
        print(f"False Positive Rate: {metrics['fpr']:.4f}")
        print(f"False Negative Rate: {metrics['fnr']:.4f}")
        
        print("\n--- Confusion Matrix ---")
        print(f"True Positive (TP):  {metrics['true_positive']}")
        print(f"True Negative (TN):  {metrics['true_negative']}")
        print(f"False Positive (FP): {metrics['false_positive']}")
        print(f"False Negative (FN): {metrics['false_negative']}")
        
        print("\n--- Dataset Statistics ---")
        print(f"Total Samples:       {metrics['total_samples']}")
        print(f"Positive (Palsy):    {metrics['positive_samples']}")
        print(f"Negative (Normal):   {metrics['negative_samples']}")
        print("="*60 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None, class_names=None):
        """
        Plot and optionally save confusion matrix.
        
        Args:
            save_path: Path to save the figure (optional)
            class_names: List of class names (default: ['Normal', 'Palsy'])
        """
        if class_names is None:
            class_names = ['Normal', 'Palsy']
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Facial Palsy Detection')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot and optionally save ROC curve.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        if len(self.all_probs) == 0:
            print("Warning: No probability scores available for ROC curve")
            return
        
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Facial Palsy Detection')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.close()
    
    def save_metrics_to_file(self, file_path):
        """
        Save all metrics to a text file.
        
        Args:
            file_path: Path to save the metrics
        """
        metrics = self.compute_metrics()
        
        with open(file_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FACIAL PALSY DETECTION - EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("--- Classification Metrics ---\n")
            f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Precision:          {metrics['precision']:.4f}\n")
            f.write(f"Recall/Sensitivity: {metrics['recall']:.4f}\n")
            f.write(f"Specificity:        {metrics['specificity']:.4f}\n")
            f.write(f"F1 Score:           {metrics['f1_score']:.4f}\n")
            f.write(f"NPV:                {metrics['npv']:.4f}\n")
            
            if 'auc_roc' in metrics and metrics['auc_roc'] is not None:
                f.write(f"AUC-ROC:            {metrics['auc_roc']:.4f}\n")
            
            f.write("\n--- Statistical Metrics ---\n")
            f.write(f"Matthews Corr Coef: {metrics['mcc']:.4f}\n")
            f.write(f"Cohen's Kappa:      {metrics['kappa']:.4f}\n")
            
            f.write("\n--- Error Rates ---\n")
            f.write(f"False Positive Rate: {metrics['fpr']:.4f}\n")
            f.write(f"False Negative Rate: {metrics['fnr']:.4f}\n")
            
            f.write("\n--- Confusion Matrix ---\n")
            f.write(f"True Positive (TP):  {metrics['true_positive']}\n")
            f.write(f"True Negative (TN):  {metrics['true_negative']}\n")
            f.write(f"False Positive (FP): {metrics['false_positive']}\n")
            f.write(f"False Negative (FN): {metrics['false_negative']}\n")
            
            f.write("\n--- Dataset Statistics ---\n")
            f.write(f"Total Samples:       {metrics['total_samples']}\n")
            f.write(f"Positive (Palsy):    {metrics['positive_samples']}\n")
            f.write(f"Negative (Normal):   {metrics['negative_samples']}\n")
            f.write("="*60 + "\n")
        
        print(f"Metrics saved to: {file_path}")
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            str: Classification report
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        target_names = ['Normal', 'Palsy']
        report = classification_report(labels, preds, target_names=target_names)
        
        return report
