#!/usr/bin/env python3
"""
Test script for evaluation metrics module.
Demonstrates how to use the FacialPalsyMetrics class.
"""

import numpy as np
from evaluation_metrics import FacialPalsyMetrics


def test_metrics():
    """Test the metrics with sample data"""
    
    print("Testing Facial Palsy Evaluation Metrics")
    print("="*60)
    
    # Create sample predictions and labels
    # Simulating predictions for 100 samples
    np.random.seed(42)
    n_samples = 100
    
    # Generate ground truth (50% palsy, 50% normal)
    labels = np.random.randint(0, 2, size=n_samples)
    
    # Generate predictions (with ~80% accuracy)
    predictions = labels.copy()
    # Introduce some errors
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    
    # Generate probability scores
    probabilities = np.random.rand(n_samples)
    # Make probabilities somewhat correlated with predictions
    probabilities = np.where(predictions == 1, 
                            np.clip(probabilities + 0.3, 0, 1), 
                            np.clip(probabilities - 0.3, 0, 1))
    
    print(f"\nSample size: {n_samples}")
    print(f"Actual positives (palsy): {np.sum(labels == 1)}")
    print(f"Actual negatives (normal): {np.sum(labels == 0)}")
    print(f"Predicted positives: {np.sum(predictions == 1)}")
    print(f"Predicted negatives: {np.sum(predictions == 0)}")
    
    # Initialize metrics tracker
    metrics = FacialPalsyMetrics()
    
    # Update metrics with predictions
    metrics.update(predictions, labels, probabilities)
    
    # Compute and print metrics
    print("\n" + "="*60)
    print("COMPUTED METRICS")
    print("="*60)
    metrics.print_metrics()
    
    # Get classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(metrics.get_classification_report())
    
    # Save metrics to file
    metrics.save_metrics_to_file('test_metrics_output.txt')
    
    # Plot confusion matrix
    metrics.plot_confusion_matrix(save_path='test_confusion_matrix.png')
    
    # Plot ROC curve
    metrics.plot_roc_curve(save_path='test_roc_curve.png')
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("Generated files:")
    print("  - test_metrics_output.txt")
    print("  - test_confusion_matrix.png")
    print("  - test_roc_curve.png")
    print("="*60)


def test_perfect_classification():
    """Test with perfect classification"""
    
    print("\n\n" + "="*60)
    print("Testing with PERFECT classification")
    print("="*60)
    
    n_samples = 50
    labels = np.array([0]*25 + [1]*25)
    predictions = labels.copy()  # Perfect predictions
    probabilities = np.array([0.1]*25 + [0.9]*25)  # Clear probabilities
    
    metrics = FacialPalsyMetrics()
    metrics.update(predictions, labels, probabilities)
    metrics.print_metrics()


def test_random_classification():
    """Test with random classification"""
    
    print("\n\n" + "="*60)
    print("Testing with RANDOM classification")
    print("="*60)
    
    np.random.seed(123)
    n_samples = 100
    labels = np.random.randint(0, 2, size=n_samples)
    predictions = np.random.randint(0, 2, size=n_samples)
    probabilities = np.random.rand(n_samples)
    
    metrics = FacialPalsyMetrics()
    metrics.update(predictions, labels, probabilities)
    metrics.print_metrics()


if __name__ == '__main__':
    # Run tests
    test_metrics()
    test_perfect_classification()
    test_random_classification()
    
    print("\n\nAll tests completed!")
