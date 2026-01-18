"""
Supervised evaluation methods (e.g., SVM classification).

This module provides supervised learning baselines for comparison,
similar to the original Diehl & Cook (2015) paper.
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Optional, Tuple


def train_linear_svm(train_features: np.ndarray, 
                     train_labels: np.ndarray,
                     C: float = 1.0,
                     max_iter: int = 1000) -> LinearSVC:
    """
    Train a linear SVM classifier.
    
    This follows the evaluation method from Diehl & Cook (2015):
    1. Extract features using STDP (unsupervised)
    2. Train SVM on features (supervised)
    3. Evaluate classification accuracy
    
    Args:
        train_features: Training features (n_samples, n_features)
        train_labels: Training labels (n_samples,)
        C: Regularization parameter
        max_iter: Maximum number of iterations
    
    Returns:
        Trained LinearSVC model
    """
    print(f"Training Linear SVM on {len(train_features)} samples...")
    
    svm = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=0,
        dual=False  # Recommended when n_samples > n_features
    )
    
    svm.fit(train_features, train_labels)
    
    print(f"Training complete. Support vectors: {len(svm.coef_)}")
    
    return svm


def train_kernel_svm(train_features: np.ndarray,
                     train_labels: np.ndarray,
                     kernel: str = 'rbf',
                     C: float = 1.0) -> SVC:
    """
    Train a kernel SVM classifier (slower but potentially more accurate).
    
    Args:
        train_features: Training features (n_samples, n_features)
        train_labels: Training labels (n_samples,)
        kernel: Kernel type ('linear', 'rbf', 'poly')
        C: Regularization parameter
    
    Returns:
        Trained SVC model
    """
    print(f"Training Kernel SVM ({kernel}) on {len(train_features)} samples...")
    
    svm = SVC(
        kernel=kernel,
        C=C,
        random_state=0
    )
    
    svm.fit(train_features, train_labels)
    
    print(f"Training complete. Support vectors: {len(svm.support_)}")
    
    return svm


def train_logistic_regression(train_features: np.ndarray,
                              train_labels: np.ndarray,
                              C: float = 1.0,
                              max_iter: int = 1000) -> LogisticRegression:
    """
    Train a logistic regression classifier (often comparable to SVM).
    
    Args:
        train_features: Training features (n_samples, n_features)
        train_labels: Training labels (n_samples,)
        C: Inverse of regularization strength
        max_iter: Maximum number of iterations
    
    Returns:
        Trained LogisticRegression model
    """
    print(f"Training Logistic Regression on {len(train_features)} samples...")
    
    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=0,
        multi_class='multinomial',
        solver='lbfgs'
    )
    
    lr.fit(train_features, train_labels)
    
    print("Training complete.")
    
    return lr


def evaluate_classifier(model,
                       test_features: np.ndarray,
                       test_labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a trained classifier.
    
    Args:
        model: Trained classifier (SVM, Logistic Regression, etc.)
        test_features: Test features (n_samples, n_features)
        test_labels: Test labels (n_samples,)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating on {len(test_features)} test samples...")
    
    # Predictions
    predictions = model.predict(test_features)
    
    # Accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    # Precision, Recall, F1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average='macro',
        zero_division=0
    )
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    return results


def run_supervised_evaluation(train_features: np.ndarray,
                              train_labels: np.ndarray,
                              test_features: np.ndarray,
                              test_labels: np.ndarray,
                              methods: list = ['linear_svm']) -> Dict[str, Dict[str, float]]:
    """
    Run supervised evaluation using various classifiers.
    
    This provides a comparison to the original Diehl & Cook evaluation.
    
    Args:
        train_features: Training features (n_train, n_features)
        train_labels: Training labels (n_train,)
        test_features: Test features (n_test, n_features)
        test_labels: Test labels (n_test,)
        methods: List of methods to use ['linear_svm', 'kernel_svm', 'logistic']
    
    Returns:
        Dictionary with results for each method
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print('='*60)
        
        # Train classifier
        if method == 'linear_svm':
            model = train_linear_svm(train_features, train_labels)
        elif method == 'kernel_svm':
            model = train_kernel_svm(train_features, train_labels)
        elif method == 'logistic':
            model = train_logistic_regression(train_features, train_labels)
        else:
            print(f"Unknown method: {method}")
            continue
        
        # Evaluate
        results[method] = evaluate_classifier(model, test_features, test_labels)
    
    return results


def compare_supervised_unsupervised(clustering_results: Dict[str, Dict],
                                   supervised_results: Dict[str, Dict]) -> None:
    """
    Print a comparison table of supervised vs unsupervised results.
    
    Args:
        clustering_results: Results from clustering evaluation
        supervised_results: Results from supervised evaluation
    """
    print("\n" + "="*80)
    print("COMPARISON: Unsupervised (Clustering) vs Supervised (Classification)")
    print("="*80)
    
    print("\n📊 Unsupervised Clustering Results:")
    print("-" * 80)
    for method, metrics in clustering_results.items():
        print(f"  {method}:")
        if 'nmi' in metrics:
            print(f"    NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, ACC={metrics['acc']:.4f}")
    
    print("\n📊 Supervised Classification Results:")
    print("-" * 80)
    for method, metrics in supervised_results.items():
        print(f"  {method}:")
        print(f"    Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    print("\n💡 Note:")
    print("  - Supervised results are expected to be much higher (uses labels)")
    print("  - Diehl & Cook (2015) reported ~95% accuracy with Linear SVM")
    print("  - Clustering results are fully unsupervised (no labels during training)")
    print("="*80 + "\n")


# Test code
if __name__ == '__main__':
    print("Testing supervised evaluation module...")
    
    # Generate dummy data
    np.random.seed(0)
    n_train, n_test = 1000, 200
    n_features = 100
    n_classes = 10
    
    train_features = np.random.randn(n_train, n_features)
    train_labels = np.random.randint(0, n_classes, n_train)
    test_features = np.random.randn(n_test, n_features)
    test_labels = np.random.randint(0, n_classes, n_test)
    
    # Run evaluation
    results = run_supervised_evaluation(
        train_features, train_labels,
        test_features, test_labels,
        methods=['linear_svm', 'logistic']
    )
    
    print("\nTest complete!")
    print(f"Results: {results}")
