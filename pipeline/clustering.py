"""
Clustering algorithms and evaluation.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from typing import Dict, List, Optional
from .metrics import evaluate_clustering_full


def kmeans_clustering(features: np.ndarray, n_clusters: int, **kwargs) -> np.ndarray:
    """
    K-Means clustering (suitable for continuous features).
    
    Args:
        features: (n_samples, n_features) array
        n_clusters: Number of clusters
        **kwargs: Additional arguments for KMeans
    
    Returns:
        Cluster labels
    """
    # Set reasonable defaults for speed
    defaults = {
        'n_init': 3,  # Reduce from default 10 to 3 for speed
        'max_iter': 100,  # Limit iterations
        'random_state': 0,  # For reproducibility
    }
    defaults.update(kwargs)
    
    kmeans = KMeans(n_clusters=n_clusters, **defaults)
    labels = kmeans.fit_predict(features)
    return labels


def kmedoids_clustering(features: np.ndarray, n_clusters: int, metric: str = 'hamming', **kwargs) -> np.ndarray:
    """
    K-Medoids clustering (suitable for binary codes with Hamming distance).
    
    Args:
        features: (n_samples, n_features) array
        n_clusters: Number of clusters
        metric: Distance metric ('hamming', 'euclidean', 'cosine', etc.)
        **kwargs: Additional arguments for KMedoids
    
    Returns:
        Cluster labels
    """
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, **kwargs)
    labels = kmedoids.fit_predict(features)
    return labels


def spectral_clustering(features: np.ndarray, n_clusters: int, **kwargs) -> np.ndarray:
    """
    Spectral clustering (works with any similarity matrix).
    
    Args:
        features: (n_samples, n_features) array
        n_clusters: Number of clusters
        **kwargs: Additional arguments for SpectralClustering
    
    Returns:
        Cluster labels
    """
    # Set reasonable defaults for speed
    defaults = {
        'random_state': 0,
        'n_init': 3,  # Reduce number of initializations
    }
    defaults.update(kwargs)
    
    spectral = SpectralClustering(n_clusters=n_clusters, **defaults)
    labels = spectral.fit_predict(features)
    return labels


def hierarchical_clustering(features: np.ndarray, n_clusters: int, metric: str = 'euclidean', 
                            linkage: str = 'average', **kwargs) -> np.ndarray:
    """
    Hierarchical clustering.
    
    Args:
        features: (n_samples, n_features) array
        n_clusters: Number of clusters
        metric: Distance metric
        linkage: Linkage criterion ('average', 'complete', 'single', 'ward')
        **kwargs: Additional arguments for AgglomerativeClustering
    
    Returns:
        Cluster labels
    """
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric=metric if linkage != 'ward' else 'euclidean',
        linkage=linkage, 
        **kwargs
    )
    labels = hierarchical.fit_predict(features)
    return labels


def run_clustering_evaluation(
    codes: np.ndarray, 
    labels_true: np.ndarray,
    n_clusters: int,
    methods: List[str] = ['kmeans', 'kmedoids', 'spectral'],
    is_binary: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run multiple clustering methods and evaluate.
    
    Args:
        codes: (n_samples, code_dim) feature codes
        labels_true: Ground truth labels
        n_clusters: Number of clusters
        methods: List of clustering methods to try
        is_binary: Whether codes are binary (affects distance metric choice)
        verbose: Print progress
    
    Returns:
        Dictionary mapping method names to metric dictionaries
    """
    results = {}
    
    for method in methods:
        if verbose:
            print(f"Running {method} clustering...")
        
        try:
            if method == 'kmeans':
                labels_pred = kmeans_clustering(codes, n_clusters)
            
            elif method == 'kmedoids':
                metric = 'hamming' if is_binary else 'euclidean'
                labels_pred = kmedoids_clustering(codes, n_clusters, metric=metric, random_state=0, max_iter=100)
            
            elif method == 'kmedoids_cosine':
                labels_pred = kmedoids_clustering(codes, n_clusters, metric='cosine', random_state=0)
            
            elif method == 'spectral':
                labels_pred = spectral_clustering(codes, n_clusters)
            
            elif method == 'hierarchical':
                metric = 'hamming' if is_binary else 'euclidean'
                labels_pred = hierarchical_clustering(codes, n_clusters, metric=metric)
            
            else:
                print(f"Unknown method: {method}, skipping...")
                continue
            
            # Evaluate
            metrics = evaluate_clustering_full(labels_true, labels_pred, features=codes)
            results[method] = metrics
            
            if verbose:
                print(f"  NMI={metrics['nmi']:.3f}, ARI={metrics['ari']:.3f}, ACC={metrics['acc']:.3f}")
        
        except Exception as e:
            print(f"Error running {method}: {e}")
            continue
    
    return results


def choose_best_clustering(results: Dict[str, Dict[str, float]], metric: str = 'acc') -> str:
    """
    Choose the best clustering method based on a specific metric.
    
    Args:
        results: Dictionary from run_clustering_evaluation
        metric: Metric to use for selection ('nmi', 'ari', 'acc')
    
    Returns:
        Name of best method
    """
    best_method = max(results.keys(), key=lambda m: results[m][metric])
    return best_method


if __name__ == '__main__':
    # Test clustering
    from sklearn.datasets import make_blobs
    
    print("Testing clustering evaluation...")
    
    # Generate synthetic data
    X, y = make_blobs(n_samples=300, n_features=50, centers=3, random_state=0)
    
    # Binarize to simulate binary codes
    X_binary = (X > np.median(X, axis=1, keepdims=True)).astype(float)
    
    # Run evaluation
    results = run_clustering_evaluation(
        codes=X_binary,
        labels_true=y,
        n_clusters=3,
        methods=['kmeans', 'kmedoids', 'spectral'],
        is_binary=True
    )
    
    print("\nResults:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.3f}")
    
    best = choose_best_clustering(results, metric='acc')
    print(f"\nBest method (by ACC): {best}")
