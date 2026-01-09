"""
Evaluation metrics for clustering and retrieval tasks.
"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from typing import List, Optional


# ==================== Clustering Metrics ====================

def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information.
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    
    Returns:
        NMI score (0-1, higher is better)
    """
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index.
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    
    Returns:
        ARI score (-1 to 1, higher is better)
    """
    return float(adjusted_rand_score(labels_true, labels_pred))


def compute_clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm for optimal matching.
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    
    Returns:
        Accuracy score (0-1, higher is better)
    """
    # Build contingency matrix
    n_clusters = max(max(labels_true), max(labels_pred)) + 1
    contingency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    
    for i in range(len(labels_true)):
        contingency_matrix[labels_true[i], labels_pred[i]] += 1
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Compute accuracy
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(labels_true)
    
    return float(accuracy)


def compute_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute clustering purity (simpler than accuracy, no optimal matching).
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    
    Returns:
        Purity score (0-1, higher is better)
    """
    # For each cluster, find the most common true label
    contingency_matrix = np.zeros((max(labels_pred)+1, max(labels_true)+1), dtype=np.int64)
    
    for i in range(len(labels_true)):
        contingency_matrix[labels_pred[i], labels_true[i]] += 1
    
    # Sum of max values per cluster
    purity = np.sum(np.max(contingency_matrix, axis=1)) / len(labels_true)
    
    return float(purity)


def compute_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Silhouette score (doesn't require ground truth).
    
    Args:
        features: Feature vectors
        labels: Cluster labels
    
    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    from sklearn.metrics import silhouette_score
    
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return float(silhouette_score(features, labels))


def compute_davies_bouldin_score(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin score (doesn't require ground truth, lower is better).
    
    Args:
        features: Feature vectors
        labels: Cluster labels
    
    Returns:
        Davies-Bouldin score (lower is better)
    """
    from sklearn.metrics import davies_bouldin_score
    
    if len(np.unique(labels)) < 2:
        return float('inf')
    
    return float(davies_bouldin_score(features, labels))


# ==================== Retrieval Metrics ====================

def compute_map(retrieved: np.ndarray, groundtruth: np.ndarray, k: int = 100) -> float:
    """
    Compute Mean Average Precision.
    
    Args:
        retrieved: (n_queries, n_retrieved) array of retrieved indices
        groundtruth: (n_queries, n_gt) array of ground truth indices
        k: Number of top results to consider
    
    Returns:
        mAP score (0-1, higher is better)
    """
    aps = []
    
    for i in range(len(retrieved)):
        gt_set = set(groundtruth[i])
        
        hits = 0
        precisions = []
        
        for j, idx in enumerate(retrieved[i][:k]):
            if idx in gt_set:
                hits += 1
                precision_at_j = hits / (j + 1)
                precisions.append(precision_at_j)
        
        # Average precision for this query
        ap = np.mean(precisions) if precisions else 0.0
        aps.append(ap)
    
    # Mean over all queries
    return float(np.mean(aps))


def compute_recall_at_k(retrieved: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    """
    Compute Recall@K.
    
    Args:
        retrieved: (n_queries, n_retrieved) array of retrieved indices
        groundtruth: (n_queries, n_gt) array of ground truth indices
        k: Number of top results to consider
    
    Returns:
        Recall@K score (0-1, higher is better)
    """
    recalls = []
    
    for i in range(len(retrieved)):
        gt_set = set(groundtruth[i])
        retrieved_set = set(retrieved[i][:k])
        
        # Recall = |intersection| / |ground truth|
        recall = len(gt_set & retrieved_set) / len(gt_set)
        recalls.append(recall)
    
    return float(np.mean(recalls))


def compute_precision_at_k(retrieved: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    """
    Compute Precision@K.
    
    Args:
        retrieved: (n_queries, n_retrieved) array of retrieved indices
        groundtruth: (n_queries, n_gt) array of ground truth indices
        k: Number of top results to consider
    
    Returns:
        Precision@K score (0-1, higher is better)
    """
    precisions = []
    
    for i in range(len(retrieved)):
        gt_set = set(groundtruth[i])
        retrieved_set = set(retrieved[i][:k])
        
        # Precision = |intersection| / k
        precision = len(gt_set & retrieved_set) / k
        precisions.append(precision)
    
    return float(np.mean(precisions))


def compute_ndcg_at_k(retrieved: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    Args:
        retrieved: (n_queries, n_retrieved) array of retrieved indices
        groundtruth: (n_queries, n_gt) array of ground truth indices
        k: Number of top results to consider
    
    Returns:
        NDCG@K score (0-1, higher is better)
    """
    ndcgs = []
    
    for i in range(len(retrieved)):
        gt_set = set(groundtruth[i])
        
        # DCG
        dcg = 0.0
        for j, idx in enumerate(retrieved[i][:k]):
            if idx in gt_set:
                # Relevance = 1, rank = j+1
                dcg += 1.0 / np.log2(j + 2)  # +2 because log2(1) = 0
        
        # IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(k, len(gt_set))))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return float(np.mean(ndcgs))


# ==================== Hamming Distance ====================

def hamming_distance(code1: np.ndarray, code2: np.ndarray) -> int:
    """
    Compute Hamming distance between two binary codes.
    
    Args:
        code1: Binary code (1D array)
        code2: Binary code (1D array)
    
    Returns:
        Hamming distance (number of differing bits)
    """
    return int(np.sum(code1 != code2))


def pairwise_hamming_distances(codes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Hamming distances between all codes.
    
    Args:
        codes: (n_samples, code_length) binary array
    
    Returns:
        (n_samples, n_samples) distance matrix
    """
    n_samples = len(codes)
    distances = np.zeros((n_samples, n_samples), dtype=np.int32)
    
    for i in range(n_samples):
        # Vectorized computation for row i
        distances[i] = np.sum(codes != codes[i], axis=1)
    
    return distances


def find_knn_hamming(query_codes: np.ndarray, database_codes: np.ndarray, k: int) -> np.ndarray:
    """
    Find k nearest neighbors using Hamming distance.
    
    Args:
        query_codes: (n_queries, code_length) binary array
        database_codes: (n_database, code_length) binary array
        k: Number of nearest neighbors
    
    Returns:
        (n_queries, k) array of neighbor indices
    """
    n_queries = len(query_codes)
    knn_indices = np.zeros((n_queries, k), dtype=np.int32)
    
    for i in range(n_queries):
        # Compute distances to all database codes
        distances = np.sum(database_codes != query_codes[i], axis=1)
        
        # Find k smallest distances
        knn_indices[i] = np.argsort(distances)[:k]
    
    return knn_indices


# ==================== Summary Functions ====================

def evaluate_clustering_full(labels_true: np.ndarray, labels_pred: np.ndarray, 
                             features: Optional[np.ndarray] = None) -> dict:
    """
    Compute all clustering metrics.
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
        features: Optional feature vectors (for silhouette/DB scores)
    
    Returns:
        Dictionary with all metrics
    """
    results = {
        'nmi': compute_nmi(labels_true, labels_pred),
        'ari': compute_ari(labels_true, labels_pred),
        'acc': compute_clustering_accuracy(labels_true, labels_pred),
        'purity': compute_purity(labels_true, labels_pred),
    }
    
    if features is not None:
        results['silhouette'] = compute_silhouette_score(features, labels_pred)
        results['davies_bouldin'] = compute_davies_bouldin_score(features, labels_pred)
    
    return results


def evaluate_retrieval_full(retrieved: np.ndarray, groundtruth: np.ndarray, 
                            k_values: List[int] = [10, 50, 100]) -> dict:
    """
    Compute all retrieval metrics.
    
    Args:
        retrieved: (n_queries, n_retrieved) array of retrieved indices
        groundtruth: (n_queries, n_gt) array of ground truth indices
        k_values: List of k values for Recall@K and Precision@K
    
    Returns:
        Dictionary with all metrics
    """
    results = {
        'map': compute_map(retrieved, groundtruth, k=max(k_values)),
    }
    
    for k in k_values:
        results[f'recall@{k}'] = compute_recall_at_k(retrieved, groundtruth, k)
        results[f'precision@{k}'] = compute_precision_at_k(retrieved, groundtruth, k)
        results[f'ndcg@{k}'] = compute_ndcg_at_k(retrieved, groundtruth, k)
    
    return results


if __name__ == '__main__':
    # Test metrics
    print("Testing clustering metrics...")
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([1, 1, 0, 2, 2, 2, 0, 0, 0])
    
    nmi = compute_nmi(labels_true, labels_pred)
    ari = compute_ari(labels_true, labels_pred)
    acc = compute_clustering_accuracy(labels_true, labels_pred)
    
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    print(f"ACC: {acc:.3f}")
    
    print("\nTesting retrieval metrics...")
    retrieved = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])
    groundtruth = np.array([[0, 1, 10], [5, 6, 11]])
    
    map_score = compute_map(retrieved, groundtruth, k=4)
    recall_2 = compute_recall_at_k(retrieved, groundtruth, k=2)
    
    print(f"mAP: {map_score:.3f}")
    print(f"Recall@2: {recall_2:.3f}")
