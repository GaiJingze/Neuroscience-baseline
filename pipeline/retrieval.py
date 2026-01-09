"""
Retrieval (ANN) evaluation for hashing methods.
"""

import numpy as np
from typing import Dict, List, Optional
from .metrics import evaluate_retrieval_full, find_knn_hamming


def cosine_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and all database vectors.
    
    Args:
        query: (d,) or (1, d) query vector
        database: (n, d) database vectors
    
    Returns:
        (n,) similarity scores
    """
    query = query.reshape(1, -1) if query.ndim == 1 else query
    
    # Normalize
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    database_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    similarities = np.dot(query_norm, database_norm.T).flatten()
    
    return similarities


def euclidean_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between query and all database vectors.
    
    Args:
        query: (d,) or (1, d) query vector
        database: (n, d) database vectors
    
    Returns:
        (n,) distance scores
    """
    query = query.reshape(1, -1) if query.ndim == 1 else query
    
    # L2 distance
    distances = np.linalg.norm(database - query, axis=1)
    
    return distances


def hamming_distance_batch(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distance between query and all database vectors.
    
    Args:
        query: (d,) or (1, d) binary query vector
        database: (n, d) binary database vectors
    
    Returns:
        (n,) distance scores
    """
    query = query.reshape(1, -1) if query.ndim == 1 else query
    
    # XOR and count
    distances = np.sum(database != query, axis=1)
    
    return distances


def brute_force_search(
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    k: int,
    metric: str = 'hamming'
) -> np.ndarray:
    """
    Brute-force k-NN search.
    
    Args:
        query_codes: (n_queries, d) query vectors
        database_codes: (n_database, d) database vectors
        k: Number of nearest neighbors
        metric: Distance metric ('hamming', 'euclidean', 'cosine')
    
    Returns:
        (n_queries, k) array of neighbor indices
    """
    n_queries = len(query_codes)
    knn_indices = np.zeros((n_queries, k), dtype=np.int32)
    
    for i in range(n_queries):
        if metric == 'hamming':
            distances = hamming_distance_batch(query_codes[i], database_codes)
        elif metric == 'euclidean':
            distances = euclidean_distance(query_codes[i], database_codes)
        elif metric == 'cosine':
            similarities = cosine_similarity(query_codes[i], database_codes)
            distances = -similarities  # Convert to distance (lower is better)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Find k smallest distances
        knn_indices[i] = np.argsort(distances)[:k]
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_queries} queries...")
    
    return knn_indices


def faiss_search(
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    k: int,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Fast search using FAISS library (if available).
    
    Args:
        query_codes: (n_queries, d) query vectors
        database_codes: (n_database, d) database vectors
        k: Number of nearest neighbors
        use_gpu: Use GPU acceleration
    
    Returns:
        (n_queries, k) array of neighbor indices
    """
    try:
        import faiss
    except ImportError:
        print("FAISS not available, falling back to brute-force search...")
        return brute_force_search(query_codes, database_codes, k, metric='euclidean')
    
    # Convert to float32 (FAISS requirement)
    database_codes = database_codes.astype(np.float32)
    query_codes = query_codes.astype(np.float32)
    
    d = database_codes.shape[1]
    
    # Build index
    if use_gpu and faiss.get_num_gpus() > 0:
        # GPU index
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        # CPU index
        index = faiss.IndexFlatL2(d)
    
    # Add database vectors
    index.add(database_codes)
    
    # Search
    distances, indices = index.search(query_codes, k)
    
    return indices


def run_retrieval_evaluation(
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    groundtruth: np.ndarray,
    k_values: List[int] = [10, 50, 100],
    metric: str = 'hamming',
    use_faiss: bool = False,
    use_gpu: bool = False,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run retrieval evaluation.
    
    Args:
        query_codes: (n_queries, d) query codes
        database_codes: (n_database, d) database codes
        groundtruth: (n_queries, n_gt) ground truth neighbor indices
        k_values: List of k values for Recall@K
        metric: Distance metric ('hamming', 'euclidean', 'cosine')
        use_faiss: Use FAISS for fast search
        use_gpu: Use GPU (requires FAISS with GPU support)
        verbose: Print progress
    
    Returns:
        Dictionary with retrieval metrics
    """
    if verbose:
        print(f"Running retrieval evaluation...")
        print(f"  Queries: {len(query_codes)}, Database: {len(database_codes)}")
        print(f"  Metric: {metric}, FAISS: {use_faiss}")
    
    # Retrieve neighbors
    max_k = max(k_values + [100])  # For mAP computation
    
    if use_faiss and metric in ['euclidean', 'cosine']:
        retrieved = faiss_search(query_codes, database_codes, max_k, use_gpu=use_gpu)
    else:
        retrieved = brute_force_search(query_codes, database_codes, max_k, metric=metric)
    
    # Evaluate
    results = evaluate_retrieval_full(retrieved, groundtruth, k_values=k_values)
    
    if verbose:
        print(f"\nRetrieval Results:")
        for metric_name, value in results.items():
            print(f"  {metric_name}: {value:.4f}")
    
    return results


def compute_retrieval_speed(
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    k: int = 100,
    metric: str = 'hamming',
    n_trials: int = 3
) -> Dict[str, float]:
    """
    Benchmark retrieval speed.
    
    Args:
        query_codes: (n_queries, d) query codes
        database_codes: (n_database, d) database codes
        k: Number of neighbors
        metric: Distance metric
        n_trials: Number of trials for averaging
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    times = []
    
    for trial in range(n_trials):
        start = time.time()
        _ = brute_force_search(query_codes, database_codes, k, metric=metric)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'queries_per_second': len(query_codes) / np.mean(times),
    }


if __name__ == '__main__':
    # Test retrieval evaluation
    print("Testing retrieval evaluation...")
    
    # Generate synthetic data
    np.random.seed(0)
    database = np.random.randn(1000, 64)
    queries = np.random.randn(100, 64)
    
    # Binarize
    database_binary = (database > 0).astype(float)
    queries_binary = (queries > 0).astype(float)
    
    # Generate fake ground truth (for testing)
    groundtruth = np.random.randint(0, 1000, size=(100, 10))
    
    # Run evaluation
    results = run_retrieval_evaluation(
        query_codes=queries_binary,
        database_codes=database_binary,
        groundtruth=groundtruth,
        k_values=[10, 50],
        metric='hamming',
        use_faiss=False
    )
    
    print("\nTiming test...")
    timing = compute_retrieval_speed(
        query_codes=queries_binary[:10],
        database_codes=database_binary,
        k=100,
        metric='hamming'
    )
    print(f"Queries per second: {timing['queries_per_second']:.2f}")
