# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
FAISS utility functions for similarity search and novelty detection.

This module provides reusable functions for:
- Creating and managing FAISS indices
- Computing novelty thresholds based on embedding statistics
- Detecting novel embeddings
- Searching for nearest neighbors
"""

import numpy as np
import faiss
from typing import Dict, Tuple, Optional, Union


def create_faiss_index(embedding_dim: int, index_type: str = "flat_l2") -> faiss.Index:
    """
    Create a FAISS index for similarity search.

    :param embedding_dim: Dimension of the embedding vectors
    :param index_type: Type of index to create. Options:
                      - "flat_l2": Brute-force L2 distance (default, exact)
                      - "flat_ip": Brute-force inner product
    :returns: FAISS index object
    """
    if index_type == "flat_l2":
        index = faiss.IndexFlatL2(embedding_dim)
    elif index_type == "flat_ip":
        index = faiss.IndexFlatIP(embedding_dim)
    else:
        raise ValueError(f"Unknown index type: {index_type}. Use 'flat_l2' or 'flat_ip'")

    return index


def add_embeddings_to_index(index: faiss.Index, embeddings: np.ndarray) -> None:
    """
    Add embeddings to a FAISS index.

    :param index: FAISS index object
    :param embeddings: Numpy array of shape (N, embedding_dim) where N is number of embeddings
    """
    if embeddings.shape[0] == 0:
        return

    # Ensure embeddings are float32 (required by FAISS)
    embeddings = embeddings.astype(np.float32)

    # Ensure embeddings are contiguous
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)

    index.add(embeddings)
    assert index.is_trained, "Index should be trained after adding embeddings"


def compute_neighbor_statistics(
    index: faiss.Index,
    embeddings: np.ndarray,
    k: int = 2,
    use_second_neighbor: bool = True
) -> Dict[str, float]:
    """
    Compute statistics (mean and std) of nearest neighbor distances.

    This is used to compute novelty thresholds. When searching embeddings that are
    already in the index, the first nearest neighbor is the embedding itself (distance=0),
    so we use the second nearest neighbor (k=2, use_second_neighbor=True).

    :param index: FAISS index object
    :param embeddings: Numpy array of shape (N, embedding_dim)
    :param k: Number of nearest neighbors to search for
    :param use_second_neighbor: If True, use the second nearest neighbor distance
                                (useful when embeddings are already in the index)
    :returns: Dictionary with 'mean_error' and 'stdev_error'
    """
    if embeddings.shape[0] == 0:
        return {'mean_error': 0.0, 'stdev_error': 0.0}

    # Ensure embeddings are float32 and contiguous
    embeddings = embeddings.astype(np.float32)
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)

    if embeddings.shape[0] == 1:
        # Need at least 2 embeddings to compute neighbor statistics
        return {'mean_error': 0.0, 'stdev_error': 0.0}

    # Search for k nearest neighbors
    distances, _ = index.search(embeddings, k)

    # Extract neighbor distances
    if use_second_neighbor and k >= 2:
        # Use second nearest neighbor (first is self-match)
        neighbor_distances = distances[:, 1]
    else:
        # Use first nearest neighbor
        neighbor_distances = distances[:, 0]

    # Compute statistics
    mean_error = float(np.mean(neighbor_distances))
    stdev_error = float(np.std(neighbor_distances))

    return {
        'mean_error': mean_error,
        'stdev_error': stdev_error,
        'min_distance': float(np.min(neighbor_distances)),
        'max_distance': float(np.max(neighbor_distances)),
        'median_distance': float(np.median(neighbor_distances))
    }


def compute_novelty_threshold(
    mean_error: float,
    stdev_error: float,
    alpha: float = 2.0
) -> float:
    """
    Compute novelty threshold using statistical approach.

    Threshold = mean_error + alpha * stdev_error

    This is based on the assumption that novel embeddings will be outliers,
    i.e., their distance to nearest neighbors will be beyond alpha standard
    deviations from the mean.

    :param mean_error: Mean of nearest neighbor distances
    :param stdev_error: Standard deviation of nearest neighbor distances
    :param alpha: Multiplier for standard deviation (default: 2.0)
    :returns: Novelty threshold value
    """
    return mean_error + alpha * stdev_error


def compute_novelty_score(
    index: faiss.Index,
    embedding: np.ndarray,
    k: int = 1
) -> Dict[str, Union[float, bool]]:
    """
    Compute novelty score for a new embedding.

    :param index: FAISS index object
    :param embedding: Single embedding vector of shape (embedding_dim,) or (1, embedding_dim)
    :param k: Number of nearest neighbors to search for
    :returns: Dictionary with 'novelty_score' (distance to nearest neighbor) and
              'nearest_neighbors' (distances and indices)
    """
    # Ensure embedding is 2D
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # Ensure float32 and contiguous
    embedding = embedding.astype(np.float32)
    if not embedding.flags['C_CONTIGUOUS']:
        embedding = np.ascontiguousarray(embedding)

    if index.ntotal == 0:
        return {
            'novelty_score': float('nan'),
            'is_novel': True,
            'nearest_neighbors': None
        }

    # Search for nearest neighbors
    distances, indices = index.search(embedding, k)

    novelty_score = float(distances[0, 0])  # Distance to nearest neighbor

    nearest_neighbors = {
        'distances': distances[0].tolist(),
        'indices': indices[0].tolist()
    }

    return {
        'novelty_score': novelty_score,
        'nearest_neighbors': nearest_neighbors
    }


def is_novel(
    index: faiss.Index,
    embedding: np.ndarray,
    threshold: float,
    k: int = 1
) -> Dict[str, Union[float, bool]]:
    """
    Determine if an embedding is novel based on a threshold.

    :param index: FAISS index object
    :param embedding: Single embedding vector
    :param threshold: Novelty threshold (from compute_novelty_threshold)
    :param k: Number of nearest neighbors to search for
    :returns: Dictionary with 'is_novel', 'novelty_score', and 'nearest_neighbors'
    """
    result = compute_novelty_score(index, embedding, k)

    if index.ntotal == 0:
        result['is_novel'] = True
    else:
        result['is_novel'] = result['novelty_score'] >= threshold

    return result


def update_index_with_statistics(
    index: faiss.Index,
    embeddings: np.ndarray,
    new_embedding: Optional[np.ndarray] = None,
    alpha: float = 2.0,
    k: int = 2,
    use_second_neighbor: bool = True
) -> Dict[str, Union[float, faiss.Index]]:
    """
    Add embeddings to index and compute updated statistics and threshold.

    This is a convenience function that combines adding embeddings and computing
    statistics in one call.

    :param index: FAISS index object
    :param embeddings: All embeddings (including new ones) of shape (N, embedding_dim)
    :param new_embedding: Optional new embedding to add (will be added to index)
    :param alpha: Multiplier for standard deviation in threshold calculation
    :param k: Number of nearest neighbors for statistics computation
    :param use_second_neighbor: Whether to use second nearest neighbor for statistics
    :returns: Dictionary with updated 'index', 'statistics', and 'threshold'
    """
    # Add new embedding if provided
    if new_embedding is not None:
        add_embeddings_to_index(index, new_embedding.reshape(1, -1))

    # Compute statistics
    statistics = compute_neighbor_statistics(
        index, embeddings, k=k, use_second_neighbor=use_second_neighbor
    )

    # Compute threshold
    threshold = compute_novelty_threshold(
        statistics['mean_error'],
        statistics['stdev_error'],
        alpha=alpha
    )

    return {
        'index': index,
        'statistics': statistics,
        'threshold': threshold
    }


def search_nearest_neighbors(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors in the index.

    :param index: FAISS index object
    :param query_embeddings: Query embeddings of shape (N, embedding_dim)
    :param k: Number of nearest neighbors to return
    :returns: Tuple of (distances, indices) where:
              - distances: shape (N, k) - distances to k nearest neighbors
              - indices: shape (N, k) - indices of k nearest neighbors in the index
    """
    if query_embeddings.shape[0] == 0:
        return np.array([]), np.array([])

    # Ensure float32 and contiguous
    query_embeddings = query_embeddings.astype(np.float32)
    if not query_embeddings.flags['C_CONTIGUOUS']:
        query_embeddings = np.ascontiguousarray(query_embeddings)

    distances, indices = index.search(query_embeddings, k)
    return distances, indices


def create_index_from_embeddings(
    embeddings: np.ndarray,
    index_type: str = "flat_l2",
    alpha: float = 2.0,
    compute_statistics: bool = True
) -> Dict[str, Union[faiss.Index, float, Dict]]:
    """
    Create a FAISS index from embeddings and optionally compute statistics.

    This is a high-level convenience function that creates an index, adds embeddings,
    and computes statistics/threshold in one call.

    :param embeddings: Embeddings of shape (N, embedding_dim)
    :param index_type: Type of FAISS index to create
    :param alpha: Multiplier for standard deviation in threshold calculation
    :param compute_statistics: Whether to compute statistics and threshold
    :returns: Dictionary with 'index', 'statistics', and 'threshold' (if computed)
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot create index from empty embeddings")

    embedding_dim = embeddings.shape[1]

    # Create index
    index = create_faiss_index(embedding_dim, index_type)

    # Add embeddings
    add_embeddings_to_index(index, embeddings)

    result = {'index': index}

    # Compute statistics if requested
    if compute_statistics:
        statistics = compute_neighbor_statistics(
            index, embeddings, k=2, use_second_neighbor=True
        )
        threshold = compute_novelty_threshold(
            statistics['mean_error'],
            statistics['stdev_error'],
            alpha=alpha
        )
        result['statistics'] = statistics
        result['threshold'] = threshold

    return result


def save_index(index: faiss.Index, filepath: str) -> None:
    """
    Save a FAISS index to disk.

    :param index: FAISS index object
    :param filepath: Path to save the index
    """
    faiss.write_index(index, filepath)


def load_index(filepath: str) -> faiss.Index:
    """
    Load a FAISS index from disk.

    :param filepath: Path to the saved index file
    :returns: FAISS index object
    """
    return faiss.read_index(filepath)


# Example usage functions for common workflows

def novelty_detection_workflow(
    initial_embeddings: np.ndarray,
    new_embedding: np.ndarray,
    alpha: float = 2.0
) -> Dict[str, Union[bool, float, Dict]]:
    """
    Complete workflow for novelty detection.

    This function demonstrates the typical workflow:
    1. Create index from initial embeddings
    2. Compute threshold
    3. Check if new embedding is novel

    :param initial_embeddings: Initial embeddings of shape (N, embedding_dim)
    :param new_embedding: New embedding to check, shape (embedding_dim,) or (1, embedding_dim)
    :param alpha: Multiplier for standard deviation
    :returns: Dictionary with novelty detection results
    """
    # Create index and compute threshold
    result = create_index_from_embeddings(
        initial_embeddings,
        alpha=alpha,
        compute_statistics=True
    )

    index = result['index']
    threshold = result['threshold']

    # Check novelty
    novelty_result = is_novel(index, new_embedding, threshold)

    return {
        'is_novel': novelty_result['is_novel'],
        'novelty_score': novelty_result['novelty_score'],
        'threshold': threshold,
        'statistics': result['statistics'],
        'nearest_neighbors': novelty_result['nearest_neighbors']
    }
