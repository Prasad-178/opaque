"""
Shared utility functions.
"""
import numpy as np
from typing import List, Optional
import time


def generate_random_vectors(
    num_vectors: int,
    dimension: int,
    normalize: bool = True,
    seed: Optional[int] = None,
    positive: bool = True,
) -> np.ndarray:
    """
    Generate random vectors for testing.

    Args:
        num_vectors: Number of vectors to generate
        dimension: Dimension of each vector
        normalize: Whether to L2-normalize vectors
        seed: Random seed for reproducibility
        positive: Whether to ensure all values are positive (required for PHE)

    Returns:
        Array of shape (num_vectors, dimension)
    """
    if seed is not None:
        np.random.seed(seed)

    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)

    if normalize:
        vectors = normalize_vectors(vectors)

    if positive:
        # Shift to positive range: normalized vectors are in [-1, 1]
        # Shift to [0.01, 1.01] to avoid zeros
        vectors = (vectors + 1.0) / 2.0 + 0.01

    return vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors.

    Args:
        vectors: Array of shape (n, d)

    Returns:
        Normalized vectors of same shape
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return vectors / norms


def compute_plaintext_similarity(
    query: np.ndarray,
    vectors: np.ndarray
) -> np.ndarray:
    """
    Compute dot product similarities in plaintext (for verification).

    Args:
        query: Query vector of shape (d,) or (1, d)
        vectors: Database vectors of shape (n, d)

    Returns:
        Similarity scores of shape (n,)
    """
    query = query.reshape(1, -1) if query.ndim == 1 else query
    return (vectors @ query.T).flatten()


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Get indices of top-k scores.

    Args:
        scores: Array of scores
        k: Number of top results to return

    Returns:
        Indices of top-k scores (sorted by descending score)
    """
    # Use argpartition for efficiency when k << len(scores)
    if k >= len(scores):
        return np.argsort(scores)[::-1]

    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.elapsed is None:
            if self.start_time is not None:
                return (time.perf_counter() - self.start_time) * 1000
            return 0.0
        return self.elapsed * 1000
