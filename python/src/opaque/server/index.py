"""
LSH (Locality Sensitive Hashing) index for coarse candidate filtering.

Supports two implementations:
1. Faiss IndexLSH - optimized C++ implementation
2. Custom RandomProjectionLSH - pure Python fallback
"""
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseLSHIndex(ABC):
    """Abstract base class for LSH index implementations."""

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[List[str]] = None) -> None:
        """Add vectors to the index."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        """Search for k nearest candidates."""
        pass

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        pass


class FaissLSHIndex(BaseLSHIndex):
    """
    LSH index using Faiss.

    Uses random hyperplane projections to hash vectors into buckets.
    Vectors in the same bucket are likely to be similar.
    """

    def __init__(self, dimension: int, nbits: int = 256):
        """
        Initialize Faiss LSH index.

        Args:
            dimension: Vector dimension
            nbits: Number of hash bits (more bits = more precision, slower)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        self.dimension = dimension
        self.nbits = nbits
        self.index = faiss.IndexLSH(dimension, nbits)
        self.ids: List[str] = []
        self._id_to_idx: dict = {}

    def add(self, vectors: np.ndarray, ids: Optional[List[str]] = None) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Array of shape (n, dimension)
            ids: Optional list of vector IDs
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))

        if ids is None:
            start_idx = len(self.ids)
            ids = [f"vec_{i}" for i in range(start_idx, start_idx + len(vectors))]

        # Track ID mappings
        for id_ in ids:
            self._id_to_idx[id_] = len(self.ids)
            self.ids.append(id_)

        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        """
        Search for k nearest candidates.

        Args:
            query: Query vector of shape (dimension,) or (1, dimension)
            k: Number of candidates to return

        Returns:
            Tuple of (candidate_ids, distances)
        """
        query = query.reshape(1, -1).astype(np.float32)
        k = min(k, self.ntotal)

        distances, indices = self.index.search(query, k)

        # Convert indices to IDs
        candidate_ids = [self.ids[i] for i in indices[0] if i >= 0]

        return candidate_ids, distances[0]

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


class RandomProjectionLSH(BaseLSHIndex):
    """
    Custom LSH implementation using random hyperplane projections.

    Pure Python fallback when Faiss is not available.
    """

    def __init__(self, dimension: int, nbits: int = 256, seed: Optional[int] = None):
        """
        Initialize random projection LSH.

        Args:
            dimension: Vector dimension
            nbits: Number of random hyperplanes
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.nbits = nbits

        # Generate random hyperplanes
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((nbits, dimension)).astype(np.float32)
        self.planes /= np.linalg.norm(self.planes, axis=1, keepdims=True)

        self.vectors: Optional[np.ndarray] = None
        self.hashes: Optional[np.ndarray] = None
        self.ids: List[str] = []

    def _hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute binary hash codes.

        Args:
            vectors: Array of shape (n, dimension)

        Returns:
            Binary hash codes of shape (n, nbits)
        """
        projections = vectors @ self.planes.T
        return (projections > 0).astype(np.uint8)

    def _hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> np.ndarray:
        """Compute Hamming distances."""
        return np.sum(hash1 != hash2, axis=-1)

    def add(self, vectors: np.ndarray, ids: Optional[List[str]] = None) -> None:
        """Add vectors to the index."""
        vectors = vectors.astype(np.float32)

        if ids is None:
            start_idx = len(self.ids)
            ids = [f"vec_{i}" for i in range(start_idx, start_idx + len(vectors))]

        # Compute hashes
        new_hashes = self._hash(vectors)

        if self.vectors is None:
            self.vectors = vectors
            self.hashes = new_hashes
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.hashes = np.vstack([self.hashes, new_hashes])

        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        """Search for k nearest candidates by Hamming distance."""
        query = query.reshape(1, -1).astype(np.float32)
        query_hash = self._hash(query)[0]

        # Compute Hamming distances to all vectors
        distances = self._hamming_distance(self.hashes, query_hash)

        # Get top-k by minimum Hamming distance
        k = min(k, len(self.ids))

        if k == len(self.ids):
            # Return all vectors sorted by distance
            top_indices = np.argsort(distances)
        else:
            top_indices = np.argpartition(distances, k)[:k]
            top_indices = top_indices[np.argsort(distances[top_indices])]

        candidate_ids = [self.ids[i] for i in top_indices]

        return candidate_ids, distances[top_indices].astype(np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.ids)


class LSHIndex:
    """
    Factory class for creating LSH indices.

    Automatically selects Faiss if available, falls back to pure Python.
    """

    def __init__(
        self,
        dimension: int,
        nbits: int = 256,
        use_faiss: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Create an LSH index.

        Args:
            dimension: Vector dimension
            nbits: Number of hash bits
            use_faiss: Try to use Faiss (faster)
            seed: Random seed (only for pure Python implementation)
        """
        self.dimension = dimension
        self.nbits = nbits

        if use_faiss:
            try:
                self._index = FaissLSHIndex(dimension, nbits)
                self._backend = "faiss"
            except ImportError:
                print("Warning: Faiss not available, using pure Python LSH")
                self._index = RandomProjectionLSH(dimension, nbits, seed)
                self._backend = "numpy"
        else:
            self._index = RandomProjectionLSH(dimension, nbits, seed)
            self._backend = "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    def add(self, vectors: np.ndarray, ids: Optional[List[str]] = None) -> None:
        """Add vectors to the index."""
        self._index.add(vectors, ids)

    def search(self, query: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        """Search for k nearest candidates."""
        return self._index.search(query, k)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
