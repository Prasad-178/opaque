"""
Server-side PHE computation engine.

The server performs encrypted dot products without seeing:
- The query vector (it's encrypted)
- The actual similarity scores (results are encrypted)

Supports parallel computation for improved performance.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from lightphe.models.Tensor import EncryptedTensor

from opaque.shared.utils import Timer, generate_random_vectors


@dataclass
class VectorStore:
    """
    In-memory vector database.

    Stores plaintext vectors that will be used for encrypted dot products.
    """
    vectors: np.ndarray  # Shape: (num_vectors, dimension)
    ids: List[str] = field(default_factory=list)
    metadata: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        if not self.ids:
            self.ids = [f"vec_{i}" for i in range(len(self.vectors))]

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1] if len(self.vectors) > 0 else 0

    def get_vectors_by_ids(self, ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get vectors for given IDs."""
        id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        indices = [id_to_idx[id_] for id_ in ids if id_ in id_to_idx]
        return self.vectors[indices], [self.ids[i] for i in indices]

    def get_all_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """Get all vectors."""
        return self.vectors, self.ids


class ComputeEngine:
    """
    Server-side computation engine for encrypted operations.

    Performs PHE dot products: encrypted_query @ plaintext_vector
    without ever seeing the query or the results in plaintext.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize compute engine.

        Args:
            vector_store: Database of plaintext vectors
        """
        self.vector_store = vector_store

    def compute_encrypted_scores(
        self,
        encrypted_query: EncryptedTensor,
        vector_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Tuple[List[Any], List[str], float]:
        """
        Compute encrypted dot products.

        Args:
            encrypted_query: Encrypted query vector from client
            vector_ids: Optional list of vector IDs to score (for candidate filtering)
            verbose: Print progress

        Returns:
            Tuple of (encrypted_scores, vector_ids, time_ms)
        """
        # Get vectors to score
        if vector_ids:
            vectors, ids = self.vector_store.get_vectors_by_ids(vector_ids)
        else:
            vectors, ids = self.vector_store.get_all_vectors()

        encrypted_scores = []

        with Timer() as t:
            for i, vec in enumerate(vectors):
                # LightPHE handles encrypted @ plaintext dot product
                encrypted_score = encrypted_query @ vec.tolist()
                encrypted_scores.append(encrypted_score)

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Computed {i+1}/{len(vectors)} scores...")

        return encrypted_scores, ids, t.elapsed_ms

    def compute_single_score(
        self,
        encrypted_query: EncryptedTensor,
        vector_id: str,
    ) -> Any:
        """
        Compute encrypted dot product for a single vector.

        Args:
            encrypted_query: Encrypted query vector
            vector_id: ID of vector to score

        Returns:
            Encrypted score (EncryptedTensor)
        """
        vectors, _ = self.vector_store.get_vectors_by_ids([vector_id])
        if len(vectors) == 0:
            raise ValueError(f"Vector {vector_id} not found")

        return encrypted_query @ vectors[0].tolist()

    def compute_encrypted_scores_parallel(
        self,
        encrypted_query: EncryptedTensor,
        vector_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        chunk_size: int = 10,
        verbose: bool = False,
    ) -> Tuple[List[Any], List[str], float]:
        """
        Compute encrypted dot products in parallel using ThreadPoolExecutor.

        Uses chunked parallel processing for better throughput.

        Args:
            encrypted_query: Encrypted query vector from client
            vector_ids: Optional list of vector IDs to score
            num_workers: Number of parallel workers (default: CPU count)
            chunk_size: Vectors per chunk
            verbose: Print progress

        Returns:
            Tuple of (encrypted_scores, vector_ids, time_ms)
        """
        # Get vectors to score
        if vector_ids:
            vectors, ids = self.vector_store.get_vectors_by_ids(vector_ids)
        else:
            vectors, ids = self.vector_store.get_all_vectors()

        if num_workers is None:
            num_workers = mp.cpu_count()

        if verbose:
            print(f"  Computing {len(vectors)} scores with {num_workers} workers...")

        def compute_dot_product(vec: np.ndarray) -> Any:
            """Compute single encrypted dot product."""
            return encrypted_query @ vec.tolist()

        def compute_chunk(chunk_vectors: List[np.ndarray]) -> List[Any]:
            """Compute dot products for a chunk of vectors."""
            return [encrypted_query @ vec.tolist() for vec in chunk_vectors]

        with Timer() as t:
            if chunk_size > 1 and len(vectors) > chunk_size:
                # Chunked parallel processing
                chunks = [
                    vectors[i:i + chunk_size]
                    for i in range(0, len(vectors), chunk_size)
                ]

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    chunk_results = list(executor.map(compute_chunk, chunks))

                # Flatten results
                encrypted_scores = [
                    score for chunk_result in chunk_results for score in chunk_result
                ]
            else:
                # Per-vector parallel processing
                vector_list = [vectors[i] for i in range(len(vectors))]
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    encrypted_scores = list(executor.map(compute_dot_product, vector_list))

        if verbose:
            print(f"  Completed in {t.elapsed_ms:.0f}ms")

        return encrypted_scores, ids, t.elapsed_ms


def _compute_dot_products_chunk(args):
    """Compute dot products for a chunk of vectors (for multiprocessing)."""
    encrypted_query, vectors_chunk = args
    return [encrypted_query @ vec.tolist() for vec in vectors_chunk]


class ParallelComputeEngine:
    """
    Compute engine with true multiprocessing for maximum parallelism.

    Uses ProcessPoolExecutor to bypass Python GIL for CPU-bound PHE operations.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def compute_encrypted_scores_multiprocess(
        self,
        encrypted_query: EncryptedTensor,
        vector_ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[List[Any], List[str], float]:
        """
        Compute encrypted dot products using multiprocessing.

        Args:
            encrypted_query: Encrypted query from client
            vector_ids: Optional subset of vectors to score
            num_workers: Number of worker processes
            chunk_size: Vectors per chunk (auto-calculated if None)
            verbose: Print progress

        Returns:
            Tuple of (encrypted_scores, ids, time_ms)
        """
        if vector_ids:
            vectors, ids = self.vector_store.get_vectors_by_ids(vector_ids)
        else:
            vectors, ids = self.vector_store.get_all_vectors()

        if num_workers is None:
            num_workers = mp.cpu_count()

        if chunk_size is None:
            chunk_size = max(1, len(vectors) // num_workers)

        if verbose:
            print(f"  Computing {len(vectors)} scores with {num_workers} workers...")

        # Create chunks
        chunks = []
        for i in range(0, len(vectors), chunk_size):
            chunk_vectors = vectors[i:i + chunk_size]
            chunks.append((encrypted_query, chunk_vectors))

        with Timer() as t:
            # Process in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                chunk_results = list(executor.map(_compute_dot_products_chunk, chunks))

            # Flatten
            encrypted_scores = [
                score for chunk_result in chunk_results for score in chunk_result
            ]

        if verbose:
            print(f"  Completed in {t.elapsed_ms:.0f}ms")

        return encrypted_scores, ids, t.elapsed_ms


def create_mock_database(
    num_vectors: int,
    dimension: int,
    seed: Optional[int] = None,
) -> VectorStore:
    """
    Create a mock vector database for testing.

    Args:
        num_vectors: Number of vectors
        dimension: Vector dimension
        seed: Random seed

    Returns:
        VectorStore instance
    """
    vectors = generate_random_vectors(num_vectors, dimension, normalize=True, seed=seed)
    ids = [f"doc_{i:06d}" for i in range(num_vectors)]

    return VectorStore(vectors=vectors, ids=ids)
