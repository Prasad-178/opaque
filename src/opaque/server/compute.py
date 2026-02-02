"""
Server-side PHE computation engine.

The server performs encrypted dot products without seeing:
- The query vector (it's encrypted)
- The actual similarity scores (results are encrypted)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
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
