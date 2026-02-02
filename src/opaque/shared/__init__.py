"""Shared utilities and protocol definitions."""
from opaque.shared.protocol import (
    VectorDimension,
    BenchmarkResult,
    EncryptedVector,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from opaque.shared.utils import (
    generate_random_vectors,
    normalize_vectors,
    compute_plaintext_similarity,
    top_k_indices,
    Timer,
)

__all__ = [
    "VectorDimension",
    "BenchmarkResult",
    "EncryptedVector",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "generate_random_vectors",
    "normalize_vectors",
    "compute_plaintext_similarity",
    "top_k_indices",
    "Timer",
]
