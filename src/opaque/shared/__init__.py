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
from opaque.shared.reduction import (
    PCAReducer,
    determine_optimal_dimension,
)

# Embeddings are optional (requires sentence-transformers)
try:
    from opaque.shared.embeddings import (
        EmbeddingModel,
        load_sample_dataset,
        create_embedded_database,
    )
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False

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
    "PCAReducer",
    "determine_optimal_dimension",
]

if _HAS_EMBEDDINGS:
    __all__.extend([
        "EmbeddingModel",
        "load_sample_dataset",
        "create_embedded_database",
    ])
