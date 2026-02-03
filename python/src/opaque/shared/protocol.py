"""
Protocol definitions for client-server communication.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum


class VectorDimension(Enum):
    """Supported vector dimensions."""
    OPENAI_SMALL = 1536      # text-embedding-3-small
    OPENAI_LARGE = 3072      # text-embedding-3-large
    REDUCED_256 = 256        # After PCA reduction
    REDUCED_128 = 128        # After aggressive PCA reduction


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    operation: str
    dimension: int
    num_operations: int
    total_time_seconds: float
    avg_time_per_op_ms: float
    throughput_ops_per_sec: float
    memory_mb: Optional[float] = None
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"Benchmark: {self.operation}\n"
            f"  Dimension: {self.dimension}\n"
            f"  Operations: {self.num_operations}\n"
            f"  Total time: {self.total_time_seconds:.3f}s\n"
            f"  Avg per op: {self.avg_time_per_op_ms:.3f}ms\n"
            f"  Throughput: {self.throughput_ops_per_sec:.2f} ops/s\n"
            f"  Notes: {self.notes}"
        )


@dataclass
class EncryptedVector:
    """
    Container for an encrypted vector.

    The actual encrypted data is stored as a LightPHE EncryptedTensor,
    but this wrapper provides metadata for protocol handling.
    """
    id: str
    dimension: int
    encrypted_data: Any  # LightPHE EncryptedTensor
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchRequest:
    """
    Request from client to server for vector search.

    Stage 1 (Coarse): obfuscated_query for LSH lookup
    Stage 2 (Fine): encrypted_query for PHE scoring
    """
    request_id: str
    # Stage 1: LSH lookup
    obfuscated_query: Optional[List[float]] = None
    num_candidates: int = 1000

    # Stage 2: PHE scoring
    encrypted_query: Optional[Any] = None  # EncryptedTensor
    candidate_ids: Optional[List[str]] = None

    # Public key for server to verify/use
    public_key: Optional[dict] = None


@dataclass
class SearchResponse:
    """
    Response from server to client.

    Stage 1: Returns candidate IDs from LSH
    Stage 2: Returns encrypted scores for candidates
    """
    request_id: str
    stage: int  # 1 or 2

    # Stage 1 response
    candidate_ids: Optional[List[str]] = None

    # Stage 2 response
    encrypted_scores: Optional[List[Any]] = None  # List of encrypted dot products
    vector_ids: Optional[List[str]] = None

    # Timing info
    server_time_ms: float = 0.0

    # Error handling
    error: Optional[str] = None


@dataclass
class SearchResult:
    """Result of a privacy-preserving search."""
    vector_id: str
    score: float
    rank: int
