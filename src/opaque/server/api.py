"""
FastAPI server for privacy-preserving vector search.

Endpoints:
- POST /keys - Register client public key
- POST /search/blind - PHE blind scoring (Phase 2)
- POST /search/stage1 - LSH candidate retrieval (Phase 3)
- POST /search/stage2 - PHE scoring on candidates (Phase 3)
"""
import pickle
import base64
from typing import List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from opaque.server.compute import ComputeEngine, VectorStore, create_mock_database
from opaque.shared.utils import Timer


# Pydantic models for API
class PublicKeyRequest(BaseModel):
    """Request to register a client's public key."""
    client_id: str
    public_key: dict


class BlindSearchRequest(BaseModel):
    """Request for blind search (Phase 2)."""
    client_id: str
    encrypted_query_b64: str = Field(..., description="Base64-encoded pickled EncryptedTensor")


class Stage1Request(BaseModel):
    """Request for LSH candidate retrieval (Phase 3)."""
    query: List[float]
    num_candidates: int = 1000


class Stage2Request(BaseModel):
    """Request for PHE scoring on candidates (Phase 3)."""
    client_id: str
    encrypted_query_b64: str
    candidate_ids: List[str]


class SearchResponse(BaseModel):
    """Response with encrypted scores."""
    encrypted_scores_b64: List[str] = Field(..., description="Base64-encoded pickled scores")
    vector_ids: List[str]
    server_time_ms: float
    num_results: int


class Stage1Response(BaseModel):
    """Response with LSH candidates."""
    candidate_ids: List[str]
    distances: List[float]
    server_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    num_vectors: int
    dimension: int
    registered_clients: int


# Server state
class ServerState:
    """Server state container."""
    def __init__(self):
        self.vector_store: Optional[VectorStore] = None
        self.compute_engine: Optional[ComputeEngine] = None
        self.client_keys: dict = {}  # client_id -> public_key
        self.lsh_index: Optional[Any] = None  # Added in Phase 3


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize server on startup."""
    # Create default mock database for testing
    print("Initializing server with mock database...")
    state.vector_store = create_mock_database(num_vectors=100, dimension=128, seed=42)
    state.compute_engine = ComputeEngine(state.vector_store)
    print(f"Server ready: {len(state.vector_store)} vectors, dim={state.vector_store.dimension}")
    yield
    # Cleanup on shutdown
    print("Server shutting down...")


app = FastAPI(
    title="Project Opaque",
    description="Privacy-preserving vector search API",
    version="0.1.0",
    lifespan=lifespan,
)


def serialize_encrypted(obj: Any) -> str:
    """Serialize encrypted object to base64 string."""
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')


def deserialize_encrypted(b64_str: str) -> Any:
    """Deserialize encrypted object from base64 string."""
    return pickle.loads(base64.b64decode(b64_str))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        num_vectors=len(state.vector_store) if state.vector_store else 0,
        dimension=state.vector_store.dimension if state.vector_store else 0,
        registered_clients=len(state.client_keys),
    )


@app.post("/keys")
async def register_public_key(request: PublicKeyRequest):
    """Register a client's public key."""
    state.client_keys[request.client_id] = request.public_key
    return {
        "status": "registered",
        "client_id": request.client_id,
    }


@app.post("/search/blind", response_model=SearchResponse)
async def blind_search(request: BlindSearchRequest):
    """
    Perform blind search (Phase 2).

    The server computes encrypted dot products for ALL vectors.
    """
    if state.compute_engine is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    if request.client_id not in state.client_keys:
        raise HTTPException(status_code=400, detail="Client not registered")

    # Deserialize encrypted query
    try:
        encrypted_query = deserialize_encrypted(request.encrypted_query_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to deserialize query: {e}")

    # Compute encrypted scores
    encrypted_scores, vector_ids, time_ms = state.compute_engine.compute_encrypted_scores(
        encrypted_query,
        verbose=False,
    )

    # Serialize encrypted scores
    encrypted_scores_b64 = [serialize_encrypted(s) for s in encrypted_scores]

    return SearchResponse(
        encrypted_scores_b64=encrypted_scores_b64,
        vector_ids=vector_ids,
        server_time_ms=time_ms,
        num_results=len(vector_ids),
    )


@app.post("/search/stage1", response_model=Stage1Response)
async def stage1_lsh_search(request: Stage1Request):
    """
    Stage 1: LSH candidate retrieval (Phase 3).

    Returns candidate IDs without PHE computation.
    """
    if state.lsh_index is None:
        raise HTTPException(
            status_code=501,
            detail="LSH index not initialized. Use /search/blind for Phase 2."
        )

    import numpy as np
    query = np.array(request.query, dtype=np.float32)

    with Timer() as t:
        candidate_ids, distances = state.lsh_index.search(query, request.num_candidates)

    return Stage1Response(
        candidate_ids=candidate_ids,
        distances=distances.tolist() if hasattr(distances, 'tolist') else list(distances),
        server_time_ms=t.elapsed_ms,
    )


@app.post("/search/stage2", response_model=SearchResponse)
async def stage2_phe_search(request: Stage2Request):
    """
    Stage 2: PHE scoring on candidates (Phase 3).

    Only scores the specified candidate vectors.
    """
    if state.compute_engine is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    if request.client_id not in state.client_keys:
        raise HTTPException(status_code=400, detail="Client not registered")

    # Deserialize encrypted query
    try:
        encrypted_query = deserialize_encrypted(request.encrypted_query_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to deserialize query: {e}")

    # Compute encrypted scores for candidates only
    encrypted_scores, vector_ids, time_ms = state.compute_engine.compute_encrypted_scores(
        encrypted_query,
        vector_ids=request.candidate_ids,
        verbose=False,
    )

    # Serialize encrypted scores
    encrypted_scores_b64 = [serialize_encrypted(s) for s in encrypted_scores]

    return SearchResponse(
        encrypted_scores_b64=encrypted_scores_b64,
        vector_ids=vector_ids,
        server_time_ms=time_ms,
        num_results=len(vector_ids),
    )


@app.post("/database/init")
async def init_database(num_vectors: int = 100, dimension: int = 128, seed: int = 42):
    """Initialize/reset the database with random vectors."""
    state.vector_store = create_mock_database(num_vectors, dimension, seed)
    state.compute_engine = ComputeEngine(state.vector_store)
    return {
        "status": "initialized",
        "num_vectors": num_vectors,
        "dimension": dimension,
    }


def create_app(
    num_vectors: int = 100,
    dimension: int = 128,
    seed: int = 42,
) -> FastAPI:
    """
    Create and configure the FastAPI app.

    For programmatic use in tests and demos.
    """
    state.vector_store = create_mock_database(num_vectors, dimension, seed)
    state.compute_engine = ComputeEngine(state.vector_store)
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the server directly."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
