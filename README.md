# Project Opaque

Privacy-preserving vector search using Partial Homomorphic Encryption (PHE) and Locality Sensitive Hashing (LSH).

## Overview

Project Opaque allows a client to query a remote vector database such that:
- **The server never sees the raw query vector**
- **The client never downloads the full database**

This is achieved through a two-stage "funnel" approach:
1. **Coarse Stage (LSH)**: Fast approximate filtering to get candidate vectors
2. **Fine Stage (PHE)**: Cryptographically secure scoring using Paillier encryption

## Installation

```bash
# Clone and setup
cd opaque
uv sync

# Or install dependencies manually
uv add lightphe numpy faiss-cpu scikit-learn tqdm fastapi uvicorn pydantic httpx
```

## Quick Start

```bash
# Phase 1: Benchmark PHE performance
uv run python scripts/benchmark_phe.py --quick

# Phase 2: End-to-end blind search (with FastAPI server)
uv run python scripts/demo_blind_search.py -n 100 -d 128

# Phase 3: Full funnel pipeline
uv run python scripts/demo_full_pipeline.py --tiny

# Benchmark parallelization
uv run python scripts/benchmark_parallel.py --quick

# Demo with real embeddings (requires sentence-transformers)
uv run python scripts/demo_real_embeddings.py --tiny

# Run tests
uv run pytest tests/ -v
```

## New Features

### Real Embeddings Support
Use actual sentence embeddings from sentence-transformers instead of random vectors:
```python
from opaque.shared.embeddings import EmbeddingModel, create_embedded_database

model = EmbeddingModel("all-MiniLM-L6-v2")  # 384 dimensions
embeddings, ids, dim = create_embedded_database(texts, positive_shift=True)
```

### Parallel Decryption (Multiprocessing)
True parallelism using ProcessPoolExecutor for ~3.6x faster decryption:
```python
from opaque.client.crypto import CryptoClient, parallel_decrypt_multiprocess

crypto = CryptoClient(key_size=2048)
keys = crypto.get_keys_dict()

# Parallel decryption with multiprocessing
scores = parallel_decrypt_multiprocess(
    encrypted_scores, keys=keys, num_workers=4
)
```

### Parallel Search
Enable multiprocessing in search operations:
```python
results, timing = search_client.funnel_search(
    query, lsh_search_fn, server_compute_fn,
    multiprocess=True,  # ~3.6x faster decryption
    num_workers=4
)
```

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│     CLIENT      │                    │     SERVER      │
├─────────────────┤                    ├─────────────────┤
│                 │  1. Public Key     │                 │
│  Generate Keys  │ ────────────────►  │  Store Key      │
│  (Paillier)     │                    │                 │
│                 │                    │                 │
│                 │  2. LSH Query      │                 │
│  Query Vector   │ ────────────────►  │  LSH Index      │
│       Q         │                    │  (Faiss)        │
│                 │  3. Candidates     │                 │
│                 │ ◄────────────────  │  Top 1000 IDs   │
│                 │                    │                 │
│  Encrypt(Q)     │  4. E(Q)           │                 │
│                 │ ────────────────►  │  Compute        │
│                 │                    │  E(Q) @ V_i     │
│                 │  5. E(Scores)      │                 │
│  Decrypt        │ ◄────────────────  │  (Encrypted)    │
│  ArgMax         │                    │                 │
└─────────────────┘                    └─────────────────┘
```

## Project Structure

```
opaque/
├── src/opaque/
│   ├── client/
│   │   ├── crypto.py      # Paillier encryption (LightPHE wrapper)
│   │   └── search.py      # Search orchestration
│   ├── server/
│   │   ├── api.py         # FastAPI server
│   │   ├── compute.py     # PHE dot product computation
│   │   └── index.py       # LSH index (Faiss + NumPy)
│   └── shared/
│       ├── protocol.py    # Data models
│       ├── reduction.py   # PCA dimensionality reduction
│       └── utils.py       # Utilities
├── scripts/
│   ├── benchmark_phe.py       # Performance benchmarks
│   ├── demo_blind_search.py   # Phase 2 demo
│   └── demo_full_pipeline.py  # Phase 3 demo
└── tests/                     # 23 test cases
```

## Performance Results

### Benchmark Summary (2048-bit Paillier, macOS)

| Operation | 128D | 256D |
|-----------|------|------|
| Encryption | 2.3s | 4.0s |
| 100 Dot Products | 8.6s | 17.2s |
| 100 Decryptions | 15.1s | 13.5s |

### End-to-End Latency

| Phase | Database | Dimension | Candidates | Total Time |
|-------|----------|-----------|------------|------------|
| Phase 2 (Blind) | 100 | 128 | 100 | ~24s |
| Phase 3 (Funnel) | 1000 | 128 | 100 | ~24s |

**Note**: Current performance is 8-10x slower than the 3s target due to LightPHE limitations.

## API Reference

### Client

```python
from opaque.client import CryptoClient, SearchClient

# Generate keys
crypto = CryptoClient(key_size=2048, precision=5)

# Encrypt query
encrypted_query = crypto.encrypt_vector(query_vector)

# Decrypt scores
scores = crypto.decrypt_scores(encrypted_scores)
```

### Server

```python
from opaque.server import ComputeEngine, LSHIndex, VectorStore

# Build index
lsh = LSHIndex(dimension=128, nbits=256)
lsh.add(vectors, ids)

# Search candidates
candidates, distances = lsh.search(query, k=1000)

# Compute encrypted scores
engine = ComputeEngine(VectorStore(vectors, ids))
scores, ids, time_ms = engine.compute_encrypted_scores(encrypted_query, candidates)
```

## Limitations & Known Issues

### 1. LightPHE Positive Value Requirement
LightPHE requires all values to be positive for element-wise multiplication. Vectors must be shifted to positive range after normalization/PCA.

### 2. Performance
- **Encryption**: ~2-4s per vector (128-256 dim)
- **Decryption**: ~135-150ms per score
- **PHE Dot Product**: ~86-172ms per operation

Current total latency (~24s) significantly exceeds the 3s target.

### 3. Ciphertext Size
2048-bit Paillier ciphertexts are large. A 128-dim encrypted vector serializes to ~270KB.

## Production Considerations

### What Works Well
- **Privacy guarantees** are mathematically sound (Paillier semantic security)
- **LSH filtering** is fast (<1ms for 1000 vectors)
- **Architecture** is clean and modular
- **FastAPI server** provides real client-server separation

### What Needs Work
- **Latency**: 24s is not acceptable for interactive use
- **Scalability**: In-memory only, no persistence
- **Ciphertext blowup**: Large network transfer for encrypted queries

### Recommendations for Production

1. **Use a faster PHE library**: Consider SEAL, HElib, or Lattigo (Go)
2. **Reduce dimensionality aggressively**: 64-dim may be necessary
3. **Parallelize PHE operations**: Use multiprocessing for dot products
4. **Consider alternative cryptographic approaches**:
   - Functional Encryption for inner products
   - Secure Multi-Party Computation (MPC)
   - Trusted Execution Environments (TEE)

## FAQ

### Why is it so slow?
Paillier encryption involves modular exponentiation with 2048-bit numbers. Each encrypted element requires this operation, and dot products require element-wise operations followed by homomorphic addition.

### Why not use FHE (Fully Homomorphic Encryption)?
FHE (e.g., TFHE, CKKS) would be even slower for this use case. Paillier's additive homomorphism is actually well-suited for dot products.

### Can this scale to millions of vectors?
The LSH stage can scale to millions of vectors. The bottleneck is the PHE stage, which is limited by cryptographic operations. With 1000 candidates, expect ~100s latency.

### Is this actually secure?
Yes. Paillier encryption is semantically secure under the Decisional Composite Residuosity Assumption (DCRA). The server learns nothing about the query vector or similarity scores.

### What about the LSH stage leaking information?
The LSH query reveals which "bucket" the query falls into. This is a privacy/performance tradeoff. For stronger privacy, you could encrypt the LSH query too, but this would be slower.

## License

MIT

## References

- [LightPHE](https://github.com/serengil/LightPHE) - Python PHE library
- [Paillier Cryptosystem](https://en.wikipedia.org/wiki/Paillier_cryptosystem)
- [Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
- [Faiss](https://github.com/facebookresearch/faiss) - Vector similarity search
