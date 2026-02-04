# Project Opaque

Privacy-preserving vector search using Homomorphic Encryption and Locality Sensitive Hashing.

**Query a remote vector database without revealing your query. The server computes on encrypted data and never sees what you're searching for.**

## Performance Highlights

| Metric | Value |
|--------|-------|
| Query Latency (100K vectors) | **66 ms** |
| Encryption (128D vector) | **5 ms** |
| Queries Per Second | **15.1** |
| Cryptographic Security | **128-bit** (BFV/RLWE) |

Built in Go using [Lattigo](https://github.com/tuneinsight/lattigo) for homomorphic encryption.

## How It Works

Project Opaque allows a client to query a remote vector database such that:
- **The server never sees the raw query vector** (encrypted with homomorphic encryption)
- **The server never sees the similarity scores** (computed on encrypted data)
- **The client never downloads the full database** (only receives top candidates)

This is achieved through a two-stage "funnel" approach:
1. **Coarse Stage (LSH)**: Fast approximate filtering using locality-sensitive hashing (~5ms)
2. **Fine Stage (HE)**: Cryptographically secure scoring using homomorphic encryption (~56ms)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           CLIENT                                  │
├──────────────────────────────────────────────────────────────────┤
│  1. Generate BFV keypair (128-bit security)                      │
│  2. Compute LSH hash locally (no server involvement)             │
│  3. Mask hash with session key (prevents query correlation)      │
│  4. Encrypt query vector → E(query)                              │
│  5. Decrypt scores → rank results (server never sees ranking)    │
└─────────────────────────────┬────────────────────────────────────┘
                              │ gRPC (TLS)
┌─────────────────────────────▼────────────────────────────────────┐
│                           SERVER                                  │
├──────────────────────────────────────────────────────────────────┤
│  1. Store client's public key (session-based)                    │
│  2. LSH lookup → return candidate IDs                            │
│  3. Homomorphic dot product: E(query) · vector = E(score)        │
│  4. Return encrypted scores (never sees plaintext!)              │
└──────────────────────────────────────────────────────────────────┘
```

### Query Flow (100K vectors)

```
User Query: "side effects of ibuprofen"
    │
    ▼ Text → Embedding (15ms, local ONNX)
    │
    ├─► LSH Hash (local) ──────────────────► Server: LSH Lookup
    │                                              │
    │                              ◄── 200 candidate IDs (5ms)
    │
    ├─► Encrypt Query (5ms) ───────────────► Server: 10× HE Dot Products
    │                                              │ (parallel, 56ms)
    │                              ◄── Encrypted Scores
    │
    ├─► Decrypt & Rank (3ms, local)
    │
    ▼ Return Top Results

Total: ~66ms (local) | ~134ms (with network)
```

## Quick Start

```bash
cd go

# Run benchmarks
go test -v ./test/... -run TestOptimizedTwoStageSearch

# Run example search
go run ./examples/search/main.go

# Start gRPC server (demo mode with 1000 vectors)
go run ./cmd/search-service/main.go -demo-vectors 1000
```

## Project Structure

```
opaque/
├── go/
│   ├── pkg/
│   │   ├── crypto/               # Lattigo BFV encryption
│   │   ├── lsh/                  # Locality-sensitive hashing
│   │   └── client/               # Client SDK
│   ├── internal/
│   │   ├── service/              # Search service
│   │   ├── session/              # Session management
│   │   └── store/                # Vector storage
│   ├── api/proto/                # gRPC definitions
│   ├── cmd/
│   │   ├── search-service/       # Server entry point
│   │   └── cli/                  # CLI tool
│   ├── test/                     # Benchmarks & tests
│   └── BENCHMARKS.md             # Detailed performance data
│
├── python/                       # Reference implementation (experimental)
└── README.md
```

## Benchmark Results

### Core Operations

| Operation | Latency |
|-----------|---------|
| Encryption (128D) | 5.2 ms |
| Decryption (128D) | 0.8 ms |
| HE Dot Product | 33 ms |
| LSH Hash | 0.006 ms |

### End-to-End Latency (100K Vectors)

| Stage | Time |
|-------|------|
| Text → Embedding (ONNX) | 15 ms |
| Network RTT (client → server) | 25 ms |
| LSH Search (200 candidates) | 5 ms |
| Query Encryption | 5 ms |
| 10× HE Dot Products (parallel) | 56 ms |
| Network RTT (server → client) | 25 ms |
| Decrypt (client-side) | 3 ms |
| **Total End-to-End** | **~134 ms** |

### Throughput

| Configuration | QPS | Latency p50 |
|---------------|-----|-------------|
| Single node | 15.1 | 66 ms |
| 3-node cluster | ~45 | 66 ms |

## Key Features

### Two-Stage Optimized Search
- **Stage 1**: LSH retrieves 200 candidates (~5ms, cheap)
- **Stage 2**: Rank by Hamming distance, HE score top 10 (~56ms)
- Result: 1.85x faster with 10-15% better accuracy

### Hash Masking for Privacy
```go
masked_hash = LSH_hash XOR session_key
```
Prevents query correlation across sessions with zero computational overhead.

### Parallel Homomorphic Operations
- Worker pool with multiple crypto engines
- 10 parallel HE dot products: 330ms → 56ms (6x speedup)
- Efficient goroutine-based concurrency

### Client-Side Ranking
Server returns encrypted scores; client decrypts and ranks locally. Server never learns which result was most relevant.

## API Reference

### Go Client

```go
import (
    "github.com/opaque/opaque/go/pkg/client"
    "github.com/opaque/opaque/go/pkg/crypto"
    "github.com/opaque/opaque/go/pkg/lsh"
)

// Create client with encryption capabilities
engine, _ := crypto.NewClientEngine()
lshIndex := lsh.NewIndex(128, 64, 100000) // dim, bits, capacity

c := client.New(engine, lshIndex)

// Search
query := []float64{0.1, 0.2, ...} // 128D embedding
hash := c.ComputeMaskedLSHHash(query, sessionKey)
encryptedQuery := c.EncryptQuery(query)

// Server returns encrypted scores
scores := c.DecryptScores(encryptedScores)
```

### Server

```go
import (
    "github.com/opaque/opaque/go/internal/service"
    "github.com/opaque/opaque/go/internal/store"
)

// Create search service
vectorStore := store.NewMemoryStore()
svc := service.NewSearchService(vectorStore, 12) // 12 worker cores

// Handle search request
candidates := svc.GetCandidates(sessionID, maskedHash, 200)
encryptedScores := svc.ComputeScores(sessionID, encryptedQuery, candidates[:10])
```

## Security Model

### Cryptographic Guarantees
- **BFV Encryption**: 128-bit security level (Lattigo)
- **Semantic Security**: Ciphertexts reveal nothing about plaintexts
- **No Query Leakage**: Server sees encrypted query, computes encrypted scores

### What the Server Learns
- LSH bucket (approximate region, not exact query)
- Which vector IDs were scored
- Timing information

### What the Server Does NOT Learn
- Raw query vector
- Similarity scores
- Final ranking order

### Threat Model
- Server is honest-but-curious (follows protocol but may analyze data)
- Client trusts its local environment
- Network is encrypted (gRPC/TLS)

## Production Deployment

### Requirements
- Go 1.21+
- 4+ CPU cores (for parallel HE operations)
- ~50MB RAM per 1000 vectors

### Scaling
- **Horizontal**: Linear scaling with more nodes (shared LSH index)
- **Vertical**: More cores = faster parallel HE (up to ~12 cores)

### Recommended Configuration
```yaml
workers: 12              # Match CPU cores
lsh_candidates: 200      # Coarse stage
he_candidates: 10        # Fine stage (scored with HE)
session_ttl: 24h         # Key rotation
```

## Optimizations Implemented

- [x] Two-stage search (200 LSH → 10 HE)
- [x] Hash masking (XOR with session key)
- [x] Worker pool (parallel crypto engines)
- [x] Client-side ranking
- [x] Fixed-point encoding for BFV

## Future Improvements

- [ ] HNSW index (95%+ recall vs 70-80% with LSH)
- [ ] GPU acceleration (Lattigo supports CUDA)
- [ ] PCA dimensionality reduction (128D → 64D)
- [ ] Redis session store (for distributed deployments)
- [ ] Streaming gRPC for large result sets

## FAQ

### Can this scale to millions of vectors?
Yes. LSH scales to millions with sub-millisecond lookup. The bottleneck is HE operations, but with two-stage filtering (200 → 10), you only compute 10 HE dot products regardless of database size.

### Is this actually secure?
Yes. BFV encryption is semantically secure under the Ring Learning With Errors (RLWE) assumption. The server computes on ciphertexts and never sees plaintexts.

### What about the LSH hash leaking information?
The LSH hash reveals which approximate "bucket" the query falls into. This is a privacy/performance tradeoff. Hash masking prevents cross-session correlation, and the bucket reveals only coarse similarity, not the actual query.

### How does this compare to Trusted Execution Environments (TEE)?
TEEs (SGX, TrustZone) offer different tradeoffs:
- **TEE**: Lower latency, but requires hardware trust
- **HE**: Higher latency, but cryptographic guarantees with no hardware trust

## References

- [Lattigo](https://github.com/tuneinsight/lattigo) - Go HE library (BFV/CKKS)
- [LightPHE](https://github.com/serengil/LightPHE) - Python PHE library
- [BFV Scheme](https://eprint.iacr.org/2012/144) - Fan-Vercauteren encryption
- [Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)

## License

MIT
