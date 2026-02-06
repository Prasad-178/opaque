# Opaque Go

Privacy-preserving vector search with multiple tiers of protection, implemented in Go.

## Overview

Opaque Go provides flexible privacy levels for vector search, from query-private to fully data-private:

| Tier | Name | Server Sees | Use Case |
|------|------|-------------|----------|
| **Tier 1** | Query-Private | Vectors, encrypted query | Traditional vector DB with query privacy |
| **Tier 2** | Data-Private | Encrypted blobs, LSH buckets | Blockchain, zero-trust storage |
| **Tier 2.5** | Hierarchical Private | Encrypted blobs, HE centroids, bucket access | Maximum crypto privacy for query + data |

### Technologies Used

- **Lattigo BFV** for homomorphic encryption (128-bit security)
- **AES-256-GCM** for symmetric encryption (Tier 2)
- **Locality-Sensitive Hashing (LSH)** for fast candidate retrieval
- **gRPC** for efficient client-server communication

## Project Structure

```
go/
├── cmd/
│   ├── search-service/  # gRPC server
│   └── cli/             # CLI testing tool
├── pkg/
│   ├── crypto/          # Homomorphic encryption (Lattigo BFV)
│   ├── lsh/             # Locality-sensitive hashing
│   ├── client/          # Client SDK (Tier 1 + Tier 2 + Hierarchical)
│   ├── hierarchical/    # Hierarchical index (Tier 3)
│   ├── encrypt/         # Symmetric encryption (AES-GCM)
│   └── blob/            # Encrypted blob storage
├── internal/
│   ├── service/         # Search service implementation
│   ├── session/         # Session management
│   └── store/           # Vector storage (in-memory, Milvus)
├── api/
│   └── proto/           # Protocol buffer definitions
├── examples/
│   ├── search/          # Tier 1 example
│   ├── tier2/           # Tier 2 example
│   ├── hierarchical/    # Tier 3 (Hierarchical) example
│   └── demo/            # Combined demo
└── test/                # Integration tests & benchmarks
```

## Quick Start

### Build

```bash
go build ./...
```

### Run Tests

```bash
go test ./... -v
```

### Run Example

```bash
go run ./examples/search/main.go
```

### Run CLI

```bash
go run ./cmd/cli/main.go -bench
```

### Run Server

```bash
go run ./cmd/search-service/main.go -demo-vectors=1000
```

Health endpoints:
- `http://localhost:8080/healthz` - Health check
- `http://localhost:8080/readyz` - Readiness check

### Run Tier 2 Example

```bash
go run ./examples/tier2/main.go
```

## Tier 2: Data-Private Search

Tier 2 provides **full data privacy** - vectors are encrypted client-side before storage.
The server/storage never sees plaintext vectors.

### Quick Start

```go
import (
    "github.com/opaque/opaque/go/pkg/blob"
    "github.com/opaque/opaque/go/pkg/client"
    "github.com/opaque/opaque/go/pkg/encrypt"
)

// Create encryption key (derive from user password in production)
key, _ := encrypt.GenerateKey()
encryptor, _ := encrypt.NewAESGCM(key)

// Create storage backend (memory, file, or custom)
store := blob.NewMemoryStore()

// Create Tier 2 client
cfg := client.Tier2Config{
    Dimension: 128,
    LSHBits:   8,    // Fewer bits = larger buckets = more privacy
    LSHSeed:   42,
}
tier2Client, _ := client.NewTier2Client(cfg, encryptor, store)

// Insert vectors (encrypted automatically)
tier2Client.Insert(ctx, "doc-1", vector, nil)

// Search (decrypt + compute happens locally)
results, _ := tier2Client.Search(ctx, queryVector, 10)
```

### Privacy Features

```go
// Enable privacy enhancements
tier2Client.SetPrivacyConfig(client.DefaultPrivacyConfig())

// Search with timing obfuscation and decoy buckets
results, _ := tier2Client.SearchWithPrivacy(ctx, query, 10)

// Available configs:
// - DefaultPrivacyConfig()  - Balanced security/performance
// - HighPrivacyConfig()     - Maximum privacy (slower)
// - LowLatencyConfig()      - Optimized for speed
```

### Storage Backends

```go
// In-memory (for testing)
store := blob.NewMemoryStore()

// File-based (persistent)
store, _ := blob.NewFileStore("/path/to/data")

// Custom (implement blob.Store interface)
type Store interface {
    Put(ctx, blob) error
    Get(ctx, id) (*Blob, error)
    GetBucket(ctx, bucket) ([]*Blob, error)
    // ... more methods
}
```

### What the Storage Backend Sees

| Visible | Hidden |
|---------|--------|
| LSH bucket identifiers | Actual vector values |
| Encrypted blobs (ciphertext) | Query vectors |
| Access patterns (which buckets) | Similarity scores |
| Blob count per bucket | Which results you selected |

## Tier 2.5: Hierarchical Private Search

Tier 2.5 combines **query privacy** (HE) with **data privacy** (AES) using a three-level hierarchy:

```
Level 1: HE on super-bucket centroids (64 operations)
         ↓ Client privately selects top super-buckets
         ↓ SERVER NEVER SEES WHICH BUCKETS WERE SELECTED
Level 2: Decoy-based sub-bucket fetch
         ↓ Server can't distinguish real from decoy
         ↓ Optional: PIR for cryptographic guarantee
Level 3: Local AES decrypt + scoring
         ↓ All computation is client-side
```

### Key Benefits

- **~20x faster** than naive Tier 1 (100K vectors)
- **Query privacy**: Server never sees query vector (HE encrypted)
- **Data privacy**: Storage never sees vectors (AES-256-GCM)
- **Selection privacy**: Server doesn't know which super-buckets selected (client-side HE decrypt)
- **Per-enterprise isolation**: Each enterprise has secret LSH mapping (planned)

### Quick Start

```go
import (
    "github.com/opaque/opaque/go/pkg/blob"
    "github.com/opaque/opaque/go/pkg/client"
    "github.com/opaque/opaque/go/pkg/encrypt"
    "github.com/opaque/opaque/go/pkg/hierarchical"
)

// Build hierarchical index
key, _ := encrypt.GenerateKey()
cfg := hierarchical.DefaultConfig()
cfg.Dimension = 128

builder, _ := hierarchical.NewBuilder(cfg, key)
store := blob.NewMemoryStore()
idx, _ := builder.Build(ctx, ids, vectors, store)

// Create client and search
hClient, _ := client.NewHierarchicalClient(idx)
result, _ := hClient.Search(ctx, query, 10)

// Access timing breakdown
fmt.Printf("HE centroid scoring: %v\n", result.Timing.HECentroidScores)
fmt.Printf("Bucket fetch: %v\n", result.Timing.BucketFetch)
fmt.Printf("Local scoring: %v\n", result.Timing.LocalScoring)
fmt.Printf("Total: %v\n", result.Timing.Total)
```

### Run Hierarchical Example

```bash
go run ./examples/hierarchical/main.go
```

### Performance (100K vectors, 128D)

| Metric | Value |
|--------|-------|
| Total query time | ~700ms |
| HE operations | 64 (not 100K!) |
| Vectors decrypted | ~1500 |
| Speedup vs naive Tier 1 | ~20x |

### Privacy Guarantees

| What | Protected From | How |
|------|---------------|-----|
| Query vector | Server | HE encryption (BFV, 128-bit) |
| Super-bucket selection | Server | Client-side HE decryption |
| Sub-bucket interest | Server | Decoy buckets + shuffling |
| Vector values | Storage | AES-256-GCM encryption |
| Final scores | Everyone | Local computation only |

### What Server Sees vs Cannot See

| Server Sees | Server Cannot See |
|-------------|-------------------|
| HE(query) - encrypted blob | Query vector contents |
| 64 HE(scores) - computed blindly | Score values |
| Bucket IDs fetched | Which are real vs decoy |
| Encrypted blobs | Vector contents |
| Timing info | Final search results |

### Future Enhancements (Planned)

- **Per-enterprise LSH**: Each enterprise gets secret LSH hyperplanes
- **Authentication service**: Token-based key distribution (Option B)
- **Optional PIR**: For cryptographic bucket access privacy

See [docs/TIER_2_5_ARCHITECTURE.md](../docs/TIER_2_5_ARCHITECTURE.md) for complete architecture details.

## Performance

### Tier 1: Query-Private (Homomorphic Encryption)

| Operation | Go (Lattigo) | Python (LightPHE) | Speedup |
|-----------|--------------|-------------------|---------|
| Encryption | 5.2 ms | 1,912 ms | **367x** |
| Decryption | 0.8 ms | 140 ms | **175x** |
| Dot Product | 33 ms | 91 ms | **2.8x** |
| LSH Hash | 0.006 ms | ~1 ms | **166x** |

**Parallel Execution (Apple M4 Pro, 12 cores)**

| Operation | Sequential | Parallel | Speedup |
|-----------|------------|----------|---------|
| 20× Dot Products | 665 ms | 128 ms | **5.2x** |
| 20× Decryptions | 18 ms | 5.4 ms | **3.3x** |

**End-to-End Latency**

| Scenario | Go Parallel | Python | Improvement |
|----------|-------------|--------|-------------|
| Full Search (20 results) | **188 ms** | 6,592 ms | **35x faster** |

### Tier 2: Data-Private (AES-256-GCM)

| Operation | 128-dim | 256-dim |
|-----------|---------|---------|
| Insert (per vector) | 0.005 ms | 0.006 ms |
| Basic search | 0.01 ms | 0.01 ms |
| Multi-probe search (3 buckets) | 0.01 ms | 0.01 ms |
| Search with decoys | 0.09 ms | 0.09 ms |
| Privacy-enhanced search* | ~60 ms | ~60 ms |

*Privacy-enhanced search includes 50ms minimum timing obfuscation.

**Encryption Overhead**

| Dimension | Encrypt | Decrypt | Ciphertext Size |
|-----------|---------|---------|-----------------|
| 64 | 4.2 µs | 3.8 µs | 556 bytes |
| 128 | 5.1 µs | 4.6 µs | 1,068 bytes |
| 256 | 7.3 µs | 6.5 µs | 2,092 bytes |
| 1024 | 21.5 µs | 18.2 µs | 8,236 bytes |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed benchmark results and methodology.

## Architecture

### Tier 1: Query-Private Search Flow

1. **Client** computes LSH hash of query vector (locally)
2. **Server** returns candidate IDs by LSH similarity
3. **Client** encrypts query vector with homomorphic encryption
4. **Server** computes encrypted similarity scores
5. **Client** decrypts scores and returns top-k results

**Privacy**: Server never sees query vectors or similarity scores.

### Tier 2: Data-Private Search Flow

1. **Client** computes LSH hash of query (locally)
2. **Client** fetches encrypted blobs from matching bucket(s)
3. **Client** optionally fetches decoy buckets (to hide interest)
4. **Client** decrypts blobs and computes similarity (locally)
5. **Client** returns top-k results

**Privacy**: Server/storage never sees plaintext vectors.

### Privacy Comparison

| Aspect | Tier 1 | Tier 2 | Tier 2.5 |
|--------|--------|--------|----------|
| Query vectors | Hidden (HE) | Hidden (local) | Hidden (HE) |
| Database vectors | Visible | Hidden (AES) | Hidden (AES) |
| Similarity scores | Hidden (HE) | Hidden (local) | Hidden (HE + local) |
| Bucket selection | Visible | Visible | Hidden (client-side HE decrypt) |
| LSH buckets | Visible | Visible | Visible (but opaque with per-enterprise LSH) |
| Access patterns | Visible | Can be obfuscated | Obfuscated (decoys) |

## Configuration

### Client Config

```go
client.Config{
    Dimension:     128,    // Vector dimension
    LSHBits:       64,     // LSH hash bits
    MaxCandidates: 100,    // Max LSH candidates
    DecryptTopN:   20,     // Scores to decrypt
}
```

### Server Config

```go
service.Config{
    LSHNumBits:          128,
    LSHDimension:        128,
    LSHSeed:             42,
    MaxSessionTTL:       24 * time.Hour,
    MaxConcurrentScores: 16,
}
```

## Dependencies

- [Lattigo v5](https://github.com/tuneinsight/lattigo) - Homomorphic encryption
- [gRPC-Go](https://github.com/grpc/grpc-go) - RPC framework

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full product roadmap.

### Completed

- [x] Tier 1: Query-private search with homomorphic encryption
- [x] Tier 2: Data-private search with AES-256-GCM
- [x] Tier 3: Hierarchical private search (HE centroids + AES vectors)
- [x] Privacy enhancements (timing obfuscation, decoy buckets, shuffling)
- [x] Combined Tier 1 + Tier 2 + Tier 3 demos
- [x] File-based persistent storage
- [x] Comprehensive benchmarks

### In Progress

- [ ] Milvus vector store integration
- [ ] Full gRPC service implementation

### Future

- [ ] Enclave-private search (AWS Nitro)
- [ ] IPFS/blockchain storage backends
- [ ] Observability (metrics, tracing)

## License

MIT
