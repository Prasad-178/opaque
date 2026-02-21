# Opaque Go

Privacy-preserving vector search implemented in Go.

## Library API

The simplest way to use Opaque is through the top-level library API:

```go
import opaque "github.com/opaque/opaque/go"

// Create a database with default settings
db, err := opaque.NewDB(opaque.Config{
    Dimension:   128,  // Required: vector dimension
    NumClusters: 64,   // Optional: defaults to 64
})
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Add vectors (buffer phase)
db.Add(ctx, "doc-1", vector1)
db.Add(ctx, "doc-2", vector2)
// or bulk: db.AddBatch(ctx, ids, vectors)

// Build the index (expensive: k-means + HE engine init)
if err := db.Build(ctx); err != nil {
    log.Fatal(err)
}

// Search (concurrent-safe after Build)
results, err := db.Search(ctx, queryVector, 10)
for _, r := range results {
    fmt.Printf("  %s: %.4f\n", r.ID, r.Score)
}
```

### Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `Dimension` | (required) | Vector dimension |
| `NumClusters` | 64 | K-means clusters. More = faster search, less privacy |
| `TopClusters` | NumClusters/2 | Clusters probed per search. More = better recall |
| `NumDecoys` | 8 | Decoy clusters for access pattern hiding |
| `WorkerPoolSize` | min(NumCPU, 8) | Parallel HE engines (~50MB each) |
| `Storage` | Memory | `opaque.Memory` or `opaque.File` |
| `StoragePath` | "" | Directory for file storage |
| `ProbeThreshold` | 0.95 | Multi-probe inclusion threshold |
| `RedundantAssignments` | 1 | Clusters per vector (2 = better boundary recall, 2x storage) |

### Lifecycle

```
NewDB → Add/AddBatch → Build → Search (concurrent)
                                  ↓
                              Rebuild (after adding more vectors)
```

## Overview

The Go implementation contains the full search pipeline:

- **Lattigo v5 CKKS** for homomorphic encryption (128-bit security)
- **AES-256-GCM** for symmetric vector encryption
- **K-means clustering** with k-means++ initialization for hierarchical indexing
- **CKKS slot packing** for batch HE operations (64 centroids in 1 operation)
- **Per-enterprise isolation** with independent keys and configurations

## Project Structure

```
go/
├── opaque.go                # Library API (NewDB, Add, Build, Search)
├── cmd/
│   ├── devserver/         # Development server with test data
│   ├── search-service/    # Production server entry point
│   └── cli/               # CLI benchmarking tool
├── pkg/
│   ├── crypto/            # CKKS homomorphic encryption (Lattigo v5)
│   ├── encrypt/           # AES-256-GCM symmetric encryption
│   ├── lsh/               # Locality-sensitive hashing
│   ├── cluster/           # K-means clustering
│   ├── blob/              # Encrypted blob storage (memory + file)
│   ├── cache/             # Centroid cache + batch CKKS packing
│   ├── hierarchical/      # Index builder (K-means + encryption)
│   ├── client/            # Search client (standard + batch)
│   ├── auth/              # Authentication service
│   ├── enterprise/        # Per-enterprise config management
│   ├── server/            # REST API server
│   └── embeddings/        # Dataset loaders (SIFT, synthetic)
├── internal/
│   ├── service/           # Search service (LSH + HE)
│   ├── session/           # Session management with TTL
│   └── store/             # Vector storage interface
├── api/proto/             # gRPC protobuf definitions
├── examples/              # Usage examples
├── test/                  # Integration benchmarks
└── BENCHMARKS.md          # Performance data
```

## Running Tests

A `Makefile` is provided for common operations:

```bash
make test-fast    # go test -short ./...
make test         # go test ./...
make lint         # go vet ./...
make test-bench   # Micro-benchmarks (crypto, LSH)
make test-sift    # SIFT10K accuracy test
make test-100k    # 100K vector benchmark
make test-sift1m  # SIFT1M benchmark (requires download, ~5 min)
```

Or run directly:

```bash
# Library API tests (fast, ~5s)
go test -v -run TestBuildAndSearch ./...

# Core packages (fast, ~10s)
go test ./pkg/encrypt/... ./pkg/crypto/... ./pkg/lsh/... ./pkg/cluster/... ./pkg/blob/... ./pkg/cache/...

# SIFT10K accuracy (real dataset, ~3 min)
go test -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

# 100K benchmark with recall (synthetic, ~8 min)
go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m

# SIFT1M benchmark (real dataset, ~5 min; download first)
../scripts/download_sift1m.sh
go test -tags sift1m -v -run TestSIFT1MAccuracy ./test/ -timeout 45m

# Core crypto micro-benchmarks
go test -bench=. -benchmem ./pkg/crypto/... ./pkg/lsh/...
```

## CI/CD

GitHub Actions runs automatically on push/PR:
- **CI** (`.github/workflows/ci.yml`): lint, short tests, core tests, crypto tests, API integration tests
- **Benchmarks** (`.github/workflows/benchmarks.yml`): weekly + manual trigger for micro-benchmarks, SIFT10K accuracy, 100K performance, and SIFT1M scaling

## Development Server

```bash
go run ./cmd/devserver/main.go
```

Starts a REST API server with:
- In-memory blob storage
- Auto-generated test enterprise and user
- Auth endpoints (login, refresh)
- Blob retrieval endpoints

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance data.
