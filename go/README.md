# Opaque Go

Privacy-preserving vector search implemented in Go.

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

```bash
# Core packages (fast, ~10s)
go test ./pkg/encrypt/... ./pkg/crypto/... ./pkg/lsh/... ./pkg/cluster/... ./pkg/blob/... ./pkg/cache/...

# SIFT10K accuracy (real dataset, ~3 min)
go test -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

# 100K benchmark with recall (synthetic, ~8 min)
go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m

# Core crypto micro-benchmarks
go test -bench=. -benchmem ./pkg/crypto/... ./pkg/lsh/...
```

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
