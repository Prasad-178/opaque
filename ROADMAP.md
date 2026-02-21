# Roadmap

## Current State

Opaque is a working privacy-preserving vector search system implemented in Go. The core pipeline is functional: CKKS homomorphic encryption, k-means clustering, AES-256-GCM blob encryption, and hierarchical search with decoy-based access pattern hiding.

### What Works Today

- **Library API** (`opaque.NewDB` → `Add` → `Build` → `Search`) for simple usage
- CKKS homomorphic encryption via Lattigo v5 (128-bit security)
- K-means clustering with k-means++ initialization
- AES-256-GCM encrypted blob storage (memory and file backends)
- Hierarchical search: HE centroid scoring -> decoy fetch -> local AES decrypt
- Batch HE via CKKS slot packing (64 centroids in 1 HE operation)
- Configurable HE engine worker pool (defaults to `min(NumCPU, 8)`)
- Redundant cluster assignment for improved boundary recall
- Multi-probe cluster selection to handle CKKS approximation noise
- Per-enterprise key isolation and configuration
- Token-based authentication service
- REST API server with auth middleware
- Development server with test data generation
- SIFT10K accuracy tests with real embeddings and ground truth

### Known Limitations

- ~~gRPC service registration is incomplete~~ (Done)
- Benchmarks run on a single machine (Apple M4 Pro) only

---

## Short Term

### ~~Library API~~ (Done)
Opaque is usable as a Go library with a clean API:

```go
db, _ := opaque.NewDB(opaque.Config{
    Dimension:  128,
    NumClusters: 64,
})

db.Add(ctx, "doc-1", vector1)
db.Add(ctx, "doc-2", vector2)
db.Build(ctx)

results, _ := db.Search(ctx, queryVector, 10)
```

See `go/opaque.go` for the full API and `go/README.md` for configuration reference.

### ~~Configurable Worker Pool~~ (Done)
`NewEnterpriseHierarchicalClientWithPoolSize()` accepts a pool size parameter. The library API defaults to `min(runtime.NumCPU(), 8)`.

### ~~Complete Remote Client~~ (Done)
Decoy generation and multi-probe cluster selection in `remote_client.go` now matches the local `EnterpriseHierarchicalClient` implementation. Shared `generateDecoySupers` function eliminates code duplication.

### ~~CI/CD Benchmarks~~ (Done)
GitHub Actions CI pipeline runs lint + tests on every push/PR. Weekly benchmark workflow tracks micro-benchmarks, SIFT10K accuracy, and 100K vector performance. Local `Makefile` provides convenience targets.

---

## Medium Term

### ~~gRPC Service~~ (Done)
Complete gRPC server implementation with all 7 RPC handlers (RegisterKey, GetPlanes, GetCandidates, ComputeScores, ComputeScoresStream, Search, HealthCheck). Includes recovery/logging interceptors, optional TLS, and full integration tests. See `go/pkg/grpcserver/`.

### ~~SIFT1M Benchmarks~~ (Done)
Benchmark suite for the full SIFT1M dataset (1 million 128-dim vectors). Includes accuracy test with Recall@1/10/100 against ground truth, and scaling test across 100K/250K/500K/1M subsets. Uses library API (`opaque.NewDB`). Download script at `scripts/download_sift1m.sh`, benchmark at `go/test/sift1m_benchmark_test.go` (build tag: `sift1m`). Runs weekly in CI with dataset caching.

### GPU Acceleration
Lattigo supports GPU acceleration for HE operations. Investigate and benchmark CUDA/Metal backends for further latency reduction.

### HNSW Index
Replace or complement k-means clustering with HNSW (Hierarchical Navigable Small World) for improved recall characteristics at scale.

### ~~PCA Dimensionality Reduction~~ (Done)
Optional PCA via SVD in `go/pkg/pca/`. Enabled with `Config.PCADimension`. Applied client-side before encryption — no privacy impact. 128D→64D achieves 77% variance retention with perfect self-match recall.

---

## Long Term

### Private Information Retrieval (PIR)
Replace decoy-based bucket fetching with cryptographic PIR for stronger access pattern privacy. The current decoy approach provides k-anonymity; PIR would provide cryptographic guarantees. This is a significant performance tradeoff that needs careful benchmarking.

### Key Rotation
Implement periodic key rotation for AES encryption keys and HE parameters without requiring full re-encryption of the database.

### Distributed Storage
Support distributed blob storage backends (e.g., Redis, S3) for horizontal scaling beyond single-node deployments.

### TEE Integration
Investigate Trusted Execution Environment (Intel SGX, AWS Nitro Enclaves) as an alternative or complement to pure-crypto approaches for specific deployment scenarios.
