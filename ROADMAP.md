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
- SIFT1M benchmarks (1M vectors, 96.8% Recall@10 at 6% probe, ~500ms query latency)
- Optional PCA dimensionality reduction (client-side, no privacy impact)
- gRPC service with full CKKS integration and streaming support

### Known Limitations

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
Benchmark suite for the full SIFT1M dataset (1 million 128-dim vectors) with production-realistic configs. Ground truth computed via brute-force cosine similarity over all 1M vectors. Key results (128 clusters, 8 decoys):

| Config | Probe % | Recall@1 | Recall@10 | Avg Query |
|--------|---------|----------|-----------|-----------|
| strict-4 | 3.1% | 82.0% | 79.4% | 274ms |
| strict-8 | 6.2% | 98.0% | 96.8% | 508ms |
| strict-16 | 12.5% | 100% | 99.8% | 991ms |
| probe-8 (multi) | 6.2%+ | 98.0% | 97.4% | 548ms |
| probe-16 (multi) | 12.5%+ | 100% | 99.8% | 1.07s |

Scaling test (strict-8 config) shows sub-linear latency growth from 100K to 1M vectors. Download script at `scripts/download_sift1m.sh`, benchmark at `go/test/sift1m_benchmark_test.go` (build tag: `sift1m`). Runs weekly in CI with dataset caching.

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
