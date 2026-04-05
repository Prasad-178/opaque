# Roadmap

## Current State

Opaque is a working privacy-preserving vector search system implemented in Go. The core pipeline is functional: CKKS homomorphic encryption, k-means clustering, AES-256-GCM blob encryption, and hierarchical search with decoy-based access pattern hiding.

### What Works Today

- **Library API** (`opaque.NewDB` → `Add` → `Build` → `Search`) for simple usage
- CKKS homomorphic encryption via Lattigo v5 (128-bit security)
- K-means clustering with k-means++ initialization, multi-init support, and empty cluster recovery
- AES-256-GCM encrypted blob storage (memory and file backends)
- Hierarchical search: HE centroid scoring -> decoy fetch -> local AES decrypt
- Parallel build pipeline (vector encryption) and parallel search pipeline (AES decryption)
- Adaptive score-gap probing and pre-normalized vector storage for faster search
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

### ~~Pipeline Optimizations~~ (Done)
Optimized the existing k-means + HE pipeline for better build speed, cluster quality, and search latency:

- **Parallel vector encryption** during Build (4-5x speedup via `runtime.NumCPU()` workers)
- **K-means multi-initialization** (`NumKMeansInit` config) — runs N parallel initializations, keeps lowest-inertia result for better centroids
- **Empty cluster recovery** — reassigns farthest vector from largest cluster to empty clusters during k-means iterations
- **ClusterStats API** — `db.ClusterStats()` exposes min/max/avg cluster sizes and iteration count after Build
- **Adaptive score-gap probing** (`ProbeStrategy: "gap"`) — detects natural breaks in HE score distributions instead of fixed threshold
- **Pre-normalized vector storage** (`NormalizedStorage`) — stores unit-length vectors, skips per-vector normalization during search (10-15% faster local scoring)
- **Parallel AES decryption** in search pipeline (multi-goroutine blob decryption)

### ~~Incremental Indexing (Tier 1: Train Once, Assign Many)~~ (Done)
After the initial `Build()`, new vectors added via `Add()` are immediately assigned to their nearest existing centroid, encrypted, and stored — making them **instantly searchable** without calling `Rebuild()`. Centroids are frozen after the initial k-means run. `Rebuild()` remains available for full re-clustering when cluster quality degrades.

This eliminates the O(n) rebuild cost for every mutation. Each `Add()` after `Build()` is now O(k·d) (one nearest-centroid lookup) instead of requiring a full re-index.

### ~~Incremental Indexing (Tier 2: Incremental Centroid Updates)~~ (Done)
Update centroids incrementally when vectors are added using the running mean formula:
```
c_new = c_old + (x - c_old) / (n + 1)    # add
c_new = (n · c_old - x) / (n - 1)         # remove
```
Add split/merge logic when cluster sizes become skewed. These formulas are exact (not approximations) and map directly to CKKS depth-1 operations, making them feasible in the encrypted domain.

### Incremental Indexing (Tier 3: HE-Native Centroid Updates)
Perform centroid updates entirely in the homomorphic encryption domain — the server updates encrypted centroids without ever decrypting them. This eliminates the plaintext-during-rebuild window and provides end-to-end encrypted index maintenance. The incremental mean formula requires only one HE subtraction, one scalar multiplication, and one HE addition (depth-1 circuit).

### GPU Acceleration — Profiled, Benchmarked, Integration Layer Built
Profiling reveals Galois rotation (key-switching) is 71-84% of HE time, not NTT. Real benchmarks on AWS g4dn.xlarge (Tesla T4) confirm **8.6x speedup on rotation** (0.76ms GPU vs 6.55ms CPU, verified across 2 benchmark runs). Cross-degree scaling measured from 4096 to 32768.

**Implementation status:**
- `GPUHEProvider` implementing `HEProvider` interface — done (`pkg/crypto/gpu_provider.go`)
- GPU HE gRPC proto + generated code — done (`api/proto/gpuhe.proto`)
- GPU server with CPU stub backend — done (`cmd/gpu-server/main.go`)
- Config wiring (`GPUServerAddress`) — done
- Tests (5 tests covering encrypt/decrypt, batch dot product, provider matching, pool, health) — done
- Terraform infra for ephemeral GPU instances — done (`deploy/gpu/`)
- **Remaining:** cgo bridge to HEonGPU C++ for real GPU execution

See `docs/GPU_ACCELERATION.md` for profiling data, benchmark results, and architecture.

### ~~Product Quantization (PQ)~~ (Done)
Optional PQ via `Config.PQSubspaces`. Compresses vectors into compact M-byte codes and uses ADC lookup tables for fast approximate scoring. Two-phase search: PQ ADC for bulk scoring, exact re-ranking of top candidates with full vectors. Applied client-side before AES encryption — zero privacy impact.

**Results (all measured, not projected):**
- GIST 100K (960-dim): PQ-M32 probe-8 achieves 98.0% Recall@10 at **497ms vs 9.53s standard (19.2x speedup)**
- SIFT 100K (128-dim): PQ-M8 probe-16 achieves 99.4% Recall@10 at 127ms vs 160ms standard (1.26x)
- Synthetic 100K: 2.1x speedup at 99.5% recall
- ADC scoring: 3.7ns vs 49.7ns per vector (13.5x faster)
- Privacy guarantees identical — PQ is entirely client-side. See `docs/BENCHMARKS.md` for full results.

### ~~PCA Dimensionality Reduction~~ (Done)
Optional PCA via SVD in `go/pkg/pca/`. Enabled with `Config.PCADimension`. Applied client-side before encryption — no privacy impact. 128D→64D achieves 77% variance retention with perfect self-match recall.

---

## Long Term

### Threshold CKKS (Key Ownership) — Phases 1–2 Complete, Parallelized
Split the CKKS secret key across an independent key committee using Lattigo's `mhe` package. The DB server holds only evaluation keys and cannot decrypt. Decryption is a single-round protocol where t-of-N committee nodes produce partial key-switches, re-encrypting results directly under the querying client's ephemeral public key.

ThresholdDecrypt is now parallelized internally — Shamir-to-additive share conversion and PCKS partial share generation run in goroutines per participant. Each `thresholdEvalEngine` has its own decryptor/encoder — Lattigo's `rlwe.Decryptor` is NOT thread-safe (sharing it caused a data race manifesting as underflow panics in concurrent decryption). Noise flooding sigma=2^20.

**Latest benchmarks** (Apple M4, SearchBatch with SIMD packing, 3-of-5 committee, real SIFT image descriptors, brute-force cosine ground truth, 2ms simulated datacenter RTT):
- **Multi-dataset validation across SIFT10K, SIFT100K, and SIFT1M**
- SIFT10K (10K vectors, 100 queries, 16 clusters): 1.13-2.06x overhead, up to 98.0% recall@10
- SIFT100K (100K vectors, 50 queries, 64 clusters): 0.87-1.49x overhead, probe-32 at 368ms/322ms with 97.8% recall@10
- SIFT1M (1M vectors, 50 queries, 128 clusters): 0.69-1.10x overhead, probe-48 at 653ms/719ms with 95.0% recall@10
- **SIFT1M probe-64 (50%) = 99.2%/99.0% recall@10 at ~860ms, probe-96 (75%) = 100% at ~1.1s**
- GIST100K (100K vectors, 960-dim, 32 clusters): ~0% overhead, 100% recall@10 at 50% probe
- **Threshold overhead ~0-10% at scale; negligible relative to HE compute time**
- Recall equivalent between direct and threshold modes across all datasets
- Micro-benchmarks: decrypt scales from 22ms (2-of-3) to 27ms (5-of-7)

Implementation phases:
1. ~~**Engine refactor**~~ ✅ — `HEProvider` interface with `DirectHEProvider` and `ThresholdHEProvider`
2. ~~**Threshold POC**~~ ✅ — local implementation with parallelized `Committee`, `Participant`, `ClientSession` using Lattigo `mhe`
3. **Committee gRPC service** — lightweight service for committee nodes (PartialKeySwitch RPC)
4. **Distributed integration** — replace local Committee with network-distributed committee
5. **Hardening** — redundancy-based bad node detection, metrics, stress testing

See `docs/THRESHOLD_CKKS.md` for the full architecture and benchmarks.

### Private Information Retrieval (PIR)
Replace decoy-based bucket fetching with cryptographic PIR for stronger access pattern privacy. The current decoy approach provides k-anonymity; PIR would provide cryptographic guarantees. This is a significant performance tradeoff that needs careful benchmarking.

### Key Rotation
Implement periodic key rotation for AES encryption keys, HE parameters, and (with threshold CKKS) committee key shares without requiring full re-encryption of the database.

### Distributed Storage
Support distributed blob storage backends (e.g., Redis, S3) for horizontal scaling beyond single-node deployments.

### TEE Integration
Investigate Trusted Execution Environment (Intel SGX, AWS Nitro Enclaves) as an alternative or complement to pure-crypto approaches for specific deployment scenarios.
