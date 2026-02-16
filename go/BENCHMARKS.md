# Benchmarks

## Test Environment

- **Machine**: Apple M4 Pro
- **Go Version**: 1.25
- **HE Library**: Lattigo v5 (CKKS scheme, 128-bit security)
- **Dataset**: SIFT10K (10K real 128-dim embeddings) + synthetic 100K vectors

## Core Operations

Measured with `go test -bench=. -benchmem`:

```
goos: darwin
goarch: arm64
cpu: Apple M4 Pro

BenchmarkEncryption-12         240    5,205,188 ns/op   2,381,208 B/op    141 allocs/op
BenchmarkDecryption-12       1,324      931,470 ns/op     790,464 B/op     58 allocs/op
BenchmarkSerialization-12    5,475      226,692 ns/op   4,196,050 B/op     38 allocs/op
BenchmarkHash-12           201,565        6,013 ns/op           8 B/op      1 allocs/op
```

| Operation | Dimension | Avg Latency |
|-----------|-----------|-------------|
| CKKS Encryption | 128 | 5.2 ms |
| CKKS Decryption | 128 | 0.8 ms |
| HE Dot Product | 128 | 33 ms |
| LSH Hash | 128 | 0.006 ms |
| LSH Search (k=50) | 128 | 0.004 ms |

## Search Latency (100K Synthetic Vectors, 64 Clusters)

Measured with `go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m`.

### Standard Search (64 sequential HE operations)

| Metric | Value |
|--------|-------|
| Average Latency | **2.56s** |
| Average Recall@10 | **96.0%** |
| HE Operations | 64 |

### Batch Search (CKKS Slot Packing)

| Metric | Value |
|--------|-------|
| Average Latency | **190ms** |
| Average Recall@10 | **96.0%** |
| HE Operations | 1 |
| **Speedup vs Standard** | **13.5x** |

Batch mode packs all 64 centroids into a single CKKS plaintext (16,384 slots / 128 dimensions = 128 centroids per pack). This replaces 64 sequential HE multiply+sum operations with a single one.

Recall is identical between standard and batch because the cluster selection logic is the same — batch only speeds up the HE centroid scoring step.

## Accuracy

### How We Measure

All recall numbers are computed against brute-force ground truth:
1. For each query, compute cosine similarity against ALL vectors in the dataset
2. Sort by similarity, take the true top-K
3. Compare against the system's top-K results
4. Recall@K = (number of matches) / K

### SIFT10K (Real Dataset)

Measured with `go test -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m`.

The SIFT10K dataset contains 10,000 128-dimensional real embeddings with provided ground truth. Note: SIFT provides Euclidean-distance ground truth, so the tests recompute cosine similarity ground truth for fair comparison.

| Metric | Value |
|--------|-------|
| Recall@1 (cosine GT) | **95.0%** |
| Recall@10 (cosine GT) | **96.0%** |
| Vectors scanned | 1,487 (14.9% of dataset) |
| Clusters | 64, top 8 selected |

### 100K Synthetic Vectors

Measured with `go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m`.

This benchmark generates 100K random normalized 128-dim vectors, builds a full index with all optimizations enabled, and measures recall against brute-force cosine similarity ground truth for 10 queries.

| Metric | Standard | Batch |
|--------|----------|-------|
| Recall@10 | **96.0%** | **96.0%** |
| Average Latency | 2.56s | **190ms** |
| Speedup | 1x | **13.5x** |

**Configuration:**
- 64 clusters, TopSelect=32
- Redundant Assignment = 2 (each vector in top-2 clusters)
- Multi-Probe: threshold=0.95, max=48 clusters
- Batch HE: CKKS slot packing

## Privacy Overhead

The privacy guarantees add overhead compared to plaintext search:

| Component | Purpose | Cost |
|-----------|---------|------|
| CKKS Encryption | Hide query from server | ~5 ms per query |
| HE Centroid Scoring | Server can't see scores | ~28 ms (batch) vs 0 ms (plaintext) |
| AES Decryption | Vectors encrypted at rest | ~45 ms for ~50K blobs |
| Decoy Buckets | Hide access patterns | ~2x network bandwidth |
| Redundant Assignment | Improve boundary recall | 2x storage |

## Reproducing

```bash
cd go

# Core crypto benchmarks
go test -bench=. -benchmem ./pkg/crypto/... ./pkg/lsh/...

# 100K latency benchmark (no recall, ~3 min)
go test -v -run TestBenchmark100K ./pkg/client/ -timeout 10m

# 100K optimized benchmark with recall (standard + batch, ~8 min)
go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m

# SIFT10K accuracy with real embeddings (~3 min)
go test -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

# SIFT10K comprehensive accuracy (multiple configs, ~5 min)
go test -v -run TestSIFTAccuracyWithGroundTruth ./pkg/client/ -timeout 10m
```
