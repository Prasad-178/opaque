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

Measured with `go test -v -run TestBenchmark100K ./pkg/client/ -timeout 10m`.

### Standard Search (64 sequential HE operations)

| Phase | Avg Time | % of Total |
|-------|----------|------------|
| HE Encrypt Query | ~17 ms | ~0.8% |
| HE Centroid Scores (64 ops) | ~1.7s | ~85% |
| HE Decrypt Scores | ~45 ms | ~2% |
| Bucket Selection | <1 ms | <0.1% |
| Bucket Fetch | ~75 ms | ~4% |
| AES Decrypt | ~85 ms | ~4% |
| Local Scoring | ~75 ms | ~4% |
| **Total** | **~2s** | |

### Batch Search (CKKS Slot Packing)

| Phase | Avg Time | % of Total |
|-------|----------|------------|
| HE Encrypt Query | ~17 ms | ~10% |
| HE Batch Centroid Scores (1 op) | ~28 ms | ~16% |
| HE Decrypt Scores | ~2 ms | ~1% |
| Bucket Selection | <1 ms | <0.1% |
| Bucket Fetch | ~50 ms | ~29% |
| AES Decrypt | ~45 ms | ~26% |
| Local Scoring | ~30 ms | ~17% |
| **Total** | **~170 ms** | |

Batch mode packs all 64 centroids into a single CKKS plaintext (16,384 slots / 128 dimensions = 128 centroids per pack). This replaces 64 sequential HE multiply+sum operations with a single one.

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

### 100K Synthetic Vectors

Measured with `go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m`.

This benchmark generates 100K random normalized 128-dim vectors, builds a full index with all optimizations enabled, and measures recall against brute-force ground truth for 10 queries.

**Configuration:**
- 64 clusters, TopSelect=32
- Redundant Assignment = 2 (each vector in top-2 clusters)
- Multi-Probe: threshold=0.95, max=48 clusters
- Batch HE: CKKS slot packing

Both standard and batch search produce identical recall (same cluster selection logic), but batch is ~12x faster for centroid scoring.

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
