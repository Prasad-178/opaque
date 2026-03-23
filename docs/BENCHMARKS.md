# Benchmarks

## Test Environment

- **Machine**: Apple M4 Pro (10 CPUs)
- **Go Version**: 1.25
- **HE Library**: Lattigo v5 (CKKS scheme, 128-bit security)
- **Datasets**: SIFT10K (10K, 128-dim), SIFT1M (1M, 128-dim), GIST1M (1M, 960-dim), GloVe 6B (400K, 300-dim), synthetic 100K

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

### SIFT1M (1 Million Real Vectors)

Measured with `go test -tags sift1m -v -run TestSIFT1MAccuracy ./test/ -timeout 45m`.

The SIFT1M dataset contains 1,000,000 128-dimensional real embeddings. All recall numbers are against brute-force cosine similarity ground truth computed over the **entire** 1M dataset. The system uses the full privacy pipeline: CKKS homomorphic encryption for centroid scoring, AES-256-GCM blob encryption, and decoy cluster fetches for access pattern hiding.

**Configuration:** 128 clusters (~7,800 vectors each), 8 decoys per query.

| Config | Clusters Probed | Data Scanned | Recall@1 | Recall@10 | Avg Query |
|--------|----------------|--------------|----------|-----------|-----------|
| strict-4 | 4 (3.1%) | ~31K vectors | 82.0% | 79.4% | 274ms |
| strict-8 | 8 (6.2%) | ~62K vectors | 98.0% | 96.8% | 508ms |
| strict-16 | 16 (12.5%) | ~125K vectors | 100% | 99.8% | 991ms |
| probe-8 (multi-probe) | 8+ (6.2%+) | ~62K+ vectors | 98.0% | 97.4% | 548ms |
| probe-16 (multi-probe) | 16+ (12.5%+) | ~125K+ vectors | 100% | 99.8% | 1.07s |

**Recommended production config: strict-8** — 96.8% Recall@10 while scanning only 6.2% of the dataset, with full privacy guarantees (HE + AES + 8 decoys).

#### Scaling (strict-8 config, multi-probe threshold=0.95)

| Vectors | Build Time | Avg Query | Recall@10 |
|---------|-----------|-----------|-----------|
| 100K | ~3s | ~100ms | ~98% |
| 500K | ~12s | ~350ms | ~97% |
| 1M | ~25s | ~500ms | ~97% |

Latency scales sub-linearly with dataset size due to clustering — doubling vectors doesn't double query time.

### GIST 100K (960-dim Real Embeddings)

Measured with `go test -tags gist -v -run TestGIST100K_PCA_Benchmark ./test/ -timeout 60m`.

The GIST dataset contains 960-dimensional real image descriptors — 7.5x larger than SIFT's 128-dim. With 960-dim vectors, CKKS slot packing fits only 8 centroids per ciphertext (vs 64 for SIFT 128-dim), requiring 4x more HE operations per query. All recall numbers are against brute-force cosine similarity on the original 960-dim vectors.

**Configuration:** 100K vectors, 32 clusters (~3,125 vectors each), 8 decoys, multi-probe threshold=0.95.

| Config | Recall@1 | Recall@10 | Avg Query | P50 Query |
|--------|----------|-----------|-----------|-----------|
| probe-8 (25%) | 100% | **98.8%** | **2.6s** | 2.3s |
| probe-16 (50%) | 100% | **99.2%** | 6.2s | 6.0s |

#### PCA Dimensionality Reduction on GIST

PCA is applied client-side before CKKS encryption — zero privacy impact. Recall is measured against the original 960-dim ground truth.

**PCA Variance Explained (fitted on 100K GIST vectors):**

| Target Dim | Variance Retained | Centroids/Pack | HE Ops (32 clusters) |
|-----------|-------------------|----------------|---------------------|
| 960 (none) | 100% | 8 | 4 |
| 512 | 98.4% | 16 | 2 |
| 256 | 93.7% | 32 | 1 |
| 128 | 86.6% | 64 | 1 |

**PCA Search Results:**

| Config | Recall@1 | Recall@10 | Avg Query | Speedup |
|--------|----------|-----------|-----------|---------|
| **960d probe-8 (baseline)** | **100%** | **98.8%** | **2.6s** | **1x** |
| PCA→256 probe-8 | 38% | 19.0% | 695ms | 3.7x |
| PCA→256 probe-16 | 36% | 19.2% | 213ms | — |
| PCA→128 probe-8 | 24% | 14.0% | 296ms | 8.8x |
| PCA→128 probe-16 | 26% | 14.2% | 270ms | — |

**Finding:** GIST image descriptors have variance spread uniformly across all 960 dimensions. Even at 93.7% variance retained (256-dim), recall@10 drops to ~19%. PCA is effective for embeddings with concentrated variance (e.g., language model embeddings where top components dominate), but not for holistic image descriptors like GIST.

> **TODO:** Run PCA→512 benchmark (98.4% variance, 2 HE ops). Expected latency ~1-1.5s — could be the sweet spot balancing recall and speed for GIST. Run with:
> ```bash
> go test -tags gist -v -run TestGIST100K_PCA_Benchmark ./test/ -timeout 60m
> ```

### GloVe 100K (300-dim Word Embeddings)

Measured with `go test -tags glove -v -run TestGloVe_PCA_Benchmark ./test/ -timeout 30m`.

GloVe 6B 300-dim word vectors — real NLP embeddings from Stanford. At 300-dim, CKKS packing fits ~27 centroids per ciphertext, requiring only 2 HE ops for 32 clusters. All recall numbers are against brute-force cosine similarity on the original 300-dim vectors.

**Configuration:** 100K vectors (from GloVe 6B 400K), 32 clusters (~3,125 vectors each), 8 decoys, multi-probe threshold=0.95.

| Config | Recall@1 | Recall@10 | Avg Query | P50 Query |
|--------|----------|-----------|-----------|-----------|
| probe-8 (25%) | 82.0% | 81.0% | **162ms** | 165ms |
| probe-16 (50%) | 98.0% | 91.2% | 207ms | 188ms |
| probe-32 (100%) | **100%** | **100%** | 311ms | 255ms |

**Recommended config: probe-16** — 91.2% Recall@10 at 207ms with full privacy guarantees. For applications requiring perfect recall, probe-32 achieves 100% Recall@10 at 311ms — latency scales sub-linearly (4x clusters, only ~2x time).

#### PCA Dimensionality Reduction on GloVe

**PCA Variance Explained (fitted on 100K GloVe vectors):**

| Target Dim | Variance Retained | Centroids/Pack | HE Ops (32 clusters) |
|-----------|-------------------|----------------|---------------------|
| 300 (none) | 100% | 27 | 2 |
| 256 | 99.5% | 32 | 1 |
| 192 | 89.1% | 42 | 1 |
| 128 | 67.0% | 64 | 1 |
| 96 | 54.3% | 85 | 1 |
| 64 | 40.6% | 128 | 1 |

**PCA Search Results:**

| Config | Recall@1 | Recall@10 | Avg Query | Speedup |
|--------|----------|-----------|-----------|---------|
| **300d probe-8 (baseline)** | **82.0%** | **81.0%** | **162ms** | **1x** |
| PCA→128 probe-8 | 68.0% | 33.8% | 80ms | 2.0x |
| PCA→128 probe-16 | 72.0% | 36.2% | 100ms | — |
| PCA→96 probe-8 | 52.0% | 22.6% | 83ms | 2.0x |
| PCA→96 probe-16 | 68.0% | 26.8% | 90ms | — |
| PCA→64 probe-8 | 38.0% | 15.2% | 71ms | 2.3x |
| PCA→64 probe-16 | 42.0% | 15.8% | 78ms | — |

**Finding:** Despite "concentrated" variance (67% at 128-dim vs GIST's 87% at 128-dim), PCA still severely degrades Recall@10 for GloVe. The cosine similarity neighborhoods change significantly under PCA even when bulk variance is preserved. The latency gain (2x) does not justify the recall loss (81% → 34%).

### PCA: Lessons Learned

PCA dimensionality reduction is available in the pipeline (`Config.PCADimension`) and is applied client-side before CKKS encryption — zero privacy impact. However, benchmarking on two fundamentally different embedding types revealed that **PCA is not generally effective** for preserving cosine-similarity search quality:

| Dataset | Type | Best PCA Config | Recall@10 (baseline → PCA) | Verdict |
|---------|------|-----------------|---------------------------|---------|
| GIST 100K | Image descriptors (960d) | PCA→256 (93.7% variance) | 98.8% → 19.0% | Not viable |
| GloVe 100K | Word embeddings (300d) | PCA→128 (67.0% variance) | 81.0% → 33.8% | Not viable |

**Why PCA fails for cosine search:** PCA preserves bulk variance (L2 distance structure) but cosine similarity depends on *angular* relationships between vectors. Projecting to a lower-dimensional subspace distorts these angles, changing which vectors are nearest neighbors — even when most variance is retained.

**When PCA might help:**
- Embeddings with extreme variance concentration (>99% in top-K components)
- Use cases where approximate recall (~50%) is acceptable for large latency gains
- As a pre-filter stage combined with re-ranking on original dimensions

**Recommended approach for latency reduction:** CKKS SIMD slot packing and parallel processing are the proven approaches. These achieve 3-13x speedups without any recall loss. PCA remains available for experimentation with custom embeddings.

## Pipeline Optimizations

The following optimizations have been applied to the k-means + HE search pipeline:

| Optimization | Phase | Impact |
|-------------|-------|--------|
| Parallel vector encryption | Build | ~4-5x faster build (multi-core AES-GCM) |
| K-means multi-initialization | Build | +0.5-1% Recall (better centroids via lowest-inertia selection) |
| Empty cluster recovery | Build | More balanced cluster sizes, fewer wasted partitions |
| SIMD batch HE (slot packing) | Search | 13.5x speedup (64 ops → 1 op + 7 rotations) |
| Parallel batch pack processing | Search | Up to ~3x for high-dim (960d: 4 packs run concurrently) |
| Lazy decoy skip | Search | ~50% less AES decrypt + scoring (skip decoy cluster blobs) |
| Adaptive score-gap probing | Search | 5-10% fewer vectors fetched at same recall |
| Pre-normalized storage | Search | 10-15% faster local scoring (skip per-vector normalization) |
| Parallel AES decryption | Search | Multi-core blob decryption in search pipeline |
| Dimension-bounded rotations | Search | Fewer HE rotations for non-power-of-2 dims (960d: 10 vs 14) |
| Heap-based top-K selection | Search | O(n log K) vs O(n log n) for scoring results (min-heap, K=10) |
| Pre-sized score map | Search | Reduced map growth allocations during local scoring |
| PCA dimensionality reduction | Both | Reduces HE ops, AES size, local scoring — situational, see notes below |

All optimizations preserve privacy guarantees: same HE encryption, same AES-GCM, same decoy patterns. PCA is applied client-side before encryption — the server never sees original or reduced vectors.

### High-Dimensional Optimization Impact (GIST 100K, 960-dim)

The parallel batch pack + lazy decoy optimizations have the most impact on high-dimensional vectors where HE packing efficiency is poor:

| Config | Before Optimization | After Optimization | Speedup |
|--------|--------------------|--------------------|---------|
| GIST 100K probe-8 (25%) | ~8.6s | **~2.6s** | **3.3x** |
| GIST 100K probe-16 (50%) | ~12.4s | **~6.2s** | **2.0x** |

The improvement comes from:
- **Parallel batch pack processing**: 4 CKKS packs run concurrently instead of sequentially (up to 4x on HE scoring phase).
- **Lazy decoy skip**: client filters fetched blobs to only decrypt/score real clusters, skipping ~50% of AES + scoring work.
- **Dimension-bounded rotations**: 10 rotations for 960-dim instead of 14 (saves ~40ms per sequential HE op).

### Build Speed (strict-8 config, 1M vectors)

| Metric | Before | After |
|--------|--------|-------|
| Build time | ~25s | ~5-8s |

### ClusterStats API

After `Build()`, call `db.ClusterStats()` to inspect cluster quality:

```go
stats := db.ClusterStats()
fmt.Printf("Clusters: %d, Min: %d, Max: %d, Avg: %.1f, Empty: %d\n",
    stats.NumClusters, stats.MinSize, stats.MaxSize, stats.AvgSize, stats.EmptyClusters)
```

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

# SIFT1M accuracy benchmark (requires dataset download, ~5 min)
../scripts/download_sift1m.sh
go test -tags sift1m -v -run TestSIFT1MAccuracy ./test/ -timeout 45m

# SIFT1M scaling benchmark (~10 min)
go test -tags sift1m -v -run TestSIFT1MScaling ./test/ -timeout 45m

# GIST 100K PCA benchmark (960-dim, requires dataset download, ~20 min)
../scripts/download_gist1m.sh
go test -tags gist -v -run TestGIST100K_PCA_Benchmark ./test/ -timeout 60m

# GloVe 100K PCA benchmark (300-dim word embeddings, ~4 min)
# Download GloVe 6B from https://nlp.stanford.edu/projects/glove/ and extract glove.6B.300d.txt to data/glove/
go test -tags glove -v -run TestGloVe_PCA_Benchmark ./test/ -timeout 30m
```
