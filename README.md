# Opaque

Privacy-preserving vector search using homomorphic encryption.

**Search encrypted vectors without revealing your query. The server computes on encrypted data and never sees what you're searching for.**

## Install

```bash
go get github.com/Prasad-178/opaque
```

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Prasad-178/opaque"
)

func main() {
	db, err := opaque.NewDB(opaque.Config{Dimension: 128, NumClusters: 16})
	if err != nil { log.Fatal(err) }
	defer db.Close()

	ctx := context.Background()
	db.Add(ctx, "doc-1", vector1)
	db.Add(ctx, "doc-2", vector2)

	db.Build(ctx) // k-means clustering + HE engine init

	results, _ := db.Search(ctx, queryVector, 10)
	for _, r := range results {
		fmt.Printf("  %s: %.4f\n", r.ID, r.Score)
	}
}
```

## Features

- **Homomorphic encryption** — queries are encrypted with CKKS; the server scores centroids without seeing the query
- **AES-256-GCM** — vectors encrypted at rest, decrypted only client-side
- **Decoy requests** — real bucket fetches are mixed with fake ones to hide access patterns
- **Incremental indexing** — vectors added after `Build()` are instantly searchable, no rebuild needed
- **Incremental centroid updates** — centroids are updated via running mean as vectors are added, maintaining cluster quality
- **Metadata & filtered search** — attach key-value metadata to vectors, filter at search time
- **CRUD operations** — Add, Update, Delete vectors with soft-delete and compaction via Rebuild
- **Persistence** — Save/Load database state to disk with encrypted metadata
- **File-backed storage** — memory or file-backed blob store for large datasets
- **Product quantization** — PQ-accelerated local scoring (2x+ speedup, <1% recall loss)
- **PCA dimensionality reduction** — optional client-side PCA for reduced latency
- **Progress callbacks** — `OnBuildProgress` hook for observability during index builds
- **Batch operations** — AddBatch, AddBatchWithMetadata for bulk ingestion

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `Dimension` | (required) | Vector dimension |
| `NumClusters` | 64 | K-means clusters. More = faster search, less privacy |
| `TopClusters` | NumClusters/2 | Clusters probed per search. More = better recall |
| `NumDecoys` | 8 | Decoy clusters for access pattern hiding (used when `TargetEpsilon == 0`) |
| `TargetEpsilon` | 0 (off) | DP-style privacy budget; when > 0, derives NumDecoys from ⌈(NumClusters - TopClusters) · e^(-ε)⌉. Smaller ε = stronger privacy = more decoys = more bandwidth. See **Privacy Tuning** below. |
| `PaddingMode` | `PaddingNone` | `PaddingNone`, `PaddingBucketed` (next-pow2, ~6-12% storage), or `PaddingMaxFixed` (~10-25% storage). Closes volume side-channel by equalizing cluster sizes. |
| `WorkerPoolSize` | min(NumCPU, 8) | Parallel HE engines (~50MB each) |
| `Storage` | Memory | `opaque.Memory` or `opaque.File` |
| `StoragePath` | "" | Directory for file storage |
| `ProbeThreshold` | 0.95 | Multi-probe inclusion threshold |
| `ProbeStrategy` | "threshold" | `"threshold"` or `"gap"` (adaptive score-gap detection) |
| `GapMultiplier` | 2.0 | Gap sensitivity for `"gap"` strategy |
| `RedundantAssignments` | 1 | Clusters per vector (2 = better boundary recall, 2x storage) |
| `NumKMeansInit` | 1 | K-means initializations (higher = better centroids) |
| `NormalizedStorage` | true | Pre-normalize vectors for faster search |
| `PCADimension` | 0 (off) | Target dimension for PCA reduction |
| `PQSubspaces` | 0 (off) | PQ subspaces for fast scoring (8=recommended for 128-dim) |
| `AutoIndexEnabled` | false | Enable automatic background re-indexing |

## Privacy Tuning

Opaque's access-pattern privacy (which cluster a query targets) is tuned via the
`TargetEpsilon` config field. The smaller the ε, the stronger the upper bound
on per-query cluster-identity distinguishability — at the cost of fetching
more decoy clusters per search. The bound is

```
NumDecoys = ⌈(NumClusters - TopClusters) · e^(-ε)⌉
```

For SIFT1M (NumClusters=128, TopClusters=8) the resulting trade-offs:

| ε     | Decoys | Privacy meaning                                    | Cost (vs ε off, baseline NumDecoys=8) |
|-------|--------|----------------------------------------------------|----------------------------------------|
| 1.0   | 45     | Very strong: distinguish ratio ≤ e ≈ 2.72         | ~5× decoy fetches, +200%+ latency      |
| 2.0   | 17     | Strong: ratio ≤ e^2 ≈ 7.4                         | ~2× decoys, +60-80% latency            |
| 2.5   | 10     | **Recommended default** — modest upgrade over 8-decoy baseline | +5-15% latency               |
| 2.71  | ~9     | ≈ baseline (matches the historical 8-fixed scheme) | ~baseline                              |
| 3.0   | 6      | Weaker than baseline — only for cost-sensitive    | -10% latency                           |

`TargetEpsilon=0` (default) leaves `NumDecoys` to take effect directly (8 by
default). Set `TargetEpsilon=2.5` for a quantifiable, paper-grade privacy claim
with minimal cost.

The bound itself is informal (per-query distinguishability ratio); composition
over many queries scales as τ·ε. See `docs/SECURITY_MODEL.md` §5.1 for the full
formalism, caveats (uniform-prior assumption, single-query nature), and how π +
padding compose with this knob to provide defense in depth.

### Privacy presets (recommended config templates)

```go
// Cost-sensitive (acceptable when workload is broad / non-targeted)
opaque.Config{
    TargetEpsilon: 3.0,
    PaddingMode:   opaque.PaddingNone,
    PQSubspaces:   8,
}

// Balanced — recommended default for SIFT-class workloads
opaque.Config{
    TargetEpsilon: 2.5,
    PaddingMode:   opaque.PaddingBucketed,
    PQSubspaces:   8,
}

// High privacy (regulated / sensitive workloads)
opaque.Config{
    TargetEpsilon: 1.5,
    PaddingMode:   opaque.PaddingMaxFixed,
    PQSubspaces:   8,
}
```

## Optimal Configuration (SIFT1M)

The best balance of latency, recall, and privacy at SIFT1M (1 M × 128-dim) is:

```go
opaque.Config{
    Dimension:      128,
    NumClusters:    128,
    TopClusters:    8,
    ProbeThreshold: 0.95,             // multi-probe expansion
    PaddingMode:    opaque.PaddingBucketed,
    TargetEpsilon:  2.5,              // derives NumDecoys ≈ 10
    // PQSubspaces: 8,                // optional: enable for ~7% faster query
                                      //   at -2 pp Recall@10 trade-off
}
```

Verified on AWS m6i.2xlarge (8 vCPU, 32 GB, Intel Xeon Platinum 8375C),
2026-05-01 (commit `429ddf7`):

| Variant                          | Recall@1 | Recall@10 | Avg query |
|----------------------------------|----------|-----------|-----------|
| **probe-8, no PQ (recommended)** | **100 %**| **99.6 %**| **462 ms**|
| probe-8, PQ-M8                   | 100 %    | 97.6 %    | 428 ms    |
| probe-16, no PQ (max recall)     | 100 %    | 100.0 %   | 635 ms    |
| probe-16, PQ-M8                  | 100 %    | 99.4 %    | 578 ms    |

Per-config trade-offs and pre/full-mitigation comparisons are in
[`deploy/bench-cpu/results/SUMMARY.md`](deploy/bench-cpu/results/SUMMARY.md).

**When to enable PQ:** PQ wins more at higher dimensions (768-dim text
embeddings, 960-dim GIST features). At SIFT1M (128-dim) the benefit is
modest. PQ also adds ~5 minutes of one-time codebook training to `Build()`.

## Incremental Indexing

After `Build()`, new vectors added via `Add()` are **instantly searchable** without calling `Rebuild()`:

```go
db.Build(ctx) // initial k-means clustering

// These are immediately searchable — no rebuild needed
db.Add(ctx, "new-1", newVector1)
db.Add(ctx, "new-2", newVector2)

results, _ := db.Search(ctx, query, 10) // finds new-1, new-2
```

**How it works:**
1. New vectors are assigned to their nearest existing centroid — O(k·d) per vector
2. Centroids are updated incrementally via the running mean formula: `c_new = c_old + (x - c_old) / (n + 1)`
3. Vectors are encrypted and stored in the appropriate cluster bucket
4. HE centroid caches are refreshed so searches use updated centroids

Call `Rebuild()` when data distribution has drifted significantly or cluster sizes become skewed.

## Examples

| Example | Description |
|---------|-------------|
| [`basic`](examples/basic/) | Minimal workflow: create, add, build, search |
| [`persistence`](examples/persistence/) | Save and load database state |
| [`metadata`](examples/metadata/) | Attach metadata and use filtered search |
| [`large-scale`](examples/large-scale/) | 10K+ vectors with batch operations |
| [`file-storage`](examples/file-storage/) | File-backed blob store for large datasets |
| [`http-server`](examples/http-server/) | HTTP API wrapping Opaque for self-hosted deployment |
| [`benchmark`](examples/benchmark/) | Comprehensive benchmark: build, search, recall, privacy, incremental indexing |

Run any example:

```bash
go run ./examples/basic/
```

## Performance

Benchmarked on Apple M4 Pro, 128-dimensional vectors, 64 clusters.

| Metric | Value |
|--------|-------|
| **Build** (1K vectors) | ~3,400 vectors/sec |
| **Search latency** | ~68ms avg (batch HE) |
| **Incremental add** | ~300µs/vector |
| **Recall@10** (SIFT10K) | 96% at 14.9% data scanned |
| **Recall@10** (SIFT1M) | 96.8% at 6.2% data scanned |
| **PQ speedup** (SIFT 100K) | 1.26x at 99.4% recall (probe-16) |
| **GPU HE speedup** (T4) | 8.6x per rotation (measured); 9.4x GIST HE total (projected) |

Run the benchmark yourself:

```bash
go run ./examples/benchmark/
```

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full details including SIFT1M results.

## Architecture

Opaque uses a three-level privacy pipeline:

1. **HE centroid scoring** — server scores encrypted query against all centroids, can't see query or results
2. **Decoy-based fetch** — client requests real + fake buckets, server can't tell them apart
3. **Local AES decrypt + rank** — all final scoring happens client-side

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design, threat model, and crypto details.

## Self-Hosting

### Docker (gRPC search service)

```bash
docker build -t opaque .
docker run -p 50051:50051 -p 8080:8080 \
  -e OPAQUE_STORAGE_BACKEND=file \
  -e OPAQUE_STORAGE_PATH=/var/lib/opaque/vectors.json \
  -v opaque-data:/var/lib/opaque \
  opaque
```

### HTTP API (example)

```bash
go run ./examples/http-server/

# Add vectors
curl -X POST localhost:8080/vectors -d '{"vectors":[{"id":"v1","values":[0.1,0.2,...]}]}'

# Build index
curl -X POST localhost:8080/admin/build

# Search
curl -X POST localhost:8080/search -d '{"vector":[0.1,0.2,...],"top_k":5}'
```

## Development

```bash
make test-fast    # go test -short ./...
make test         # full test suite
make lint         # go vet ./...
make test-bench   # crypto/LSH micro-benchmarks
make test-sift    # SIFT10K accuracy
make test-100k    # 100K vector benchmark
```

## License

[MIT](LICENSE)
