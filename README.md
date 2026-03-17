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
- **Metadata & filtered search** — attach key-value metadata to vectors, filter at search time
- **CRUD operations** — Add, Update, Delete vectors with soft-delete and compaction via Rebuild
- **Persistence** — Save/Load database state to disk
- **File-backed storage** — memory or file-backed blob store for large datasets
- **Progress callbacks** — `OnBuildProgress` hook for observability during index builds
- **Batch operations** — AddBatch, AddBatchWithMetadata for bulk ingestion

## Configuration

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
| `ProbeStrategy` | "threshold" | `"threshold"` or `"gap"` (adaptive score-gap detection) |
| `GapMultiplier` | 2.0 | Gap sensitivity for `"gap"` strategy |
| `RedundantAssignments` | 1 | Clusters per vector (2 = better boundary recall, 2x storage) |
| `NumKMeansInit` | 1 | K-means initializations (higher = better centroids) |
| `NormalizedStorage` | true | Pre-normalize vectors for faster search |

## Examples

| Example | Description |
|---------|-------------|
| [`basic`](examples/basic/) | Minimal workflow: create, add, build, search |
| [`persistence`](examples/persistence/) | Save and load database state |
| [`metadata`](examples/metadata/) | Attach metadata and use filtered search |
| [`large-scale`](examples/large-scale/) | 10K+ vectors with batch operations |
| [`file-storage`](examples/file-storage/) | File-backed blob store for large datasets |
| [`http-server`](examples/http-server/) | HTTP API wrapping Opaque for self-hosted deployment |

Run any example:

```bash
go run ./examples/basic/
```

## Performance

Benchmarked on Apple M4 Pro, 100K 128-dimensional vectors, 64 clusters.

| Metric | Standard (64 HE ops) | Batch (1 HE op) |
|--------|---------------------|------------------|
| **Recall@10** | 96.0% | 96.0% |
| **Latency** | 2.56s | 190ms |

SIFT10K (real dataset): 95% Recall@1, 96% Recall@10 scanning 14.9% of data.

See [BENCHMARKS.md](BENCHMARKS.md) for full details.

## Architecture

Opaque uses a three-level privacy pipeline:

1. **HE centroid scoring** — server scores encrypted query against all centroids, can't see query or results
2. **Decoy-based fetch** — client requests real + fake buckets, server can't tell them apart
3. **Local AES decrypt + rank** — all final scoring happens client-side

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design, threat model, and crypto details.

## Self-Hosting

### Docker (SDK search service)

```bash
docker build -t opaque .
docker run -p 8080:8080 \
  -e OPAQUE_DB_PATH=/var/lib/opaque/db \
  -e OPAQUE_DIMENSION=128 \
  -e OPAQUE_AUTO_INDEX_ENABLED=true \
  -v opaque-data:/var/lib/opaque \
  opaque
```

### HTTP API (search service)

```bash
go run ./cmd/search-service/

# Add vectors (queued for auto-indexing)
curl -X POST localhost:8080/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{"vectors":[{"id":"v1","values":[0.1,0.2,...]}]}'

# Search (returns 503 until index is ready)
curl -X POST localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,...],"top_k":5}'

# Service stats
curl localhost:8080/v1/stats
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
