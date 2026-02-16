# Opaque

Privacy-preserving vector search using homomorphic encryption.

**Search a remote vector database without revealing your query. The server computes on encrypted data and never sees what you're searching for or what it returns.**

## How It Works

Opaque uses a three-level hierarchical approach to provide both **query privacy** and **data privacy**:

```
Level 1: HE centroid scoring  -> Server can't see query or which clusters were selected
Level 2: Decoy-based fetch    -> Server can't distinguish real from fake bucket requests
Level 3: Local AES decrypt    -> All final scoring happens client-side
```

### The Search Pipeline

```
Client                                          Server
  |                                               |
  |-- 1. Encrypt query with CKKS HE ------------->|
  |                                               |-- Compute HE(query) . centroid[i]
  |                                               |   for ALL centroids (blindly)
  |<-- Return encrypted scores -------------------|
  |                                               |
  |-- 2. Decrypt scores locally (server never sees)
  |-- 3. Select top clusters + generate decoys
  |                                               |
  |-- 4. Request [real + decoy] bucket IDs ------>|
  |      (shuffled, server can't tell apart)      |-- Return AES-encrypted blobs
  |<-- Return encrypted blobs --------------------|
  |                                               |
  |-- 5. AES decrypt vectors locally
  |-- 6. Score and rank locally
  |-- 7. Return top-K results
```

### What the Server Never Sees

| Information | Protected? | How |
|-------------|-----------|-----|
| Query vector | Yes | CKKS homomorphic encryption |
| Which clusters selected | Yes | Client-side HE decryption |
| Bucket access patterns | Obfuscated | Decoy buckets + shuffling |
| Vector contents | Yes | AES-256-GCM encryption |
| Similarity scores | Yes | Client-side computation |
| Final ranking | Yes | Client-side only |

## Cryptography

| Component | Implementation | Security |
|-----------|---------------|----------|
| Homomorphic Encryption | [Lattigo v5](https://github.com/tuneinsight/lattigo) CKKS | 128-bit (RLWE) |
| Symmetric Encryption | AES-256-GCM | 256-bit |
| Key Derivation | Argon2id | Memory-hard |
| Clustering | K-means with k-means++ init | N/A |

CKKS (Cheon-Kim-Kim-Song) is used for approximate arithmetic on encrypted floating-point vectors. This allows the server to compute dot products on encrypted queries without ever seeing the plaintext.

## Performance

Benchmarked on Apple M4 Pro with 100K 128-dimensional vectors.

### Core Operations

| Operation | Latency |
|-----------|---------|
| CKKS Encryption (128D) | ~5 ms |
| CKKS Decryption (128D) | ~0.8 ms |
| HE Dot Product (128D) | ~33 ms |
| AES-256-GCM Encrypt | <1 ms |

### Search Latency (100K Vectors, 64 Clusters)

| Phase | Standard | Batch (CKKS Slot Packing) |
|-------|----------|---------------------------|
| HE Encrypt Query | ~17 ms | ~17 ms |
| HE Centroid Scoring | ~1.7s (64 ops) | ~28 ms (1 op) |
| HE Decrypt Scores | ~45 ms | ~2 ms |
| Bucket Fetch | ~75 ms | ~50 ms |
| AES Decrypt + Score | ~160 ms | ~75 ms |
| **Total** | **~2s** | **~170 ms** |

*Batch mode uses CKKS slot packing to score all 64 centroids in a single HE operation instead of 64 separate ones.*

> **Note:** Run `go test -v -run TestBenchmark100KOptimized ./go/pkg/client/ -timeout 10m` to reproduce these benchmarks with actual recall measurement against brute-force ground truth. See [go/BENCHMARKS.md](go/BENCHMARKS.md) for full details.

## Quick Start

```bash
cd go

# Run core tests (crypto, LSH, clustering, blob storage)
go test ./pkg/encrypt/... ./pkg/crypto/... ./pkg/lsh/... ./pkg/cluster/... ./pkg/blob/... ./pkg/cache/...

# Run accuracy test on SIFT10K dataset (real embeddings, real ground truth)
go test -v -run TestSIFTKMeansEndToEnd ./pkg/client/ -timeout 5m

# Run 100K benchmark with recall measurement
go test -v -run TestBenchmark100KOptimized ./pkg/client/ -timeout 10m

# Run the dev server
go run ./cmd/devserver/main.go
```

## Project Structure

```
opaque/
├── go/
│   ├── pkg/
│   │   ├── crypto/           # CKKS homomorphic encryption (Lattigo v5)
│   │   ├── encrypt/          # AES-256-GCM symmetric encryption
│   │   ├── lsh/              # Locality-sensitive hashing
│   │   ├── cluster/          # K-means clustering (k-means++ init)
│   │   ├── blob/             # Encrypted blob storage (memory + file backends)
│   │   ├── cache/            # Centroid cache + batch CKKS packing
│   │   ├── hierarchical/     # Index builder (K-means + AES encryption)
│   │   ├── client/           # Search client (standard + batch modes)
│   │   ├── auth/             # Authentication service (token-based)
│   │   ├── enterprise/       # Per-enterprise config + key isolation
│   │   ├── server/           # REST API server
│   │   └── embeddings/       # SIFT dataset loader, embedding client
│   ├── internal/
│   │   ├── service/          # Search service (LSH + HE scoring)
│   │   ├── session/          # Session management with TTL
│   │   └── store/            # Vector storage interface
│   ├── cmd/
│   │   ├── devserver/        # Development server
│   │   ├── search-service/   # Production server entry point
│   │   └── cli/              # CLI benchmarking tool
│   ├── test/                 # Integration benchmarks
│   └── BENCHMARKS.md         # Detailed benchmark data
├── data/
│   └── siftsmall/            # SIFT10K dataset (10K real embeddings)
├── docs/
│   └── ARCHITECTURE.md       # System architecture details
├── ROADMAP.md                # Future plans
└── README.md
```

## Key Optimizations

### CKKS Slot Packing (Batch Mode)
CKKS supports packing multiple values into a single ciphertext (up to 16,384 slots with LogN=14). Opaque packs all 64 centroids into a single plaintext and computes all dot products in one HE operation, reducing centroid scoring from ~1.7s to ~28ms.

### Redundant Cluster Assignment
Each vector is assigned to its top-K nearest clusters (default K=2). This costs 2x storage but significantly improves recall for boundary queries that fall between clusters.

### Multi-Probe Cluster Selection
After HE scoring, clusters within a configurable threshold of the top-K score are also included. This compensates for CKKS approximation noise that can cause near-miss exclusions.

### Decoy Buckets
When fetching encrypted blobs, the client adds random "decoy" bucket requests and shuffles the order. The server cannot distinguish real requests from decoys, hiding the client's true access pattern.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design, privacy model, and security analysis.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including library API, gRPC service completion, PIR integration, and more.

## References

- [Lattigo](https://github.com/tuneinsight/lattigo) - Go library for lattice-based homomorphic encryption
- [CKKS Scheme](https://eprint.iacr.org/2016/421) - Cheon-Kim-Kim-Song approximate HE
- [SIFT Dataset](http://corpus-texmex.irisa.fr/) - Standard benchmark for nearest neighbor search

## License

MIT
