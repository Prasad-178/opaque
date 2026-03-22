# Architecture

## Overview

Opaque is a privacy-preserving vector search system. It allows a client to search a remote vector database without the server learning the query, the results, or which vectors were accessed.

The system uses three cryptographic layers:

1. **CKKS Homomorphic Encryption** (Lattigo v5) - Server computes dot products on encrypted queries
2. **AES-256-GCM** - Vectors are encrypted at rest; only the client can decrypt
3. **Decoy Requests** - Real bucket accesses are mixed with fake ones to hide access patterns

## Search Pipeline

### Index Building (Offline)

```
Input: N vectors (128-dim, normalized)
  |
  v
K-Means Clustering (k-means++ init, 64 clusters)
  |
  v
For each vector:
  1. Assign to top-K nearest clusters (redundant assignment, default K=2)
  2. Encrypt with AES-256-GCM using per-enterprise key
  3. Store as blob in cluster bucket
  |
  v
Output: 64 cluster centroids + encrypted blob store
```

### Query (Online)

```
Level 1: HE Centroid Scoring (server-side, encrypted)
  - Client encrypts query with CKKS
  - Server computes HE(query) . centroid[i] for all centroids
  - Client decrypts scores locally, selects top clusters
  - Server NEVER sees query, scores, or selection

Level 2: Decoy-Based Fetch
  - Client picks top clusters + random decoy clusters
  - Shuffles all bucket IDs, sends to server
  - Server returns encrypted blobs for all requested buckets
  - Server CANNOT distinguish real from decoy

Level 3: Local Scoring
  - Client AES-decrypts all fetched vectors
  - Computes cosine similarity locally
  - Returns top-K results
  - Server sees NOTHING from this step
```

## Homomorphic Encryption Details

### CKKS Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| LogN | 14 | Ring degree 2^14 = 16,384 slots |
| LogQ | [60,45,45,45,45,45,45,45] | Ciphertext modulus chain |
| LogP | [61, 61] | Special primes for key-switching |
| LogDefaultScale | 45 | Encoding precision (2^45) |
| Security | 128-bit | Based on RLWE hardness |

### HE Dot Product

The core operation is computing `dot(encrypted_query, plaintext_centroid)`:

1. **Encode** centroid as CKKS plaintext (padded to 16,384 slots)
2. **Multiply** encrypted query by plaintext centroid (component-wise)
3. **Rescale** to manage scale growth
4. **Rotate + Add** in a tree pattern to sum all slots into slot[0]

Result: an encrypted scalar (the dot product) that only the client can decrypt.

### Batch Mode (CKKS Slot Packing)

With 16,384 slots and 128-dimensional vectors, we can pack `16384 / 128 = 128` centroids per plaintext. For 64 centroids, this means **a single HE operation** replaces 64 separate ones.

The query is replicated across all centroid positions in the packed plaintext. After multiplication and partial summation (rotating only within each 128-slot segment), each segment's first slot contains one dot product result.

This reduces centroid scoring from ~1.7s to ~28ms.

## AES-256-GCM Encryption

Each vector is encrypted with AES-256-GCM using:
- A per-enterprise 256-bit key (generated via `crypto/rand`)
- A random 12-byte nonce per encryption
- The vector ID as additional authenticated data (AAD)

Key derivation uses Argon2id with 64MB memory cost and 4 threads.

## Clustering

### K-Means with K-Means++ Initialization

- **Initialization**: K-means++ selects initial centroids proportional to squared distance from existing centroids
- **Assignment**: Euclidean distance minimization
- **Convergence**: Inertia delta with configurable tolerance
- **Parallelism**: 4 workers for datasets > 1000 vectors

### Redundant Assignment

Each vector is assigned to its top-K nearest centroids (default K=2). This creates duplicate encrypted blobs but ensures boundary vectors are findable from multiple clusters. Storage cost is Kx, but recall improves significantly.

## Privacy Model

### Threat Model

- **Honest-but-curious server**: Follows protocol correctly but analyzes all data it sees
- **Trusted client**: Client environment must be secure
- **Per-enterprise isolation**: Each enterprise has independent AES keys, LSH seeds, and centroids

### What the Server Sees

| During HE Scoring | Server Sees |
|---|---|
| Encrypted query ciphertext | Yes (but can't decrypt - no secret key) |
| Centroid plaintexts | Yes (these are public to the server) |
| Encrypted dot product results | Yes (but can't decrypt) |
| Which clusters client selected | **No** (client decrypts scores locally) |

| During Blob Fetch | Server Sees |
|---|---|
| Requested bucket IDs | Yes (but mixed with decoys) |
| Which buckets are real vs decoy | **No** (indistinguishable) |
| Encrypted blob contents | Yes (but can't decrypt - no AES key) |

| During Local Scoring | Server Sees |
|---|---|
| Decrypted vectors | **No** (client-side only) |
| Similarity scores | **No** (client-side only) |
| Final ranking | **No** (client-side only) |

### Limitations

- **CKKS approximation**: Results are approximate due to the nature of CKKS arithmetic. Error is typically <1e-4 for normalized vectors.
- **Decoys are not PIR**: Decoy-based hiding provides k-anonymity but not cryptographic access pattern privacy. A future PIR integration is planned for stronger guarantees.
- **Cluster count leaks structure**: The number of clusters and their sizes are visible to the server. This reveals the coarse distribution of the dataset but not individual vectors.

### Key Ownership: Threshold CKKS (Implemented)

The system supports two HE modes via the pluggable `HEProvider` interface:

1. **Direct mode** (default): A single party holds the full CKKS secret key. Simple, fast, suitable for single-tenant deployments or development.

2. **Threshold mode** (optional): The decryption key is split into shares held by an independent key committee of N nodes, with any t-of-N required to jointly decrypt. The DB server performs all HE computation with evaluation keys only (no decryption capability). Decryption uses Lattigo's `PublicKeySwitchProtocol` — each committee node produces a noise-flooded partial key-switch share, and shares are aggregated into a ciphertext under the querying client's ephemeral public key.

To use threshold mode, create a committee and pass the provider to the client:

```go
committee, _ := threshold.NewCommittee(5, 3)  // 3-of-5 threshold
provider, _ := crypto.NewThresholdHEProvider(committee, 4)
client, _ := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, provider)
```

**Performance impact** (Apple M4 benchmarks):
- Encryption: identical (~7ms, both use collective public key)
- HE dot product: identical (~82ms, evaluator-only)
- Decryption: ~29ms threshold vs ~9ms direct (3.3x overhead from protocol)
- Committee setup: 591–844ms (one-time cost)

The threshold overhead does not affect recall or precision. See `docs/THRESHOLD_CKKS.md` for the full architecture.

## Incremental Indexing

The index supports incremental updates without full re-clustering, following the "train once, assign many" pattern used by production vector databases (FAISS, Milvus, Pinecone).

### Post-Build Additions

After `Build()` creates the initial k-means index, subsequent `Add()` calls are handled incrementally:

1. **Normalize** the new vector (for cosine similarity)
2. **Find nearest centroid(s)** from the frozen centroid set — O(k·d) per vector
3. **Encrypt** with AES-256-GCM using the existing enterprise key
4. **Store** in the appropriate cluster bucket(s) in the blob store
5. Vector is **immediately searchable** — no rebuild needed

### Tier 2: Incremental Centroid Updates

Rather than freezing centroids, Opaque updates them incrementally using the running mean formula:

```
Add:    c_new = c_old + (x - c_old) / (n + 1)
Delete: c_new = (n · c_old - x) / (n - 1)
```

This keeps centroids accurate as new data arrives without re-clustering. After each update, HE centroid caches are refreshed so subsequent searches use the updated centroids.

**Batch optimization:** When adding multiple vectors via `AddBatch()`, centroid caches are refreshed once after all vectors are processed, not per-vector. This reduces incremental add cost from ~117ms/vec to ~300µs/vec.

This is the same approach used by Milvus (segment-level centroid updates) and Elasticsearch (incremental IVF maintenance).

### When to Rebuild

`Rebuild()` performs a full re-clustering and is recommended when:
- Data distribution has drifted significantly from the training set
- Cluster sizes become highly skewed (some clusters much larger than others)
- Recall has degraded below an acceptable threshold
- The dataset size has roughly doubled since the last build

### Deletion

Deletion uses a two-phase soft-delete pattern:
1. **Immediate**: `deletedIDs` map filters deleted vectors from search results at query time
2. **Deferred**: `Rebuild()` physically removes deleted vectors and reclaims storage

## Authentication

The auth service provides:
- User registration with enterprise association
- Token generation and validation (configurable TTL)
- Token refresh within configurable windows
- Credential distribution: AES key, centroids, enterprise config

Credentials are scoped per-enterprise. Each enterprise has:
- A unique AES-256 key for vector encryption
- A unique LSH seed for hash generation
- Its own set of cluster centroids
