# Project Opaque: Product Roadmap

Privacy-preserving vector search with multiple security tiers.

---

## Vision

Build the most comprehensive privacy-preserving vector search SDK, offering multiple tiers of privacy to match different use cases and threat models.

**Target users:**
- Developers building privacy-sensitive AI applications
- Enterprises with confidential data (healthcare, legal, finance)
- Decentralized applications (blockchain, Web3)
- Anyone who needs "private AI search"

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OPAQUE SDK                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   TIER 1    │  │   TIER 2    │  │  TIER 2.5   │  │  TIER 3   │ │
│  │   Query     │  │    Data     │  │ Hierarchical│  │  Enclave  │ │
│  │  Private    │  │  Private    │  │     FHE     │  │  Private  │ │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├───────────┤ │
│  │ HE query    │  │ Encrypted   │  │ Full FHE    │  │ TEE-based │ │
│  │ LSH filter  │  │ blobs       │  │ buckets     │  │ isolation │ │
│  │ ~66ms       │  │ ~200-500ms  │  │ ~2-5s       │  │ ~10-50ms  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
│                                                                     │
│  Privacy:  ████░░░░░░    ████████░░    ██████████    ██████████    │
│            Partial       High          Maximum       Maximum        │
│                                                                     │
│  Speed:    ██████████    ██████░░░░    ████░░░░░░    ██████████    │
│            Fast          Medium        Slow          Fast           │
│                                                                     │
│  Trust:    Server sees   Server sees   Server sees   Trust HW       │
│            vectors       nothing       nothing       (Intel/AWS)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tier Comparison

| Aspect | Tier 1 | Tier 2 | Tier 2.5 | Tier 3 |
|--------|--------|--------|----------|--------|
| **Name** | Query-Private | Data-Private | Hierarchical FHE | Enclave-Private |
| **Query encrypted** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Vectors encrypted** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Scores hidden** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Access patterns hidden** | ❌ No | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |
| **Server trust required** | Yes (sees vectors) | No | No | Hardware only |
| **Latency** | ~66ms | ~200-500ms | ~2-5s | ~10-50ms |
| **Max vectors** | Millions | Thousands/bucket | 100K+ | 50M+ |
| **Best for** | Shared DBs | Blockchain, Zero-trust | Max privacy | Enterprise |

---

## Tier 1: Query-Private (Current - Production Ready)

### Status: ✅ Core Complete, Needs Polish

### What It Does
- Client encrypts query with homomorphic encryption
- Server performs similarity search on encrypted query
- Server never sees the actual query or similarity scores
- Fast: ~66ms for 100K vectors

### What Server Sees
- LSH bucket (approximate query region)
- Which candidate vectors were considered
- Timing information

### What Server Does NOT See
- Exact query vector
- Similarity scores
- Final ranking

### Use Cases
- Search medical literature without revealing symptoms
- Search legal databases without revealing strategy
- Search market data without revealing trading signals

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| BFV Encryption (Lattigo) | ✅ Done | 128-bit security |
| LSH Index | ✅ Done | With hash masking |
| Client SDK | ✅ Done | Full encryption/decryption |
| Server Service | ✅ Done | Session management, HE compute |
| gRPC Proto | ✅ Done | Needs code generation |
| gRPC Server | 🔄 Partial | Needs wiring to service |
| Documentation | ✅ Done | README, benchmarks |

### Remaining Work (Tier 1)
1. Generate proto code (`protoc`)
2. Wire gRPC handlers to existing service
3. Add TLS configuration
4. Package as Go module with proper versioning
5. Create example applications

**Effort:** ~1-2 weeks

---

## Tier 2: Data-Private (Encrypted Blob Storage)

### Status: 📋 Planned

### What It Does
- Client encrypts vectors BEFORE uploading
- Server stores only encrypted blobs
- Search happens partially on server (bucket lookup), partially on client (decryption + scoring)
- Server NEVER sees plaintext vectors

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TIER 2: DATA-PRIVATE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STORAGE LAYER (Cloud / Blockchain / IPFS)                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  Bucket Index:                                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ LSH_bucket_0x1a2b → [blob_id_1, blob_id_2, ...]         │ │  │
│  │  │ LSH_bucket_0x3c4d → [blob_id_5, blob_id_6, ...]         │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  │  Encrypted Vectors:                                           │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ blob_id_1 → { iv, ciphertext, metadata_enc }            │ │  │
│  │  │ blob_id_2 → { iv, ciphertext, metadata_enc }            │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  CLIENT                                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Encryption Key (AES-256-GCM)                                │  │
│  │  LSH Planes (shared or derived)                              │  │
│  │  Local decryption + similarity computation                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Insertion:**
```
1. Client generates embedding locally
2. Client computes LSH hash locally
3. Client encrypts vector: AES-GCM(vector, key, iv)
4. Client uploads: (lsh_bucket, encrypted_blob, encrypted_metadata)
5. Storage indexes by LSH bucket
```

**Search:**
```
1. Client generates query embedding locally
2. Client computes LSH hash locally
3. Client requests bucket(s) from storage
4. Storage returns encrypted blobs (can't read them)
5. Client downloads blobs
6. Client decrypts vectors locally
7. Client computes similarity locally
8. Client returns top-K results
```

### What Server/Storage Sees
- LSH bucket access patterns
- Encrypted blobs (opaque)
- Timing information

### What Server Does NOT See
- Vector contents
- Query contents
- Similarity scores
- Final results

### Use Cases
- **Blockchain vector DB**: Store encrypted embeddings on-chain
- **Zero-trust cloud**: Don't trust your cloud provider
- **User-owned data**: Users control their own encryption keys
- **GDPR compliance**: Data encrypted at rest with user keys

### Implementation Plan

#### Phase 2.1: Core Encrypted Storage (Week 1-2)

**New packages to create:**

```
go/
├── pkg/
│   ├── encrypt/                 # NEW: Symmetric encryption
│   │   ├── encrypt.go           # AES-256-GCM encryption
│   │   ├── key.go               # Key derivation, rotation
│   │   └── encrypt_test.go
│   │
│   ├── blob/                    # NEW: Encrypted blob management
│   │   ├── blob.go              # Blob structure, serialization
│   │   ├── store.go             # Blob store interface
│   │   └── memory.go            # In-memory blob store
│   │
│   └── client/
│       └── client_tier2.go      # NEW: Tier 2 client methods
│
├── internal/
│   └── store/
│       ├── store.go             # Existing interface
│       ├── encrypted.go         # NEW: Encrypted store wrapper
│       └── s3.go                # NEW: S3 backend (optional)
```

**Key interfaces:**

```go
// pkg/encrypt/encrypt.go
type Encryptor interface {
    Encrypt(plaintext []byte) (ciphertext []byte, err error)
    Decrypt(ciphertext []byte) (plaintext []byte, err error)
    RotateKey(newKey []byte) error
}

// pkg/blob/store.go
type BlobStore interface {
    Put(bucket string, id string, blob *EncryptedBlob) error
    Get(bucket string, id string) (*EncryptedBlob, error)
    GetBucket(bucket string) ([]*EncryptedBlob, error)
    Delete(bucket string, id string) error
    ListBuckets() ([]string, error)
}

// Encrypted blob structure
type EncryptedBlob struct {
    ID             string
    IV             []byte    // Initialization vector
    Ciphertext     []byte    // Encrypted vector
    MetadataCipher []byte    // Encrypted metadata (optional)
    LSHBucket      string    // Bucket identifier
    CreatedAt      time.Time
}
```

#### Phase 2.2: Client-Side Search (Week 2-3)

**New client methods:**

```go
// pkg/client/client_tier2.go

// Tier2Client extends the base client with encrypted storage
type Tier2Client struct {
    *Client                    // Embed base client
    encryptor  encrypt.Encryptor
    blobStore  blob.BlobStore
}

// Insert encrypts and stores a vector
func (c *Tier2Client) Insert(id string, vector []float64, metadata map[string]any) error

// InsertBatch encrypts and stores multiple vectors
func (c *Tier2Client) InsertBatch(ids []string, vectors [][]float64, metadata []map[string]any) error

// Search performs client-side encrypted search
func (c *Tier2Client) Search(query []float64, topK int) ([]Result, error)

// SearchWithOptions allows tuning privacy/speed tradeoff
func (c *Tier2Client) SearchWithOptions(query []float64, opts SearchOptions) ([]Result, error)

type SearchOptions struct {
    TopK           int
    NumBuckets     int  // Fetch multiple buckets for privacy (adds noise)
    DecoyBuckets   int  // Additional random buckets (pure noise)
    MaxDownload    int  // Limit total blobs to download
}
```

#### Phase 2.3: Storage Backends (Week 3-4)

**Implement storage backends:**

| Backend | Priority | Use Case |
|---------|----------|----------|
| Memory | High | Testing, demos |
| File system | High | Local persistence |
| S3 | Medium | Cloud deployment |
| IPFS | Medium | Decentralized |
| Ethereum/Polygon | Low | Blockchain |
| Redis | Low | Caching layer |

**Backend interface:**

```go
// internal/store/backends.go

type StorageBackend interface {
    // Basic operations
    Put(key string, value []byte) error
    Get(key string) ([]byte, error)
    Delete(key string) error
    List(prefix string) ([]string, error)

    // Batch operations
    PutBatch(items map[string][]byte) error
    GetBatch(keys []string) (map[string][]byte, error)

    // Metadata
    Exists(key string) bool
    Size(key string) (int64, error)
}
```

#### Phase 2.4: Privacy Enhancements (Week 4)

**Additional privacy features:**

1. **Multi-bucket fetch**: Request multiple buckets to hide true interest
2. **Decoy queries**: Periodically fetch random buckets
3. **Bucket padding**: Ensure all buckets have similar sizes
4. **Timing obfuscation**: Add random delays

```go
// Privacy-enhanced search
func (c *Tier2Client) PrivateSearch(query []float64, opts PrivacyOptions) ([]Result, error)

type PrivacyOptions struct {
    // Noise injection
    DecoyBuckets    int           // Random buckets to fetch
    DecoyQueries    bool          // Background dummy queries

    // Timing
    MinLatency      time.Duration // Minimum response time
    JitterRange     time.Duration // Random delay range

    // Onion routing (future)
    UseTor          bool
    TorCircuits     int
}
```

### Tier 2 API Example

```go
package main

import (
    "github.com/opaque/opaque/go/pkg/client"
    "github.com/opaque/opaque/go/pkg/blob"
    "github.com/opaque/opaque/go/pkg/encrypt"
)

func main() {
    // Create encryption key (user controls this!)
    key := encrypt.DeriveKey("user-password", "salt")
    encryptor := encrypt.NewAESGCM(key)

    // Create blob store (could be S3, IPFS, blockchain, etc.)
    store := blob.NewMemoryStore()

    // Create Tier 2 client
    cfg := client.Tier2Config{
        Dimension:  128,
        LSHBits:    64,  // Fewer bits = more privacy
    }
    c, _ := client.NewTier2Client(cfg, encryptor, store)

    // Insert vectors (encrypted automatically)
    vectors := [][]float64{{0.1, 0.2, ...}, {0.3, 0.4, ...}}
    ids := []string{"doc1", "doc2"}
    c.InsertBatch(ids, vectors, nil)

    // Search (decryption + scoring happens locally)
    query := []float64{0.15, 0.25, ...}
    results, _ := c.Search(query, 10)

    // Privacy-enhanced search
    results, _ = c.PrivateSearch(query, client.PrivacyOptions{
        DecoyBuckets: 5,   // Fetch 5 extra random buckets
        MinLatency:   100 * time.Millisecond,
    })
}
```

### Performance Expectations

| Operation | Latency | Notes |
|-----------|---------|-------|
| Insert (single) | ~1ms | Encrypt + upload |
| Insert (batch 1000) | ~100ms | Parallelized |
| Search (100 vectors/bucket) | ~50ms | Download + decrypt + compute |
| Search (1000 vectors/bucket) | ~200ms | Depends on network |
| Search with 5 decoy buckets | ~300ms | 6x data, but private |

### Effort Estimate

| Phase | Work | Time |
|-------|------|------|
| 2.1 Core encryption | AES-GCM, blob structure | 1 week |
| 2.2 Client-side search | Tier2Client, search logic | 1 week |
| 2.3 Storage backends | Memory, file, S3 | 1 week |
| 2.4 Privacy features | Decoys, padding, timing | 1 week |
| **Total** | | **4 weeks** |

---

## Tier 2.5: Hierarchical FHE

### Status: 📋 Research/Planned

### What It Does
- Organize vectors into buckets with representative centroids
- Use FHE to compare query against ALL bucket representatives
- Client picks top buckets (server doesn't know which)
- Use FHE to score vectors in selected buckets
- Maximum privacy without hardware trust

### Architecture

```
100K vectors organized into 100 buckets
           │
           ▼
┌─────────────────────────────────────────────┐
│  Stage 1: FHE on 100 centroids (~500ms)     │
│  Server returns 100 encrypted scores        │
│  Client decrypts, picks top 5 buckets       │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Stage 2: FHE on 5000 vectors (~2-3s)       │
│  Server returns 5000 encrypted scores       │
│  Client decrypts, ranks locally             │
└─────────────────────────────────────────────┘
```

### Privacy Level
- Server sees: Which buckets were requested (but can add decoys)
- Server doesn't see: Query, scores, final ranking
- With decoys: Server learns almost nothing

### Use Cases
- Maximum privacy for sensitive applications
- When hardware trust (TEE) is not acceptable
- Acceptable for agentic AI (2-5s latency OK)

### Effort Estimate: 3-4 weeks

---

## Tier 3: Enclave-Private (TEE)

### Status: 📋 Planned (Enterprise Feature)

### What It Does
- Vectors encrypted at rest
- Decrypted ONLY inside Nitro Enclave
- Even AWS operators cannot see data
- Fast: native computation inside enclave (~10-50ms)

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    AWS NITRO ENCLAVE                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ENCLAVE (Isolated VM)                                  │ │
│  │                                                         │ │
│  │  • Decryption keys (via KMS attestation)               │ │
│  │  • Plaintext vectors (in-memory only)                  │ │
│  │  • Search engine (LSH + scoring)                       │ │
│  │  • Encrypted I/O                                       │ │
│  │                                                         │ │
│  │  NO: network, disk, debugging, operator access         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↑ vsock                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  PARENT (EC2)                                           │ │
│  │  • Proxy service (can't read payloads)                 │ │
│  │  • Encrypted vector storage (S3)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### What's Protected
- Vectors (encrypted at rest, decrypted only in enclave)
- Queries (encrypted end-to-end to enclave)
- Results (encrypted end-to-end from enclave)
- Even cloud operator cannot see data

### AWS Services Required
- EC2 with Nitro Enclave support
- KMS for key management
- S3 for encrypted vector storage
- ACM for TLS certificates

### Pricing
- ~$125-150/month for c5.xlarge running 24/7
- Enclaves themselves are free

### Use Cases
- Enterprise code search (Cursor-like)
- Healthcare AI with patient data
- Financial AI with trading data
- Government/defense applications

### Effort Estimate: 6-8 weeks

---

## Implementation Roadmap

```
2024 Q1                    2024 Q2                    2024 Q3
───────────────────────────────────────────────────────────────────

TIER 1 POLISH              TIER 2                     TIER 3
├─ gRPC wiring             ├─ Core encryption         ├─ Nitro setup
├─ TLS setup               ├─ Client-side search      ├─ KMS integration
├─ Go module               ├─ Storage backends        ├─ Enclave code
├─ Examples                ├─ Privacy features        ├─ Attestation
├─ Docs                    ├─ Blockchain demo         ├─ Enterprise docs
│                          │                          │
▼                          ▼                          ▼
MVP Release                Tier 2 Release             Enterprise Release


                           TIER 2.5 (parallel)
                           ├─ Hierarchical FHE
                           ├─ Bucket clustering
                           └─ Benchmarks
```

---

## Success Metrics

### Tier 1 (Query-Private)
- [ ] <100ms latency at 100K vectors
- [ ] >10 QPS sustained
- [ ] Zero plaintext query leakage
- [ ] Published Go module

### Tier 2 (Data-Private)
- [ ] <500ms latency for typical bucket sizes
- [ ] Zero plaintext vector leakage
- [ ] Working blockchain demo
- [ ] S3 + IPFS backends

### Tier 2.5 (Hierarchical FHE)
- [ ] <5s latency at 100K vectors
- [ ] Privacy: server learns only bucket access pattern
- [ ] Benchmarks showing privacy/speed tradeoff

### Tier 3 (Enclave-Private)
- [ ] <50ms latency inside enclave
- [ ] Cryptographic attestation working
- [ ] Pass enterprise security review
- [ ] SOC2 compliance path documented

---

## Open Questions

1. **Tier 2 bucket sizing**: Optimal bucket size for privacy vs download speed?
2. **Tier 2.5 parallelization**: How many cores needed for acceptable latency?
3. **Tier 3 cold start**: How long to load vectors into enclave on startup?
4. **Cross-tier**: Can we combine tiers? (e.g., Tier 2 + Tier 3)

---

## References

- [Lattigo](https://github.com/tuneinsight/lattigo) - Go HE library
- [AWS Nitro Enclaves](https://aws.amazon.com/ec2/nitro/nitro-enclaves/)
- [Cape Privacy](https://capeprivacy.com/) - Similar approach with TEE
- [BFV Scheme](https://eprint.iacr.org/2012/144) - Encryption scheme we use
