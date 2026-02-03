# Project Opaque - Go Production Implementation Plan

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Tech Stack](#tech-stack)
4. [Core Components](#core-components)
5. [Implementation Phases](#implementation-phases)
6. [Gotchas & Pitfalls](#gotchas--pitfalls)
7. [Performance Targets](#performance-targets)
8. [Security Considerations](#security-considerations)
9. [Deployment Guide](#deployment-guide)
10. [Testing Strategy](#testing-strategy)
11. [Monitoring & Observability](#monitoring--observability)
12. [Cost Estimation](#cost-estimation)
13. [Risk Assessment](#risk-assessment)

---

## Executive Summary

### What We're Building
A production-grade privacy-preserving vector search system in Go that allows clients to search encrypted queries against a vector database without revealing query content to the server.

### Why Go?
| Python Problem | Go Solution |
|----------------|-------------|
| GIL limits parallelism | True concurrency via goroutines |
| LightPHE is slow (pure Python) | Lattigo is 100x faster (optimized C/ASM) |
| 2-3s latency | Target: <300ms |
| Heavy dependencies (2GB+ torch) | Single static binary (~50MB) |
| Memory overhead | Low footprint, no GC pressure |

### Expected Outcomes
- **Latency**: 2500ms → **<300ms** (8x improvement)
- **Throughput**: 1 QPS → **100+ QPS**
- **Availability**: 99.9% with horizontal scaling
- **Security**: 128-bit cryptographic security level

---

## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OPAQUE SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    CLIENTS                           OPAQUE CLUSTER                             │
│   ┌────────┐                        ┌─────────────────────────────────────────┐ │
│   │ Mobile │                        │                                         │ │
│   │  App   │                        │  ┌─────────┐     ┌──────────────────┐  │ │
│   └───┬────┘                        │  │  Load   │     │   Search Service │  │ │
│       │         gRPC/TLS            │  │Balancer │────►│   ┌───────────┐  │  │ │
│   ┌───┴────┐    ◄──────────────────►│  │ (Envoy) │     │   │    LSH    │  │  │ │
│   │  Web   │                        │  └─────────┘     │   │  Service  │  │  │ │
│   │  App   │                        │       │          │   └─────┬─────┘  │  │ │
│   └───┬────┘                        │       │          │         │        │  │ │
│       │                             │       ▼          │   ┌─────▼─────┐  │  │ │
│   ┌───┴────┐                        │  ┌─────────┐     │   │    PHE    │  │  │ │
│   │  CLI   │                        │  │  Redis  │     │   │  Engine   │  │  │ │
│   │ Tools  │                        │  │ (Keys/  │     │   └─────┬─────┘  │  │ │
│   └────────┘                        │  │ Cache)  │     │         │        │  │ │
│                                     │  └─────────┘     │   ┌─────▼─────┐  │  │ │
│   CLIENT SDK                        │                  │   │  Vector   │  │  │ │
│   ┌────────────────┐                │                  │   │   Store   │  │  │ │
│   │ opaque-sdk-go  │                │                  │   │ (Milvus)  │  │  │ │
│   ├────────────────┤                │                  │   └───────────┘  │  │ │
│   │ • Key Gen      │                │                  └──────────────────┘  │ │
│   │ • Encryption   │                │                                         │ │
│   │ • Decryption   │                │  ┌──────────────────────────────────┐   │ │
│   │ • LSH Hashing  │                │  │       Embedding Service          │   │ │
│   │ • gRPC Client  │                │  │  (Triton / ONNX / External API)  │   │ │
│   └────────────────┘                │  └──────────────────────────────────┘   │ │
│                                     │                                         │ │
│                                     └─────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SEARCH FLOW                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  CLIENT                                              SERVER                      │
│                                                                                 │
│  1. User enters search query                                                    │
│     "cancer treatment side effects"                                             │
│            │                                                                    │
│            ▼                                                                    │
│  2. Embed query (local or via API)                                             │
│     query_vector = [0.12, -0.45, ...]                                          │
│            │                                                                    │
│            ▼                                                                    │
│  3. Compute LSH hash (CLIENT-SIDE)                                             │
│     lsh_hash = LSH(query_vector)  ───────────────►  4. LSH Lookup              │
│     [Does NOT reveal query!]                           Find candidates by      │
│                                                        hash similarity         │
│                                              ◄─────────  candidate_ids [100]   │
│            │                                                                    │
│            ▼                                                                    │
│  5. Encrypt query vector                                                        │
│     E(query) = Encrypt(query_vector)  ───────────►  6. Compute E(scores)       │
│     [2048-bit Paillier / Lattigo BFV]                  For each candidate:     │
│                                                        E(score) = E(q) · v     │
│                                                        (homomorphic dot prod)  │
│                                              ◄─────────  E(scores) [20]        │
│            │                                            (only top-20 by LSH)   │
│            ▼                                                                    │
│  7. Decrypt scores (CLIENT-SIDE)                                               │
│     scores = Decrypt(E(scores))                                                │
│            │                                                                    │
│            ▼                                                                    │
│  8. Return top-k results                                                        │
│     [{id: "doc_42", score: 0.95}, ...]                                         │
│                                                                                 │
│  ═══════════════════════════════════════════════════════════════════════════   │
│  PRIVACY GUARANTEES:                                                            │
│  ✓ Server NEVER sees query vector (encrypted)                                  │
│  ✓ Server NEVER sees similarity scores (encrypted)                             │
│  ✓ LSH hash reveals only approximate "bucket" (acceptable leakage)             │
│  ✓ Client NEVER downloads full database                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Core Technologies

| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| Language | Go | 1.21+ | Performance, concurrency, deployment simplicity |
| PHE Library | Lattigo | v5.x | Best-in-class Go HE library, 100x faster than Python |
| Vector Store | Milvus | 2.3+ | Scalable, GPU support, filtering, open-source |
| LSH Index | Custom Go | - | Simple, no CGO dependency, SIMD optimized |
| RPC | gRPC | 1.58+ | Efficient binary protocol, streaming, codegen |
| Serialization | Protocol Buffers | 3 | Compact, fast, schema evolution |
| Cache | Redis | 7+ | Session keys, hot vectors, rate limiting |
| Gateway | Envoy | 1.28+ | Load balancing, mTLS, observability |
| Embeddings | Triton / External | - | Keep separate from search service |

### Alternative Considerations

| Component | Alternative | Trade-off |
|-----------|-------------|-----------|
| Milvus | Qdrant | Qdrant: Rust, simpler. Milvus: More features, GPU |
| Lattigo | Microsoft SEAL | SEAL: C++, faster. Lattigo: Pure Go, easier deploy |
| Redis | DragonflyDB | Dragonfly: Faster. Redis: More mature, ecosystem |
| Envoy | Traefik | Traefik: Simpler. Envoy: More control |

### Why Lattigo over SEAL/HElib?

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHE LIBRARY COMPARISON                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Library    │ Language │ Speed    │ Deploy   │ Maintenance     │
│  ───────────┼──────────┼──────────┼──────────┼─────────────    │
│  LightPHE   │ Python   │ 1x       │ Easy     │ Active          │
│  Lattigo    │ Go       │ 50-100x  │ Easy     │ Active (EPFL)   │
│  SEAL       │ C++      │ 100-200x │ Hard     │ Active (MSFT)   │
│  HElib      │ C++      │ 100-200x │ Hard     │ Moderate (IBM)  │
│  concrete   │ Rust     │ 50-100x  │ Medium   │ Active (Zama)   │
│                                                                 │
│  CHOICE: Lattigo                                                │
│  • Pure Go = no CGO hassles, easy cross-compilation            │
│  • 50-100x faster than Python = meets latency targets          │
│  • Active maintenance by EPFL cryptography lab                 │
│  • Supports BFV, BGV, CKKS schemes                             │
│  • Good documentation and examples                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Client SDK (`opaque-sdk-go`)

```go
// pkg/client/client.go

package client

import (
    "context"
    "sync"

    "github.com/tuneinsight/lattigo/v5/core/rlwe"
    "github.com/tuneinsight/lattigo/v5/schemes/bfv"
    "google.golang.org/grpc"

    pb "github.com/yourorg/opaque/api/proto"
)

// Config holds client configuration
type Config struct {
    ServerAddr     string
    SecurityLevel  int    // 128 or 256 bit
    SessionTTL     int    // seconds
    MaxCandidates  int
    DecryptTopN    int    // Optimization: only decrypt top N
}

// Client is the main SDK entry point
type Client struct {
    config    Config
    params    bfv.Parameters
    secretKey *rlwe.SecretKey
    publicKey *rlwe.PublicKey
    encoder   *bfv.Encoder
    encryptor *rlwe.Encryptor
    decryptor *rlwe.Decryptor
    evaluator *bfv.Evaluator

    // LSH
    lshPlanes [][]float64

    // gRPC
    conn      *grpc.ClientConn
    rpcClient pb.OpaqueSearchClient
    sessionID string

    mu sync.RWMutex
}

// NewClient creates a new Opaque client
func NewClient(cfg Config) (*Client, error) {
    // Initialize Lattigo parameters
    // PN14QP438 gives ~128-bit security
    params, err := bfv.NewParametersFromLiteral(bfv.PN14QP438)
    if err != nil {
        return nil, fmt.Errorf("failed to create parameters: %w", err)
    }

    // Generate key pair
    kgen := rlwe.NewKeyGenerator(params)
    sk, pk := kgen.GenKeyPairNew()

    // Create client
    c := &Client{
        config:    cfg,
        params:    params,
        secretKey: sk,
        publicKey: pk,
        encoder:   bfv.NewEncoder(params),
        encryptor: rlwe.NewEncryptor(params, pk),
        decryptor: rlwe.NewDecryptor(params, sk),
        evaluator: bfv.NewEvaluator(params, nil),
    }

    // Initialize LSH planes
    c.initLSH(256) // 256-bit LSH

    // Connect to server
    if err := c.connect(); err != nil {
        return nil, err
    }

    // Register public key
    if err := c.registerKey(); err != nil {
        return nil, err
    }

    return c, nil
}

// Search performs privacy-preserving search
func (c *Client) Search(ctx context.Context, queryVector []float64, topK int) ([]Result, error) {
    // 1. Compute LSH hash locally
    lshHash := c.computeLSHHash(queryVector)

    // 2. Get candidates from server
    candidates, err := c.rpcClient.GetCandidates(ctx, &pb.CandidateRequest{
        SessionId:     c.sessionID,
        LshHash:       lshHash,
        NumCandidates: int32(c.config.MaxCandidates),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to get candidates: %w", err)
    }

    // 3. Optimization: Only process top N by LSH distance
    numToProcess := min(c.config.DecryptTopN, len(candidates.Ids))
    candidateIDs := candidates.Ids[:numToProcess]

    // 4. Encrypt query
    encryptedQuery, err := c.encryptVector(queryVector)
    if err != nil {
        return nil, fmt.Errorf("failed to encrypt query: %w", err)
    }

    // 5. Server computes encrypted scores
    scores, err := c.rpcClient.ComputeScores(ctx, &pb.ScoreRequest{
        SessionId:      c.sessionID,
        EncryptedQuery: c.serializeCiphertext(encryptedQuery),
        CandidateIds:   candidateIDs,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to compute scores: %w", err)
    }

    // 6. Decrypt scores in parallel
    decryptedScores, err := c.decryptScoresParallel(scores.EncryptedScores)
    if err != nil {
        return nil, fmt.Errorf("failed to decrypt scores: %w", err)
    }

    // 7. Sort and return top-k
    return c.topKResults(candidateIDs, decryptedScores, topK), nil
}

// decryptScoresParallel decrypts scores using goroutines
func (c *Client) decryptScoresParallel(encrypted [][]byte) ([]float64, error) {
    scores := make([]float64, len(encrypted))
    errors := make([]error, len(encrypted))

    var wg sync.WaitGroup

    for i, encBytes := range encrypted {
        wg.Add(1)
        go func(idx int, data []byte) {
            defer wg.Done()

            ct, err := c.deserializeCiphertext(data)
            if err != nil {
                errors[idx] = err
                return
            }

            pt := c.decryptor.DecryptNew(ct)
            result := make([]uint64, 1)
            c.encoder.Decode(pt, result)
            scores[idx] = float64(result[0]) / math.Pow(2, 40) // Fixed-point decode
        }(i, encBytes)
    }

    wg.Wait()

    // Check for errors
    for _, err := range errors {
        if err != nil {
            return nil, err
        }
    }

    return scores, nil
}

// computeLSHHash computes locality-sensitive hash
func (c *Client) computeLSHHash(vector []float64) []byte {
    hash := make([]byte, len(c.lshPlanes)/8)

    for i, plane := range c.lshPlanes {
        dot := dotProduct(vector, plane)
        if dot > 0 {
            hash[i/8] |= (1 << (i % 8))
        }
    }

    return hash
}
```

### 2. Search Service (Server)

```go
// internal/service/search.go

package service

import (
    "context"
    "sync"
    "time"

    "github.com/tuneinsight/lattigo/v5/core/rlwe"
    "github.com/tuneinsight/lattigo/v5/schemes/bfv"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"

    pb "github.com/yourorg/opaque/api/proto"
    "github.com/yourorg/opaque/internal/lsh"
    "github.com/yourorg/opaque/internal/store"
)

type SearchService struct {
    pb.UnimplementedOpaqueSearchServer

    lshIndex    *lsh.Index
    vectorStore store.VectorStore
    sessions    *SessionManager
    tracer      trace.Tracer

    // Lattigo (shared params, no secret key on server!)
    params    bfv.Parameters
    encoder   *bfv.Encoder
    evaluator *bfv.Evaluator
}

func NewSearchService(cfg Config) (*SearchService, error) {
    // Initialize Lattigo (MUST match client params)
    params, _ := bfv.NewParametersFromLiteral(bfv.PN14QP438)

    // Connect to vector store
    vs, err := store.NewMilvusStore(cfg.MilvusAddr)
    if err != nil {
        return nil, err
    }

    // Load LSH index
    lshIdx, err := lsh.LoadIndex(cfg.LSHIndexPath)
    if err != nil {
        return nil, err
    }

    return &SearchService{
        lshIndex:    lshIdx,
        vectorStore: vs,
        sessions:    NewSessionManager(cfg.RedisAddr),
        tracer:      otel.Tracer("search-service"),
        params:      params,
        encoder:     bfv.NewEncoder(params),
        evaluator:   bfv.NewEvaluator(params, nil),
    }, nil
}

// RegisterKey stores client's public key
func (s *SearchService) RegisterKey(ctx context.Context, req *pb.RegisterKeyRequest) (*pb.RegisterKeyResponse, error) {
    ctx, span := s.tracer.Start(ctx, "RegisterKey")
    defer span.End()

    // Generate session ID
    sessionID := generateSessionID()

    // Store public key in Redis with TTL
    err := s.sessions.Store(sessionID, req.PublicKey, time.Duration(req.SessionTtlSeconds)*time.Second)
    if err != nil {
        return nil, err
    }

    return &pb.RegisterKeyResponse{
        SessionId: sessionID,
    }, nil
}

// GetCandidates performs LSH lookup
func (s *SearchService) GetCandidates(ctx context.Context, req *pb.CandidateRequest) (*pb.CandidateResponse, error) {
    ctx, span := s.tracer.Start(ctx, "GetCandidates")
    defer span.End()

    // Validate session
    if _, err := s.sessions.Get(req.SessionId); err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid session")
    }

    // LSH lookup - O(1) amortized
    ids, distances := s.lshIndex.Search(req.LshHash, int(req.NumCandidates))

    return &pb.CandidateResponse{
        Ids:       ids,
        Distances: distances,
    }, nil
}

// ComputeScores computes encrypted dot products
func (s *SearchService) ComputeScores(ctx context.Context, req *pb.ScoreRequest) (*pb.ScoreResponse, error) {
    ctx, span := s.tracer.Start(ctx, "ComputeScores")
    defer span.End()

    // Validate session and get public key
    pkBytes, err := s.sessions.Get(req.SessionId)
    if err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid session")
    }

    // Deserialize encrypted query
    encryptedQuery, err := s.deserializeCiphertext(req.EncryptedQuery)
    if err != nil {
        return nil, status.Error(codes.InvalidArgument, "invalid ciphertext")
    }

    // Fetch vectors from store
    vectors, err := s.vectorStore.GetByIDs(ctx, req.CandidateIds)
    if err != nil {
        return nil, err
    }

    // Compute encrypted scores in parallel
    encryptedScores := make([][]byte, len(vectors))
    errors := make([]error, len(vectors))

    var wg sync.WaitGroup
    sem := make(chan struct{}, 16) // Limit concurrency

    for i, vec := range vectors {
        wg.Add(1)
        sem <- struct{}{}

        go func(idx int, vector []float64) {
            defer wg.Done()
            defer func() { <-sem }()

            // Compute homomorphic dot product
            score, err := s.homomorphicDotProduct(encryptedQuery, vector)
            if err != nil {
                errors[idx] = err
                return
            }

            encryptedScores[idx], err = s.serializeCiphertext(score)
            if err != nil {
                errors[idx] = err
            }
        }(i, vec)
    }

    wg.Wait()

    // Check errors
    for _, err := range errors {
        if err != nil {
            return nil, err
        }
    }

    return &pb.ScoreResponse{
        EncryptedScores: encryptedScores,
        Ids:             req.CandidateIds,
    }, nil
}

// homomorphicDotProduct computes E(q · v) from E(q) and plaintext v
func (s *SearchService) homomorphicDotProduct(encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error) {
    // Encode vector as plaintext
    // Scale to fixed-point representation
    scaled := make([]uint64, len(vector))
    for i, v := range vector {
        scaled[i] = uint64(v * math.Pow(2, 40)) // 40-bit precision
    }

    pt := bfv.NewPlaintext(s.params, encQuery.Level())
    s.encoder.Encode(scaled, pt)

    // Multiply: E(q) * v = E(q * v) component-wise
    result := s.evaluator.MulNew(encQuery, pt)

    // Sum components using rotations
    // result = E(q[0]*v[0] + q[1]*v[1] + ... + q[n]*v[n])
    n := len(vector)
    for i := 1; i < n; i *= 2 {
        rotated, _ := s.evaluator.RotateColumnsNew(result, i)
        s.evaluator.Add(result, rotated, result)
    }

    return result, nil
}
```

### 3. LSH Index

```go
// internal/lsh/index.go

package lsh

import (
    "encoding/binary"
    "math/bits"
    "sort"
    "sync"

    "github.com/viterin/vek" // SIMD vector operations
)

// Index is a locality-sensitive hash index
type Index struct {
    planes    [][]float32         // Random hyperplanes
    buckets   map[uint64][]string // Hash -> document IDs
    vectors   map[string][]float32 // ID -> vector (for re-ranking)
    numBits   int
    dimension int
    mu        sync.RWMutex
}

// NewIndex creates a new LSH index
func NewIndex(dimension, numBits int, seed int64) *Index {
    rng := rand.New(rand.NewSource(seed))

    idx := &Index{
        planes:    make([][]float32, numBits),
        buckets:   make(map[uint64][]string),
        vectors:   make(map[string][]float32),
        numBits:   numBits,
        dimension: dimension,
    }

    // Generate random hyperplanes
    for i := 0; i < numBits; i++ {
        plane := make([]float32, dimension)
        var norm float32
        for j := 0; j < dimension; j++ {
            plane[j] = float32(rng.NormFloat64())
            norm += plane[j] * plane[j]
        }
        // Normalize
        norm = float32(math.Sqrt(float64(norm)))
        for j := range plane {
            plane[j] /= norm
        }
        idx.planes[i] = plane
    }

    return idx
}

// Add adds vectors to the index
func (idx *Index) Add(ids []string, vectors [][]float32) {
    idx.mu.Lock()
    defer idx.mu.Unlock()

    for i, id := range ids {
        vec := vectors[i]
        hash := idx.hash(vec)

        idx.buckets[hash] = append(idx.buckets[hash], id)
        idx.vectors[id] = vec
    }
}

// Search finds k nearest candidates by Hamming distance
func (idx *Index) Search(queryHash []byte, k int) ([]string, []float32) {
    idx.mu.RLock()
    defer idx.mu.RUnlock()

    queryHashInt := binary.BigEndian.Uint64(padTo8Bytes(queryHash))

    type candidate struct {
        id       string
        distance int
    }

    candidates := make([]candidate, 0, k*10)

    // Compute Hamming distance to all buckets
    for bucketHash, ids := range idx.buckets {
        dist := bits.OnesCount64(queryHashInt ^ bucketHash)
        for _, id := range ids {
            candidates = append(candidates, candidate{id: id, distance: dist})
        }
    }

    // Sort by Hamming distance
    sort.Slice(candidates, func(i, j int) bool {
        return candidates[i].distance < candidates[j].distance
    })

    // Return top k
    n := min(k, len(candidates))
    ids := make([]string, n)
    distances := make([]float32, n)

    for i := 0; i < n; i++ {
        ids[i] = candidates[i].id
        distances[i] = float32(candidates[i].distance)
    }

    return ids, distances
}

// hash computes LSH signature using SIMD
func (idx *Index) hash(vector []float32) uint64 {
    var hash uint64

    for i, plane := range idx.planes {
        // SIMD dot product (using viterin/vek)
        dot := vek.Dot(vector, plane)
        if dot > 0 {
            hash |= (1 << i)
        }
    }

    return hash
}

// HashBytes returns hash as bytes (for gRPC)
func (idx *Index) HashBytes(vector []float32) []byte {
    hash := idx.hash(vector)
    buf := make([]byte, 8)
    binary.BigEndian.PutUint64(buf, hash)
    return buf
}
```

### 4. Vector Store Interface

```go
// internal/store/store.go

package store

import (
    "context"
)

// VectorStore is the interface for vector storage backends
type VectorStore interface {
    // GetByIDs retrieves vectors by their IDs
    GetByIDs(ctx context.Context, ids []string) ([][]float64, error)

    // Add adds vectors to the store
    Add(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error

    // Delete removes vectors by ID
    Delete(ctx context.Context, ids []string) error

    // Count returns total vector count
    Count(ctx context.Context) (int64, error)

    // Close closes the connection
    Close() error
}

// MilvusStore implements VectorStore using Milvus
type MilvusStore struct {
    client     *milvus.Client
    collection string
}

func NewMilvusStore(addr, collection string) (*MilvusStore, error) {
    client, err := milvus.NewClient(context.Background(), milvus.Config{
        Address: addr,
    })
    if err != nil {
        return nil, err
    }

    return &MilvusStore{
        client:     client,
        collection: collection,
    }, nil
}

func (s *MilvusStore) GetByIDs(ctx context.Context, ids []string) ([][]float64, error) {
    // Query Milvus for vectors
    result, err := s.client.Query(
        ctx,
        s.collection,
        "",
        []string{"vector"},
        milvus.WithIDs(ids),
    )
    if err != nil {
        return nil, err
    }

    // Extract vectors
    vectors := make([][]float64, len(ids))
    for i, row := range result {
        vectors[i] = row.Get("vector").([]float64)
    }

    return vectors, nil
}
```

---

## Implementation Phases

### Phase 1: Core Library (Weeks 1-3)

```
Week 1: Lattigo Integration
├── [ ] Set up Go module structure
├── [ ] Implement BFV parameter selection
├── [ ] Implement key generation
├── [ ] Implement encryption/decryption
├── [ ] Implement homomorphic dot product
├── [ ] Unit tests for crypto operations
└── [ ] Benchmark vs LightPHE

Week 2: LSH Implementation
├── [ ] Implement random hyperplane LSH
├── [ ] Add SIMD optimization (viterin/vek)
├── [ ] Implement multi-probe LSH (optional)
├── [ ] Serialization/deserialization
├── [ ] Unit tests
└── [ ] Benchmark hash and search

Week 3: Integration & Testing
├── [ ] End-to-end test: encrypt → search → decrypt
├── [ ] Accuracy comparison with Python version
├── [ ] Memory profiling
├── [ ] Document API
└── [ ] Create example usage
```

### Phase 2: Server Implementation (Weeks 4-6)

```
Week 4: gRPC Service
├── [ ] Define protobuf schemas
├── [ ] Implement RegisterKey RPC
├── [ ] Implement GetCandidates RPC
├── [ ] Implement ComputeScores RPC
├── [ ] Add OpenTelemetry tracing
└── [ ] Unit tests with mocks

Week 5: Storage Integration
├── [ ] Implement Milvus store
├── [ ] Implement Redis session manager
├── [ ] Add connection pooling
├── [ ] Implement health checks
└── [ ] Integration tests

Week 6: Performance & Reliability
├── [ ] Add connection retry logic
├── [ ] Implement circuit breaker
├── [ ] Add rate limiting
├── [ ] Load testing (wrk, ghz)
└── [ ] Optimize hot paths
```

### Phase 3: Client SDK (Weeks 7-8)

```
Week 7: Go SDK
├── [ ] Implement Client struct
├── [ ] Add automatic reconnection
├── [ ] Implement Search method
├── [ ] Add context cancellation
├── [ ] Error handling and retries
└── [ ] Comprehensive tests

Week 8: Additional SDKs
├── [ ] Python client (gRPC)
├── [ ] TypeScript client (gRPC-web)
├── [ ] CLI tool
├── [ ] SDK documentation
└── [ ] Example applications
```

### Phase 4: Production Readiness (Weeks 9-12)

```
Week 9: Deployment
├── [ ] Dockerfile (multi-stage build)
├── [ ] Kubernetes manifests
├── [ ] Helm chart
├── [ ] CI/CD pipeline (GitHub Actions)
└── [ ] Container security scan

Week 10: Observability
├── [ ] Prometheus metrics
├── [ ] Grafana dashboards
├── [ ] Distributed tracing (Jaeger)
├── [ ] Structured logging (zerolog)
└── [ ] Alerting rules

Week 11: Security Hardening
├── [ ] mTLS configuration
├── [ ] Secrets management (Vault)
├── [ ] Security audit
├── [ ] Penetration testing
└── [ ] Compliance documentation

Week 12: Launch Preparation
├── [ ] Load testing at scale
├── [ ] Chaos engineering (Litmus)
├── [ ] Runbook documentation
├── [ ] On-call training
└── [ ] Gradual rollout plan
```

---

## Gotchas & Pitfalls

### 1. Lattigo Parameter Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Wrong parameters = broken security or poor performance              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Lattigo Parameter Sets:                                                     │
│                                                                             │
│ PN12QP109  │ 128-bit │ Fast      │ Small vectors only (<1024 slots)        │
│ PN13QP218  │ 128-bit │ Medium    │ Good for most use cases                 │
│ PN14QP438  │ 128-bit │ Slower    │ Larger vectors, more operations         │
│ PN15QP880  │ 128-bit │ Slowest   │ Very large computations                 │
│                                                                             │
│ RECOMMENDATION: Start with PN14QP438                                        │
│ • Supports vectors up to 8192 dimensions                                    │
│ • 128-bit security                                                          │
│ • Good balance of speed and capability                                      │
│                                                                             │
│ CRITICAL: Client and server MUST use identical parameters!                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Fixed-Point Encoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: BFV/BGV work with integers, not floats!                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PROBLEM:                                                                    │
│   Vectors are float64: [0.123, -0.456, 0.789, ...]                         │
│   BFV operates on uint64: [123, 456, 789, ...]                             │
│                                                                             │
│ SOLUTION: Fixed-point encoding                                              │
│                                                                             │
│   // Encode: float → uint64                                                 │
│   scale := math.Pow(2, 40)  // 40 bits of precision                        │
│   encoded := uint64((value + offset) * scale)                              │
│                                                                             │
│   // Decode: uint64 → float                                                │
│   decoded := (float64(encoded) / scale) - offset                           │
│                                                                             │
│ WATCH OUT FOR:                                                              │
│   • Overflow: value * scale must fit in uint64                             │
│   • Negative numbers: add offset to make positive                          │
│   • Precision loss: choose scale carefully                                 │
│   • Noise growth: more operations = more noise = less precision            │
│                                                                             │
│ RECOMMENDATION:                                                             │
│   • Normalize vectors to [-1, 1] range                                     │
│   • Add offset of 1.0 to make [0, 2]                                       │
│   • Use 40-bit scale (precision ≈ 1e-12)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Ciphertext Size & Serialization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Ciphertexts are HUGE compared to plaintexts                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Size comparison (128-dim vector):                                           │
│   Plaintext:   128 × 8 bytes = 1 KB                                        │
│   Ciphertext:  ~500 KB - 2 MB (depending on parameters)                    │
│                                                                             │
│ IMPLICATIONS:                                                               │
│   • Network bandwidth is significant                                        │
│   • Serialize efficiently (don't use JSON!)                                │
│   • Consider compression (limited benefit for random-looking data)         │
│   • gRPC streaming for large responses                                     │
│                                                                             │
│ SERIALIZATION CODE:                                                         │
│                                                                             │
│   // Serialize                                                              │
│   func serializeCiphertext(ct *rlwe.Ciphertext) ([]byte, error) {          │
│       buf := new(bytes.Buffer)                                              │
│       _, err := ct.WriteTo(buf)                                            │
│       return buf.Bytes(), err                                              │
│   }                                                                         │
│                                                                             │
│   // Deserialize                                                            │
│   func deserializeCiphertext(data []byte, params rlwe.Parameters)          │
│       (*rlwe.Ciphertext, error) {                                          │
│       ct := rlwe.NewCiphertext(params, 1, params.MaxLevel())               │
│       _, err := ct.ReadFrom(bytes.NewReader(data))                         │
│       return ct, err                                                        │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Noise Budget Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Each HE operation adds noise; too much noise = wrong results        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ NOISE GROWTH:                                                               │
│   • Addition: noise grows slowly (additive)                                │
│   • Multiplication: noise grows fast (multiplicative)                      │
│   • Rotation: noise grows moderately                                       │
│                                                                             │
│ DOT PRODUCT OPERATIONS:                                                     │
│   E(q) × v         = 1 multiplication (high noise growth)                  │
│   Σ (rotate + add) = log(n) rotations + additions                          │
│                                                                             │
│ TOTAL: 1 mult + log(n) rotations + log(n) additions                        │
│                                                                             │
│ FOR 128-DIM VECTOR: 1 mult + 7 rotations + 7 adds                          │
│                                                                             │
│ MITIGATION:                                                                 │
│   • Use larger parameters (more noise budget)                              │
│   • Reduce vector dimension via PCA                                        │
│   • Use CKKS instead of BFV for floating-point (native)                    │
│   • Bootstrapping (expensive, usually not needed for dot product)          │
│                                                                             │
│ MONITORING:                                                                 │
│   // Check remaining noise budget                                          │
│   level := ct.Level()  // Higher = more budget remaining                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. LSH Accuracy vs Speed Tradeoff

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: More LSH bits ≠ always better                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LSH BITS vs BEHAVIOR:                                                       │
│                                                                             │
│   64 bits:   Fast hash, many collisions, low precision                     │
│   128 bits:  Balanced                                                       │
│   256 bits:  Slow hash, few collisions, high precision                     │
│   512 bits:  Very slow, possibly no collisions (bad!)                      │
│                                                                             │
│ PROBLEM: Too many bits → query falls in empty bucket → no results          │
│                                                                             │
│ SOLUTION: Multi-probe LSH                                                   │
│   • Hash query                                                              │
│   • Also check neighboring buckets (flip 1-2 bits)                         │
│   • Increases recall without increasing hash size                          │
│                                                                             │
│ TUNING GUIDE:                                                               │
│   Database size   │ Recommended bits │ Multi-probe                         │
│   ────────────────┼──────────────────┼─────────────                        │
│   < 10K           │ 64-128           │ Optional                             │
│   10K - 100K      │ 128-256          │ Recommended                          │
│   100K - 1M       │ 256-512          │ Required                             │
│   > 1M            │ Multiple tables  │ Required                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6. Session Management & Key Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Improper key management = security vulnerability                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ RULES:                                                                      │
│                                                                             │
│ 1. NEVER store secret keys on server                                       │
│    • Server only needs public key for operations                           │
│    • Secret key stays on client ONLY                                       │
│                                                                             │
│ 2. Session TTL should be short                                             │
│    • Recommended: 1-24 hours                                               │
│    • Force re-registration periodically                                    │
│                                                                             │
│ 3. One key pair per client session                                         │
│    • Don't reuse keys across sessions                                      │
│    • Don't share keys between users                                        │
│                                                                             │
│ 4. Secure key generation                                                   │
│    • Use crypto/rand, not math/rand                                        │
│    • Lattigo handles this correctly by default                             │
│                                                                             │
│ 5. Key rotation                                                            │
│    • Rotate keys if client suspects compromise                             │
│    • Implement key revocation mechanism                                    │
│                                                                             │
│ SESSION FLOW:                                                               │
│                                                                             │
│   Client                              Server                                │
│     │                                    │                                  │
│     │── Generate (sk, pk) locally ──►    │                                  │
│     │                                    │                                  │
│     │── RegisterKey(pk, ttl=1h) ────────►│── Store pk in Redis with TTL    │
│     │                                    │                                  │
│     │◄─── session_id ───────────────────│                                  │
│     │                                    │                                  │
│     │── Search(session_id, E(q)) ───────►│── Lookup pk by session_id       │
│     │                                    │── Compute with pk                │
│     │◄─── E(scores) ────────────────────│                                  │
│     │                                    │                                  │
│     │── Decrypt with sk locally ──►      │                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7. Concurrency & Goroutine Leaks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Unbounded goroutines = OOM; Blocked goroutines = resource leak      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ BAD: Unbounded parallelism                                                  │
│                                                                             │
│   for _, vec := range vectors {                                            │
│       go computeScore(vec)  // 10,000 goroutines? Bad!                     │
│   }                                                                         │
│                                                                             │
│ GOOD: Bounded parallelism with semaphore                                   │
│                                                                             │
│   sem := make(chan struct{}, runtime.NumCPU())                             │
│   var wg sync.WaitGroup                                                    │
│                                                                             │
│   for _, vec := range vectors {                                            │
│       wg.Add(1)                                                            │
│       sem <- struct{}{} // Acquire                                         │
│                                                                             │
│       go func(v []float64) {                                               │
│           defer wg.Done()                                                  │
│           defer func() { <-sem }() // Release                              │
│           computeScore(v)                                                  │
│       }(vec)                                                               │
│   }                                                                         │
│   wg.Wait()                                                                │
│                                                                             │
│ BETTER: Worker pool pattern                                                │
│                                                                             │
│   pool := pond.New(runtime.NumCPU(), 1000) // github.com/alitto/pond       │
│   defer pool.StopAndWait()                                                 │
│                                                                             │
│   for _, vec := range vectors {                                            │
│       v := vec // Capture                                                  │
│       pool.Submit(func() { computeScore(v) })                              │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8. gRPC Message Size Limits

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Default gRPC message limit is 4MB; ciphertexts can exceed this      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PROBLEM:                                                                    │
│   • Default max message size: 4 MB                                         │
│   • 20 ciphertexts × 500KB each = 10 MB → REJECTED                        │
│                                                                             │
│ SOLUTION 1: Increase limits (simple)                                       │
│                                                                             │
│   // Server                                                                 │
│   grpc.NewServer(                                                          │
│       grpc.MaxRecvMsgSize(50 * 1024 * 1024), // 50 MB                      │
│       grpc.MaxSendMsgSize(50 * 1024 * 1024),                               │
│   )                                                                         │
│                                                                             │
│   // Client                                                                 │
│   grpc.Dial(addr,                                                          │
│       grpc.WithDefaultCallOptions(                                         │
│           grpc.MaxCallRecvMsgSize(50 * 1024 * 1024),                       │
│           grpc.MaxCallSendMsgSize(50 * 1024 * 1024),                       │
│       ),                                                                   │
│   )                                                                         │
│                                                                             │
│ SOLUTION 2: Streaming (better for large responses)                         │
│                                                                             │
│   // Proto                                                                  │
│   rpc ComputeScores(ScoreRequest) returns (stream ScoreChunk);             │
│                                                                             │
│   // Server sends one ciphertext at a time                                 │
│   for _, ct := range ciphertexts {                                         │
│       stream.Send(&ScoreChunk{EncryptedScore: ct})                         │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9. Error Handling in Crypto Operations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: Silent failures in crypto = garbage results (not errors!)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PROBLEM:                                                                    │
│   Many Lattigo operations return results without errors.                   │
│   If inputs are wrong, you get garbage output, not an error.               │
│                                                                             │
│ EXAMPLES OF SILENT FAILURES:                                                │
│   • Mismatched parameters between client/server                            │
│   • Wrong ciphertext level after operations                                │
│   • Overflow in fixed-point encoding                                       │
│   • Corrupted ciphertext from bad deserialization                          │
│                                                                             │
│ DEFENSIVE MEASURES:                                                         │
│                                                                             │
│   // 1. Validate parameters match                                          │
│   func validateParams(clientParams, serverParams rlwe.Parameters) error {  │
│       if clientParams.LogN() != serverParams.LogN() {                      │
│           return errors.New("parameter mismatch: LogN")                    │
│       }                                                                    │
│       // ... more checks                                                   │
│   }                                                                         │
│                                                                             │
│   // 2. Check ciphertext level                                             │
│   if ct.Level() < minRequiredLevel {                                       │
│       return errors.New("insufficient noise budget")                       │
│   }                                                                         │
│                                                                             │
│   // 3. Sanity check decrypted values                                      │
│   if decoded < -2.0 || decoded > 2.0 {                                     │
│       log.Warn("decoded value outside expected range", "value", decoded)   │
│   }                                                                         │
│                                                                             │
│   // 4. Use checksums for serialized ciphertexts                           │
│   type CiphertextWithChecksum struct {                                     │
│       Data     []byte                                                      │
│       Checksum uint32 // CRC32                                             │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10. Testing Homomorphic Operations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOTCHA: HE results are approximate; exact equality tests will fail          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ BAD TEST:                                                                   │
│                                                                             │
│   expected := 0.95432                                                      │
│   assert.Equal(t, expected, decrypted) // FAILS!                           │
│                                                                             │
│ GOOD TEST:                                                                  │
│                                                                             │
│   expected := 0.95432                                                      │
│   tolerance := 1e-6  // Depends on your precision                          │
│   assert.InDelta(t, expected, decrypted, tolerance)                        │
│                                                                             │
│ COMPREHENSIVE TEST:                                                         │
│                                                                             │
│   func TestHomomorphicDotProduct(t *testing.T) {                           │
│       // Test with known values                                            │
│       query := []float64{0.1, 0.2, 0.3, 0.4}                               │
│       vector := []float64{0.4, 0.3, 0.2, 0.1}                              │
│       expected := 0.1*0.4 + 0.2*0.3 + 0.3*0.2 + 0.4*0.1 // = 0.20         │
│                                                                             │
│       encrypted := client.Encrypt(query)                                   │
│       result := server.HomomorphicDotProduct(encrypted, vector)            │
│       decrypted := client.Decrypt(result)                                  │
│                                                                             │
│       assert.InDelta(t, expected, decrypted, 1e-6)                         │
│                                                                             │
│       // Test with random values                                           │
│       for i := 0; i < 100; i++ {                                           │
│           q := randomVector(128)                                           │
│           v := randomVector(128)                                           │
│           expected := plainDotProduct(q, v)                                │
│           // ... encrypt, compute, decrypt ...                             │
│           assert.InDelta(t, expected, decrypted, 1e-4)                     │
│       }                                                                     │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Targets

### Latency Breakdown

| Stage | Python (Current) | Go (Target) | Notes |
|-------|------------------|-------------|-------|
| Key Generation | 250ms | 50ms | One-time per session |
| LSH Hash | <1ms | <0.1ms | SIMD optimized |
| LSH Search | <1ms | <0.1ms | In-memory |
| Encryption (32D) | 1500ms | **50-100ms** | Lattigo is ~30x faster |
| Server Dot Product (×20) | 450ms | **10-20ms** | Parallel + optimized |
| Decryption (×20) | 2600ms | **100-200ms** | Parallel goroutines |
| Network (round-trips) | ~50ms | ~50ms | 2 round-trips |
| **Total** | **~5000ms** | **<300ms** | **16x improvement** |

### Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| QPS (single node) | 50-100 | CPU-bound on PHE |
| QPS (3-node cluster) | 150-300 | Linear scaling |
| Concurrent sessions | 10,000+ | Limited by Redis |
| Max vector dimension | 512 | Higher = slower |
| Max candidates | 1000 | Limited by memory |

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Search Service | 4 cores | 4 GB | - |
| Milvus | 8 cores | 16 GB | 100 GB SSD |
| Redis | 2 cores | 4 GB | 10 GB |
| Load Balancer | 1 core | 512 MB | - |

---

## Security Considerations

### Threat Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THREAT MODEL                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ TRUSTED:                                                                    │
│   • Client device and SDK                                                  │
│   • Client's secret key storage                                            │
│                                                                             │
│ UNTRUSTED (Honest-but-Curious):                                            │
│   • Server infrastructure                                                  │
│   • Network between client and server                                      │
│   • Database storage                                                       │
│                                                                             │
│ WHAT SERVER CAN LEARN:                                                      │
│   • That a search happened (timing)                                        │
│   • Approximate query location (LSH bucket)                                │
│   • Which candidates were scored                                           │
│   • Number of results requested                                            │
│                                                                             │
│ WHAT SERVER CANNOT LEARN:                                                   │
│   • Actual query vector values                                             │
│   • Actual similarity scores                                               │
│   • Which result was "best" (ranking)                                      │
│   • Query semantics/meaning                                                │
│                                                                             │
│ ATTACKS WE DEFEND AGAINST:                                                  │
│   ✓ Passive eavesdropping                                                  │
│   ✓ Server-side data breach                                                │
│   ✓ Malicious server returning wrong results                               │
│   ✓ Traffic analysis (with padding)                                        │
│                                                                             │
│ ATTACKS WE DON'T DEFEND AGAINST:                                           │
│   ✗ Compromised client device                                              │
│   ✗ Side-channel attacks on client                                         │
│   ✗ Denial of service                                                      │
│   ✗ Malicious server returning no results                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Security Checklist

```
[ ] Cryptographic
    [ ] 128-bit security level (Lattigo PN14QP438)
    [ ] No secret key on server
    [ ] Secure random number generation
    [ ] Key rotation mechanism
    [ ] Constant-time operations where possible

[ ] Transport
    [ ] TLS 1.3 for client connections
    [ ] mTLS between internal services
    [ ] Certificate pinning in SDK
    [ ] No sensitive data in URLs

[ ] Application
    [ ] Input validation (vector dimensions, IDs)
    [ ] Rate limiting per client
    [ ] Request size limits
    [ ] Timeout enforcement
    [ ] Audit logging (without query content!)

[ ] Infrastructure
    [ ] Network segmentation
    [ ] Secrets in Vault (not env vars)
    [ ] No persistent logs of ciphertexts
    [ ] Regular security scanning
```

---

## Deployment Guide

### Docker Setup

```dockerfile
# Dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /opaque-search ./cmd/search-service

FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /opaque-search /opaque-search
EXPOSE 50051 8080
ENTRYPOINT ["/opaque-search"]
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opaque-search
  labels:
    app: opaque-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opaque-search
  template:
    metadata:
      labels:
        app: opaque-search
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      containers:
      - name: search
        image: yourorg/opaque-search:v1.0.0
        ports:
        - containerPort: 50051
          name: grpc
        - containerPort: 8080
          name: metrics
        env:
        - name: MILVUS_ADDR
          value: "milvus:19530"
        - name: REDIS_ADDR
          value: "redis:6379"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            cpu: "2"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: opaque-search
spec:
  selector:
    app: opaque-search
  ports:
  - port: 50051
    targetPort: 50051
    name: grpc
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opaque-search-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opaque-search
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Helm Chart Structure

```
opaque-helm/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── servicemonitor.yaml
└── charts/
    ├── milvus/
    └── redis/
```

---

## Testing Strategy

### Test Pyramid

```
                    ┌─────────┐
                    │   E2E   │  ← 10% (Slow, expensive)
                   ─┴─────────┴─
                 ┌───────────────┐
                 │  Integration  │  ← 30% (Service boundaries)
                ─┴───────────────┴─
              ┌───────────────────────┐
              │        Unit           │  ← 60% (Fast, isolated)
             ─┴───────────────────────┴─
```

### Test Categories

```go
// Unit: Crypto operations
func TestEncryptDecryptRoundtrip(t *testing.T)
func TestHomomorphicDotProductAccuracy(t *testing.T)
func TestLSHHashConsistency(t *testing.T)

// Integration: Service interactions
func TestSearchServiceWithMilvus(t *testing.T)
func TestSessionManagerWithRedis(t *testing.T)
func TestGRPCClientServer(t *testing.T)

// E2E: Full flow
func TestPrivateSearchE2E(t *testing.T)
func TestSearchAccuracyVsPlaintext(t *testing.T)

// Benchmarks
func BenchmarkEncryption(b *testing.B)
func BenchmarkHomomorphicDotProduct(b *testing.B)
func BenchmarkDecryptionParallel(b *testing.B)
func BenchmarkLSHSearch(b *testing.B)
```

---

## Monitoring & Observability

### Key Metrics

```
# Latency (histograms)
opaque_search_latency_seconds{stage="lsh"}
opaque_search_latency_seconds{stage="encryption"}
opaque_search_latency_seconds{stage="phe_compute"}
opaque_search_latency_seconds{stage="decryption"}
opaque_search_latency_seconds{stage="total"}

# Throughput (counters)
opaque_searches_total
opaque_candidates_processed_total

# Errors (counters)
opaque_errors_total{type="crypto"}
opaque_errors_total{type="storage"}
opaque_errors_total{type="timeout"}

# Resources (gauges)
opaque_active_sessions
opaque_goroutines
opaque_memory_bytes
```

### Grafana Dashboard Panels

1. Request rate and latency (p50, p95, p99)
2. Error rate by type
3. Active sessions over time
4. CPU and memory utilization
5. Milvus query latency
6. Redis hit rate

---

## Cost Estimation

### Cloud Resources (AWS)

| Resource | Spec | Monthly Cost |
|----------|------|--------------|
| EKS Cluster | 3 × m5.xlarge | ~$400 |
| Milvus (managed) | 8 vCPU, 32GB | ~$300 |
| ElastiCache Redis | cache.m5.large | ~$150 |
| ALB | - | ~$50 |
| Data transfer | 100 GB | ~$10 |
| **Total** | | **~$900/month** |

### At Scale (1M searches/month)

| Resource | Spec | Monthly Cost |
|----------|------|--------------|
| EKS Cluster | 10 × m5.2xlarge | ~$2,500 |
| Milvus | 16 vCPU, 64GB, GPU | ~$1,000 |
| ElastiCache | cache.m5.xlarge cluster | ~$500 |
| **Total** | | **~$4,000/month** |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Lattigo bugs | Low | High | Extensive testing, stay updated |
| Performance misses target | Medium | Medium | Early benchmarking, fallback to more resources |
| Milvus scaling issues | Medium | Medium | Load testing, consider Qdrant alternative |
| Security vulnerability | Low | Critical | Security audit, bug bounty |
| Key management breach | Low | Critical | HSM for production, strict access control |

---

## Appendix: Reference Implementation

Full working code available at: `github.com/yourorg/opaque/go` (to be created)

### Quick Start

```bash
# Clone
git clone https://github.com/yourorg/opaque/go
cd opaque/go

# Run tests
go test ./...

# Run benchmarks
go test -bench=. ./pkg/crypto/

# Start server
docker-compose up -d
go run ./cmd/search-service

# Run client example
go run ./examples/search/main.go
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: Project Opaque Team*
