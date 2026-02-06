# Opaque Tier 3 Implementation Plan

## Executive Summary

This plan outlines the steps to perfect the Tier 3 (Client-Server) architecture for Opaque, focusing on the three priorities: **Privacy**, **Speed**, and **Accuracy**.

**Current State (from test results):**
- Accuracy: 10-12% Recall@10 (Target: >80%)
- Speed: ~1.3s per query (Target: <2s) ✅
- Privacy: 100% working (HE matches plaintext perfectly)

**After Implementation:**
- Accuracy: >80% Recall@10
- Speed: <2s per query
- Privacy: Maintained with full client-server separation

---

## Key Architectural Decisions

### 1. Remove Sub-Buckets
**Decision:** Fetch ALL vectors from selected super-buckets instead of sub-bucket filtering.

**Rationale:**
- Test `TestSIFTSubBucketDistance` proved sub-bucket IDs have NO correlation with vector similarity
- Current `getNeighborSubBuckets()` fetches by ID increment which is useless
- Removing sub-buckets simplifies architecture and fixes accuracy

**Impact:**
- Accuracy: 12% → 80%+ Recall@10
- Data scanned: ~5% → ~25% of dataset (acceptable trade-off)
- Local scoring is fast (~0.01ms/vector), so scanning more is fine

### 2. K-Means for Super-Buckets
**Decision:** Use K-means clustering exclusively (not LSH) for super-bucket assignment.

**Rationale:**
- Test shows K-means: 98% GT hit rate vs LSH: 83%
- K-means centroids are optimal by construction (actual cluster centers)
- Already implemented in `pkg/cluster/kmeans.go` and `pkg/hierarchical/kmeans_builder.go`

### 3. Centroid Trade-offs

| Centroids | HE Time (~33ms each) | Vectors/Cluster (10K) | Top-4 Scan % |
|-----------|---------------------|----------------------|--------------|
| 64        | ~2.1s               | ~156                 | 6.2%         |
| 32        | ~1.0s               | ~312                 | 12.5%        |
| 16        | ~0.5s               | ~625                 | 25%          |

**Recommendation:** Start with 32 centroids (balanced). Can tune later.

**Why fewer centroids is usually better:**
- HE dot product: ~33ms
- Local AES decrypt + score: ~0.01ms per vector
- HE is ~3300x slower than local scoring
- Reducing 1 centroid saves 33ms, costs ~312 local ops (3ms)
- Net win until clusters become extremely large

### 4. Encrypted Centroid Caching
**Decision:** Cache encrypted centroids with staleness tracking.

**Implementation:**
```go
type CentroidCache struct {
    encrypted map[int]*rlwe.Ciphertext
    stale     map[int]bool  // Marked stale when vectors added/removed
    mu        sync.RWMutex
}
```

**Invalidation Rules:**
- Add vector → Mark affected cluster's centroid as stale
- Remove vector → Mark affected cluster's centroid as stale
- Batch recompute stale centroids periodically

### 5. Client-Server Architecture
**Decision:** REST API for MVP (simpler to implement and debug on laptop).

**Why REST over gRPC for now:**
- Easier to test with curl/Postman
- Simpler error handling and debugging
- JSON is human-readable for development
- Can add gRPC later for production efficiency

**Future:** Add gRPC with protobuf for binary efficiency (HE ciphertexts are large).

### 6. No Cloud Dependencies
**Decision:** Everything runs locally on laptop for now.

- No AWS, no Nitro Enclaves, no cloud KMS
- Local file storage for blobs
- In-memory auth service
- Can add cloud services later

---

## Implementation Phases

### Phase 1: Fix Accuracy (Remove Sub-Buckets)
**Goal:** Achieve >80% Recall@10

**Files to modify:**
- `go/pkg/client/enterprise_hierarchical.go`
  - Remove `getNeighborSubBuckets()` logic
  - Add `getAllVectorsFromSuper()` method
  - Update `Search()` to fetch all vectors from selected supers

- `go/pkg/hierarchical/kmeans_builder.go`
  - Remove sub-bucket assignment during build
  - Store vectors by super-bucket only (key: `"super_XX"`)

- `go/pkg/blob/store.go`
  - Add `GetBySuperBucket(ctx, superID int)` method

**Tests to add:**
- `TestNoSubBucketAccuracy` - Verify >80% Recall@10
- `TestNoSubBucketPerformance` - Verify <2s query time

**Commit:** `feat: remove sub-buckets for improved accuracy`

---

### Phase 2: K-Means Only Mode
**Goal:** Use K-means exclusively, remove LSH for super-buckets

**Files to modify:**
- `go/pkg/hierarchical/config.go`
  - Remove LSH-related config for super-buckets
  - Keep LSH config only for legacy compatibility

- `go/pkg/client/enterprise_hierarchical.go`
  - Remove LSH hasher for super-bucket selection
  - Use K-means centroid scoring only

- `go/pkg/hierarchical/builder.go` (if needed)
  - Deprecate or remove LSH-based builder

**Tests:**
- Update all accuracy tests to use K-means builder
- Add migration test for existing LSH-based indexes

**Commit:** `refactor: use k-means exclusively for super-bucket clustering`

---

### Phase 3: Centroid Caching
**Goal:** Cache encrypted centroids, track staleness

**Files to create:**
- `go/pkg/cache/centroid_cache.go`
  ```go
  type CentroidCache struct {
      encrypted map[int]*rlwe.Ciphertext
      stale     map[int]bool
      heEngine  *crypto.Engine
      mu        sync.RWMutex
  }

  func (c *CentroidCache) Get(centroidID int) (*rlwe.Ciphertext, error)
  func (c *CentroidCache) MarkStale(centroidID int)
  func (c *CentroidCache) RefreshStale(centroids [][]float64) error
  ```

**Files to modify:**
- `go/pkg/client/enterprise_hierarchical.go`
  - Use cache for HE centroid operations
  - Pass cache to Search() or store in client

**Tests:**
- `TestCentroidCacheHit` - Verify cache returns same ciphertext
- `TestCentroidCacheStaleness` - Verify staleness tracking
- `TestCentroidCacheRefresh` - Verify stale centroids refresh

**Commit:** `feat: add encrypted centroid caching with staleness tracking`

---

### Phase 4: Batch HE Operations
**Goal:** Reduce HE overhead by batching

**Approach:** CKKS supports SIMD - pack multiple centroids per ciphertext

**Files to modify:**
- `go/pkg/crypto/crypto.go`
  - Add `BatchHomomorphicDotProduct(query, centroids [][]float64)`
  - Pack multiple centroids into slots
  - Single HE multiply + rotate to get all scores

**Expected improvement:**
- Current: 32 centroids × 33ms = ~1s
- Batched: 1-4 HE operations = ~100-150ms

**Tests:**
- `TestBatchHEAccuracy` - Verify batch results match individual
- `TestBatchHEPerformance` - Verify speedup

**Commit:** `perf: batch HE operations for centroid scoring`

---

### Phase 5: Verify Parallelism
**Goal:** Ensure parallel HE execution is optimal

**Current state:** 4 workers in `enterprise_hierarchical.go:139`

**Tasks:**
- Profile with `go tool pprof` to identify bottlenecks
- Test different worker counts (2, 4, 8, 16)
- Verify no lock contention in crypto package
- Consider using worker pool pattern

**Files to review:**
- `go/pkg/client/enterprise_hierarchical.go` (worker pool)
- `go/pkg/crypto/crypto.go` (thread safety)

**Commit:** `perf: optimize parallel HE execution`

---

### Phase 6: Client-Server Split (REST API)
**Goal:** Separate client and server into distinct processes

#### Server Component
**Create:** `go/cmd/server/main.go`

**Endpoints:**
```
POST /api/v1/score-centroids
  Request:  { encrypted_query: base64, enterprise_id: string }
  Response: { encrypted_scores: [base64...], timing_ms: int }

GET /api/v1/buckets/:enterprise_id/:super_id
  Response: { blobs: [{ id, ciphertext: base64 }...] }

POST /api/v1/auth/token
  Request:  { user_id: string, enterprise_id: string }
  Response: { token: string, aes_key: base64, centroids: [[float]...], expires_at: timestamp }

POST /api/v1/auth/refresh
  Request:  { token: string }
  Response: { token: string, expires_at: timestamp }
```

**Create:** `go/internal/server/`
- `server.go` - HTTP server setup
- `handlers.go` - Request handlers
- `middleware.go` - Auth middleware

#### Client Component
**Create:** `go/pkg/client/remote_client.go`

```go
type RemoteClient struct {
    serverURL   string
    httpClient  *http.Client
    credentials *auth.ClientCredentials
    heEngine    *crypto.Engine
    encryptor   *encrypt.AESGCM
}

func (c *RemoteClient) Search(ctx context.Context, query []float64, topK int) (*SearchResult, error)
```

#### Local Development Setup
**Create:** `go/cmd/devserver/main.go`
- Runs server locally on `localhost:8080`
- Uses file-based blob storage
- In-memory auth service

**Create:** `docker-compose.yml` (optional)
```yaml
services:
  server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
```

**Tests:**
- `TestRemoteClientSearch` - End-to-end with local server
- `TestServerScoreCentroids` - Unit test endpoint
- `TestServerAuth` - Token issuance and refresh

**Commits:**
- `feat: add REST API server for HE centroid scoring`
- `feat: add remote client for client-server search`
- `feat: add local development server`

---

### Phase 7: Integration & Benchmarks
**Goal:** Verify everything works together at scale

**Benchmarks to add:**
- 10K vectors, 32 centroids, 100 queries
- 100K vectors, 32 centroids, 100 queries
- Measure: Recall@10, Query latency (p50, p95, p99), Throughput

**Files to create:**
- `go/test/integration_test.go`
- `go/test/benchmark_100k_test.go`

**Success Criteria:**
| Metric | Target |
|--------|--------|
| Recall@10 | >80% |
| Query latency (p95) | <2s |
| Privacy | Server sees no plaintext |

**Commit:** `test: add integration tests and 100K benchmarks`

---

## File Structure After Implementation

```
go/
├── cmd/
│   ├── server/           # Production server
│   │   └── main.go
│   ├── devserver/        # Local development server
│   │   └── main.go
│   └── cli/              # CLI tool
│       └── main.go
├── internal/
│   └── server/
│       ├── server.go
│       ├── handlers.go
│       └── middleware.go
├── pkg/
│   ├── client/
│   │   ├── remote_client.go      # NEW: Remote HTTP client
│   │   ├── enterprise_hierarchical.go  # Updated
│   │   └── ...
│   ├── cache/
│   │   └── centroid_cache.go     # NEW
│   ├── cluster/
│   │   └── kmeans.go             # Existing
│   ├── hierarchical/
│   │   ├── kmeans_builder.go     # Updated
│   │   └── ...
│   └── ...
└── test/
    ├── integration_test.go       # NEW
    └── benchmark_100k_test.go    # NEW
```

---

## Commit Strategy

Each phase should produce 1-3 atomic commits:

1. **Phase 1:** `feat: remove sub-buckets for improved accuracy`
2. **Phase 2:** `refactor: use k-means exclusively for super-bucket clustering`
3. **Phase 3:** `feat: add encrypted centroid caching with staleness tracking`
4. **Phase 4:** `perf: batch HE operations for centroid scoring`
5. **Phase 5:** `perf: optimize parallel HE execution`
6. **Phase 6:**
   - `feat: add REST API server for HE centroid scoring`
   - `feat: add remote client for client-server search`
   - `feat: add local development server`
7. **Phase 7:** `test: add integration tests and 100K benchmarks`

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Batched HE accuracy differs | Add comparison tests against individual ops |
| REST too slow for large payloads | Use streaming, add gRPC later |
| Centroid cache memory pressure | Add LRU eviction, configurable size |
| 100K benchmark OOM | Use streaming blob fetch, limit concurrent decryptions |

---

## Success Metrics

| Metric | Current | Target | Phase |
|--------|---------|--------|-------|
| Recall@10 | 12% | >80% | 1-2 |
| Query latency | 1.3s | <2s | ✅ already |
| HE correctness | 100% | 100% | Maintain |
| Client-server split | No | Yes | 6 |
| Local dev setup | No | Yes | 6 |

---

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 (Remove Sub-Buckets)
3. Run tests after each phase to verify improvements
4. Commit after each phase with descriptive messages

Ready to execute?
