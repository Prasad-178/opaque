package client

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"runtime"
	"container/heap"
	"strings"
	"sync"
	"time"

	"github.com/Prasad-178/opaque/pkg/auth"
	"github.com/Prasad-178/opaque/pkg/blob"
	"github.com/Prasad-178/opaque/pkg/cache"
	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/encrypt"
	"github.com/Prasad-178/opaque/pkg/hierarchical"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

// EnterpriseHierarchicalClient performs privacy-preserving vector search
// using enterprise-specific credentials received from the auth service.
//
// Architecture (no sub-buckets - simplified for better accuracy):
//   - Level 1: HE scoring on centroids (k-means clusters)
//   - Level 2: Fetch ALL vectors from selected super-buckets + decoys
//   - Level 3: Local AES decrypt + scoring
//
// Key security properties:
//   - Uses enterprise-specific AES key for decryption
//   - HE scoring on centroids (server never sees query or selection)
//   - Decoy buckets hide real bucket access patterns
//
// The HE backend is pluggable via [crypto.HEProvider]:
//   - DirectHEProvider: single-key mode (default)
//   - ThresholdHEProvider: t-of-N threshold CKKS mode
type EnterpriseHierarchicalClient struct {
	config      hierarchical.Config
	credentials *auth.ClientCredentials

	// Derived from credentials
	encryptor *encrypt.AESGCM
	heProvider crypto.HEProvider // Pluggable HE backend (direct or threshold)

	// Centroid cache for faster HE operations
	// Pre-encodes centroids as HE plaintexts
	centroidCache *cache.CentroidCache

	// Batch centroid cache for SIMD operations
	// Packs multiple centroids per plaintext for massive speedup
	batchCentroidCache *cache.BatchCentroidCache

	// Storage backend
	store blob.Store

	mu sync.RWMutex
}

// DefaultPoolSize is the default number of HE engines in the worker pool.
// Each engine has independent evaluators but shared keys, enabling true parallelism.
const DefaultPoolSize = 4

// NewEnterpriseHierarchicalClient creates a client from authenticated credentials
// using the default HE engine pool size of 4 and single-key (direct) HE mode.
//
// The credentials contain:
//   - AES key for vector decryption
//   - Centroids for HE scoring
//
// For threshold CKKS or custom pool size, use [NewEnterpriseHierarchicalClientWithProvider].
func NewEnterpriseHierarchicalClient(
	cfg hierarchical.Config,
	credentials *auth.ClientCredentials,
	store blob.Store,
) (*EnterpriseHierarchicalClient, error) {
	return NewEnterpriseHierarchicalClientWithPoolSize(cfg, credentials, store, DefaultPoolSize)
}

// NewEnterpriseHierarchicalClientWithPoolSize creates a client with single-key HE mode
// and a configurable number of HE engines in the worker pool.
//
// Each HE engine consumes significant memory (~50MB for CKKS parameters with LogN=14)
// but enables parallel homomorphic dot product computations during search.
//
// Recommended: poolSize = runtime.NumCPU(), capped at 8.
// Minimum: 1. Values < 1 are clamped to 1.
func NewEnterpriseHierarchicalClientWithPoolSize(
	cfg hierarchical.Config,
	credentials *auth.ClientCredentials,
	store blob.Store,
	poolSize int,
) (*EnterpriseHierarchicalClient, error) {
	if poolSize < 1 {
		poolSize = 1
	}
	provider, err := crypto.NewDirectHEProvider(poolSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create HE provider: %w", err)
	}
	return NewEnterpriseHierarchicalClientWithProvider(cfg, credentials, store, provider)
}

// NewEnterpriseHierarchicalClientWithProvider creates a client with a custom HE provider.
// Use this to enable threshold CKKS mode or any other HEProvider implementation.
//
// Example (threshold mode):
//
//	committee, _ := threshold.NewCommittee(5, 3)
//	provider, _ := crypto.NewThresholdHEProvider(committee, 4)
//	client, _ := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, provider)
func NewEnterpriseHierarchicalClientWithProvider(
	cfg hierarchical.Config,
	credentials *auth.ClientCredentials,
	store blob.Store,
	provider crypto.HEProvider,
) (*EnterpriseHierarchicalClient, error) {
	if credentials == nil {
		return nil, fmt.Errorf("credentials are required")
	}
	if store == nil {
		return nil, fmt.Errorf("store is required")
	}
	if provider == nil {
		return nil, fmt.Errorf("HE provider is required")
	}

	// Create encryptor from credentials
	encryptor, err := encrypt.NewAESGCM(credentials.AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create encryptor: %w", err)
	}

	// Override config from credentials if not set
	if cfg.Dimension <= 0 {
		cfg.Dimension = credentials.Dimension
	}
	if cfg.NumSuperBuckets <= 0 {
		cfg.NumSuperBuckets = credentials.NumSuperBuckets
	}

	// Create centroid cache and pre-load centroids
	centroidCache := cache.NewCentroidCache(provider.GetParams(), provider.GetEncoder())
	if len(credentials.Centroids) > 0 {
		if err := centroidCache.LoadCentroids(credentials.Centroids, provider.GetParams().MaxLevel()); err != nil {
			return nil, fmt.Errorf("failed to load centroids into cache: %w", err)
		}
	}

	// Create batch centroid cache for SIMD operations
	batchCentroidCache := cache.NewBatchCentroidCache(provider.GetParams(), provider.GetEncoder(), cfg.Dimension)
	if len(credentials.Centroids) > 0 {
		if err := batchCentroidCache.LoadCentroids(credentials.Centroids, provider.GetParams().MaxLevel()); err != nil {
			return nil, fmt.Errorf("failed to load batch centroids: %w", err)
		}
	}

	return &EnterpriseHierarchicalClient{
		config:             cfg,
		credentials:        credentials,
		encryptor:          encryptor,
		heProvider:         provider,
		centroidCache:      centroidCache,
		batchCentroidCache: batchCentroidCache,
		store:              store,
	}, nil
}

// Search performs the full hierarchical private search.
// Returns search results with timing breakdown and statistics.
func (c *EnterpriseHierarchicalClient) Search(ctx context.Context, query []float64, topK int) (*hierarchical.SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	startTotal := time.Now()
	result := &hierarchical.SearchResult{
		Stats: hierarchical.SearchStats{},
	}

	if len(query) != c.config.Dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.config.Dimension)
	}

	// Check token expiry
	if c.credentials.IsExpired() {
		return nil, fmt.Errorf("credentials have expired, please refresh token")
	}

	// Normalize query
	normalizedQuery := normalizeVectorCopy(query)

	// ==========================================
	// LEVEL 1: HE Centroid Scoring
	// Server computes on encrypted query, never sees results
	// ==========================================

	// Step 1a: Encrypt query with HE
	startEncrypt := time.Now()
	encQuery, err := c.heProvider.EncryptVector(normalizedQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt query: %w", err)
	}
	result.Timing.HEEncryptQuery = time.Since(startEncrypt)

	// Step 1b: Compute HE scores for ALL centroids (parallel using engine pool)
	// Each engine in the pool has its own evaluator, enabling true parallelism
	startHE := time.Now()
	centroids := c.credentials.Centroids
	encScores := make([]*rlwe.Ciphertext, len(centroids))
	scoreErrs := make([]error, len(centroids))

	// Get cached plaintexts (pre-encoded centroids)
	cachedPlaintexts := c.centroidCache.GetAll(len(centroids))
	useCached := cachedPlaintexts != nil && len(cachedPlaintexts) == len(centroids) && cachedPlaintexts[0] != nil

	var wg sync.WaitGroup
	// Use pool size as the number of workers (each has its own engine)
	numWorkers := c.heProvider.Size()
	if numWorkers > len(centroids) {
		numWorkers = len(centroids)
	}
	workChan := make(chan int, len(centroids))

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Acquire an engine from the pool for this worker
			engine := c.heProvider.Acquire()
			defer c.heProvider.Release(engine)

			for i := range workChan {
				if useCached {
					// Use cached pre-encoded plaintext (faster)
					encScores[i], scoreErrs[i] = engine.HomomorphicDotProductCached(encQuery, cachedPlaintexts[i])
				} else {
					// Fallback to encoding on-the-fly
					encScores[i], scoreErrs[i] = engine.HomomorphicDotProduct(encQuery, centroids[i])
				}
			}
		}()
	}

	for i := range centroids {
		workChan <- i
	}
	close(workChan)
	wg.Wait()

	for i, err := range scoreErrs {
		if err != nil {
			return nil, fmt.Errorf("failed to compute HE score for centroid %d: %w", i, err)
		}
	}
	result.Timing.HECentroidScores = time.Since(startHE)
	result.Stats.HEOperations = len(centroids)

	// Step 1c: Decrypt scores privately (only client sees this!)
	// CKKS returns actual dot products directly - no bias correction needed!
	// With engine pool, we can also parallelize decryption
	startDecrypt := time.Now()
	scores := make([]float64, len(encScores))
	decryptErrs := make([]error, len(encScores))

	// Parallel decryption using pool
	decryptChan := make(chan int, len(encScores))
	var decryptWg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		decryptWg.Add(1)
		go func() {
			defer decryptWg.Done()
			engine := c.heProvider.Acquire()
			defer c.heProvider.Release(engine)

			for i := range decryptChan {
				scores[i], decryptErrs[i] = engine.DecryptScalar(encScores[i])
			}
		}()
	}

	for i := range encScores {
		decryptChan <- i
	}
	close(decryptChan)
	decryptWg.Wait()

	for i, err := range decryptErrs {
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt score %d: %w", i, err)
		}
	}
	result.Timing.HEDecryptScores = time.Since(startDecrypt)

	// Step 1d: Select top super-buckets (SERVER NEVER SEES THIS!)
	// Use multi-probe selection to capture clusters within threshold of top-K
	// This addresses HE precision noise that can cause near-miss exclusions
	topSupers := selectClusters(
		scores,
		c.config.TopSuperBuckets,
		c.config.ProbeThreshold,
		c.config.MaxProbeClusters,
		c.config.ProbeStrategy,
		c.config.GapMultiplier,
	)
	result.Stats.SuperBucketsSelected = len(topSupers)
	result.Stats.SelectedClusters = topSupers // Track for diagnostics

	// ==========================================
	// LEVEL 2: Decoy-Based Bucket Fetch
	// Server can't distinguish real from decoy buckets
	// No sub-buckets - fetch ALL vectors from selected super-buckets
	// ==========================================

	startSelection := time.Now()

	// Generate decoy super-buckets from non-selected super-buckets
	decoySupers := c.generateDecoySupers(topSupers, c.config.NumDecoys)

	// Combine real + decoy super-buckets and shuffle
	allSupers := append(append([]int{}, topSupers...), decoySupers...)
	shuffleInts(allSupers)

	result.Stats.RealSubBuckets = len(topSupers)
	result.Stats.DecoySubBuckets = len(decoySupers)
	result.Stats.TotalSubBuckets = len(allSupers)
	result.Timing.BucketSelection = time.Since(startSelection)

	// Fetch ALL vectors from selected super-buckets (server can't tell which are real)
	startFetch := time.Now()
	blobs, err := c.store.GetSuperBuckets(ctx, allSupers)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch buckets: %w", err)
	}
	result.Timing.BucketFetch = time.Since(startFetch)
	result.Stats.BlobsFetched = len(blobs)

	// ==========================================
	// LEVEL 3: Local AES Decrypt + Scoring
	// All computation is client-side, server sees nothing
	// ==========================================

	// Filter to real clusters only — skip decoy blobs to avoid wasted decrypt/scoring.
	// Privacy is preserved: the server already saw all bucket requests (real + decoy, shuffled).
	realBlobs := filterRealBlobs(blobs, topSupers)

	// Decrypt real vectors in parallel
	startAES := time.Now()
	decrypted, decryptStats := parallelDecryptBlobs(realBlobs, c.encryptor)
	result.Timing.AESDecrypt = time.Since(startAES)
	result.Stats.DecryptSucceeded = decryptStats.Succeeded
	result.Stats.DecryptFailed = decryptStats.Failed

	// Score locally with deduplication for redundant assignment
	// Keep only the highest score for each unique original ID
	startScore := time.Now()

	// Map to track best score per original ID
	scoreMap := make(map[string]float64, len(decrypted))

	for _, d := range decrypted {
		vec := d.vector
		if !c.config.NormalizedStorage {
			vec = normalizeVectorCopy(d.vector)
		}
		score := dotProductVec(normalizedQuery, vec)

		// Keep highest score for each original ID
		if existing, ok := scoreMap[d.id]; !ok || score > existing {
			scoreMap[d.id] = score
		}
	}

	// Select top-K using min-heap: O(n log K) instead of O(n log n) full sort.
	result.Results = topKFromMap(scoreMap, topK)

	result.Timing.LocalScoring = time.Since(startScore)
	result.Stats.VectorsScored = len(scoreMap) // Unique vectors scored

	result.Timing.Total = time.Since(startTotal)
	return result, nil
}

// SearchBatch performs hierarchical search using SIMD batch HE operations.
// This is significantly faster than Search() as it reduces 64 HE ops to just 1-2.
//
// For 64 centroids with 128-dim vectors:
//   - Standard: 64 HE multiply+sum operations (~2s)
//   - Batch: 1 HE multiply + partial sum (~100ms)
//
// The results are mathematically equivalent to Search().
func (c *EnterpriseHierarchicalClient) SearchBatch(ctx context.Context, query []float64, topK int) (*hierarchical.SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	startTotal := time.Now()
	result := &hierarchical.SearchResult{
		Stats: hierarchical.SearchStats{},
	}

	if len(query) != c.config.Dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.config.Dimension)
	}

	if c.credentials.IsExpired() {
		return nil, fmt.Errorf("credentials have expired, please refresh token")
	}

	// Normalize query
	normalizedQuery := normalizeVectorCopy(query)

	// ==========================================
	// LEVEL 1: SIMD Batch HE Centroid Scoring
	// Uses packed centroids for massive speedup
	// ==========================================

	// Pack query for batch operation (replicate across slots)
	startEncrypt := time.Now()
	packedQuery := c.batchCentroidCache.PackQuery(normalizedQuery)
	encPackedQuery, err := c.heProvider.EncryptVector(packedQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt packed query: %w", err)
	}
	result.Timing.HEEncryptQuery = time.Since(startEncrypt)

	// Compute all centroid scores with batch operation(s)
	// When dimension is large (e.g. 960), multiple packs are needed and processed in parallel.
	startHE := time.Now()
	numCentroids := c.batchCentroidCache.GetNumCentroids()
	dimension := c.batchCentroidCache.GetDimension()
	packedPlaintexts := c.batchCentroidCache.GetPackedPlaintexts()
	centroidsPerPack := c.batchCentroidCache.GetCentroidsPerPack()

	// Each packed plaintext can score up to centroidsPerPack centroids.
	// Process packs in parallel when there are multiple (common with high-dim vectors).
	packScores := make([][]float64, len(packedPlaintexts))
	packErrors := make([]error, len(packedPlaintexts))

	if len(packedPlaintexts) == 1 {
		// Single pack: no goroutine overhead
		encResult, err := c.heProvider.HomomorphicBatchDotProduct(
			encPackedQuery, packedPlaintexts[0], centroidsPerPack, dimension,
		)
		if err != nil {
			return nil, fmt.Errorf("failed batch dot product: %w", err)
		}
		packScores[0], packErrors[0] = c.heProvider.DecryptBatchScalars(
			encResult, centroidsPerPack, dimension,
		)
	} else {
		// Multiple packs: process in parallel using the HE engine pool.
		// encPackedQuery is only read (not mutated) by MulNew, so sharing is safe.
		var packWg sync.WaitGroup
		for packIdx := range packedPlaintexts {
			packWg.Add(1)
			go func(idx int) {
				defer packWg.Done()
				encResult, err := c.heProvider.HomomorphicBatchDotProduct(
					encPackedQuery, packedPlaintexts[idx], centroidsPerPack, dimension,
				)
				if err != nil {
					packErrors[idx] = fmt.Errorf("failed batch dot product (pack %d): %w", idx, err)
					return
				}
				packScores[idx], packErrors[idx] = c.heProvider.DecryptBatchScalars(
					encResult, centroidsPerPack, dimension,
				)
			}(packIdx)
		}
		packWg.Wait()
	}

	// Check for errors and collect scores
	allScores := make([]float64, 0, numCentroids)
	for i, err := range packErrors {
		if err != nil {
			return nil, fmt.Errorf("pack %d: %w", i, err)
		}
		allScores = append(allScores, packScores[i]...)
	}

	// Truncate to actual number of centroids
	scores := allScores[:numCentroids]

	result.Timing.HECentroidScores = time.Since(startHE)
	result.Stats.HEOperations = len(packedPlaintexts) // Just 1-2 ops instead of 64!

	// Select clusters using multi-probe
	topSupers := selectClusters(
		scores,
		c.config.TopSuperBuckets,
		c.config.ProbeThreshold,
		c.config.MaxProbeClusters,
		c.config.ProbeStrategy,
		c.config.GapMultiplier,
	)
	result.Stats.SuperBucketsSelected = len(topSupers)
	result.Stats.SelectedClusters = topSupers

	// ==========================================
	// LEVEL 2 & 3: Same as Search()
	// ==========================================

	startSelection := time.Now()
	decoySupers := c.generateDecoySupers(topSupers, c.config.NumDecoys)
	allSupers := append(append([]int{}, topSupers...), decoySupers...)
	shuffleInts(allSupers)

	result.Stats.RealSubBuckets = len(topSupers)
	result.Stats.DecoySubBuckets = len(decoySupers)
	result.Stats.TotalSubBuckets = len(allSupers)
	result.Timing.BucketSelection = time.Since(startSelection)

	startFetch := time.Now()
	blobs, err := c.store.GetSuperBuckets(ctx, allSupers)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch buckets: %w", err)
	}
	result.Timing.BucketFetch = time.Since(startFetch)
	result.Stats.BlobsFetched = len(blobs)

	// Filter to real clusters only — skip decoy blobs to save AES decrypt + scoring time.
	realBlobs := filterRealBlobs(blobs, topSupers)

	// Decrypt and score locally (same as Search)
	startAES := time.Now()
	decrypted, batchDecryptStats := parallelDecryptBlobs(realBlobs, c.encryptor)
	result.Timing.AESDecrypt = time.Since(startAES)
	result.Stats.DecryptSucceeded = batchDecryptStats.Succeeded
	result.Stats.DecryptFailed = batchDecryptStats.Failed

	startScore := time.Now()
	scoreMap := make(map[string]float64, len(decrypted))
	for _, d := range decrypted {
		vec := d.vector
		if !c.config.NormalizedStorage {
			vec = normalizeVectorCopy(d.vector)
		}
		score := dotProductVec(normalizedQuery, vec)
		if existing, ok := scoreMap[d.id]; !ok || score > existing {
			scoreMap[d.id] = score
		}
	}

	// Select top-K using min-heap: O(n log K) instead of O(n log n) full sort.
	result.Results = topKFromMap(scoreMap, topK)

	result.Timing.LocalScoring = time.Since(startScore)
	result.Stats.VectorsScored = len(scoreMap)

	result.Timing.Total = time.Since(startTotal)
	return result, nil
}

// generateDecoySupers delegates to the shared package-level function in hierarchical.go.
func (c *EnterpriseHierarchicalClient) generateDecoySupers(selectedSupers []int, numDecoys int) []int {
	return generateDecoySupers(selectedSupers, c.config.NumSuperBuckets, numDecoys)
}

// filterRealBlobs returns only blobs belonging to the specified real super-bucket IDs.
// The server fetches blobs from both real and decoy clusters (it can't tell which is which).
// The client knows which clusters are real, so it skips decoy blobs to avoid wasted
// AES decryption and local scoring. This has no privacy impact — the server never sees
// which blobs are actually processed.
func filterRealBlobs(blobs []*blob.Blob, realSupers []int) []*blob.Blob {
	realSet := make(map[int]bool, len(realSupers))
	for _, id := range realSupers {
		realSet[id] = true
	}

	filtered := make([]*blob.Blob, 0, len(blobs))
	for _, b := range blobs {
		superID := parseSuperBucketID(b.LSHBucket)
		if superID >= 0 && realSet[superID] {
			filtered = append(filtered, b)
		}
	}
	return filtered
}

// parseSuperBucketID extracts the super-bucket ID from a bucket key.
// Bucket keys are "XX" or "XX_YY" where XX is the super-bucket integer ID.
func parseSuperBucketID(bucketKey string) int {
	key := bucketKey
	for i, c := range bucketKey {
		if c == '_' {
			key = bucketKey[:i]
			break
		}
	}
	id := 0
	for _, c := range key {
		if c < '0' || c > '9' {
			return -1
		}
		id = id*10 + int(c-'0')
	}
	if key == "" {
		return -1
	}
	return id
}

// scoredItem is an (id, score) pair used for heap-based top-K selection.
type scoredItem struct {
	id    string
	score float64
}

// minScoreHeap implements a min-heap on score. We keep the K highest-scored
// items by maintaining a min-heap of size K: if a new score exceeds the heap
// minimum we pop the min and push the new item. After processing all items
// the heap contains the top-K, extracted in descending order.
type minScoreHeap []scoredItem

func (h minScoreHeap) Len() int            { return len(h) }
func (h minScoreHeap) Less(i, j int) bool   { return h[i].score < h[j].score } // min-heap
func (h minScoreHeap) Swap(i, j int)        { h[i], h[j] = h[j], h[i] }
func (h *minScoreHeap) Push(x interface{})  { *h = append(*h, x.(scoredItem)) }
func (h *minScoreHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// topKFromMap selects the top-K items from a score map using a min-heap.
// Returns results in descending score order. O(n log K) instead of O(n log n).
func topKFromMap(scoreMap map[string]float64, k int) []hierarchical.Result {
	if k > len(scoreMap) {
		k = len(scoreMap)
	}
	if k == 0 {
		return nil
	}

	h := make(minScoreHeap, 0, k+1)
	heap.Init(&h)

	for id, score := range scoreMap {
		if h.Len() < k {
			heap.Push(&h, scoredItem{id: id, score: score})
		} else if score > h[0].score {
			h[0] = scoredItem{id: id, score: score}
			heap.Fix(&h, 0)
		}
	}

	// Extract in descending order
	results := make([]hierarchical.Result, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		item := heap.Pop(&h).(scoredItem)
		results[i] = hierarchical.Result{ID: item.id, Score: item.score}
	}
	return results
}

// decryptedVec holds a decrypted vector with its original and blob IDs.
type decryptedVec struct {
	id     string    // Original ID (without _dupN suffix)
	blobID string    // Blob ID (may have _dupN suffix)
	vector []float64 // Decrypted vector
}

// DecryptStats reports on blob decryption outcomes.
type DecryptStats struct {
	Succeeded int // Vectors successfully decrypted
	Failed    int // Decryption failures (expected for decoys)
}

// parallelDecryptBlobs decrypts blobs in parallel using multiple goroutines.
// Decoy blobs are expected to fail decryption and are skipped.
// Returns decrypted vectors and statistics about the decryption outcomes.
func parallelDecryptBlobs(blobs []*blob.Blob, encryptor *encrypt.AESGCM) ([]decryptedVec, DecryptStats) {
	if len(blobs) == 0 {
		return nil, DecryptStats{}
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > len(blobs) {
		numWorkers = len(blobs)
	}
	chunkSize := (len(blobs) + numWorkers - 1) / numWorkers

	// Each worker writes to its own slice to avoid locking
	workerResults := make([][]decryptedVec, numWorkers)
	workerFailed := make([]int, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(blobs) {
			end = len(blobs)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(w, start, end int) {
			defer wg.Done()
			local := make([]decryptedVec, 0, end-start)
			for i := start; i < end; i++ {
				b := blobs[i]
				origID := extractOriginalID(b.ID)
				vec, err := encryptor.DecryptVectorWithID(b.Ciphertext, origID)
				if err != nil {
					workerFailed[w]++
					continue // Skip failed decryptions (expected for decoy blobs)
				}
				local = append(local, decryptedVec{id: origID, blobID: b.ID, vector: vec})
			}
			workerResults[w] = local
		}(w, start, end)
	}
	wg.Wait()

	// Merge results
	total := 0
	totalFailed := 0
	for i, r := range workerResults {
		total += len(r)
		totalFailed += workerFailed[i]
	}
	result := make([]decryptedVec, 0, total)
	for _, r := range workerResults {
		result = append(result, r...)
	}
	return result, DecryptStats{Succeeded: total, Failed: totalFailed}
}

// shuffleInts shuffles a slice of ints in place using crypto/rand.
func shuffleInts(s []int) {
	for i := len(s) - 1; i > 0; i-- {
		jBig, _ := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		j := int(jBig.Int64())
		s[i], s[j] = s[j], s[i]
	}
}

// GetCredentials returns the current credentials.
func (c *EnterpriseHierarchicalClient) GetCredentials() *auth.ClientCredentials {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.credentials
}

// UpdateCredentials updates the client with refreshed credentials.
// Call this after refreshing the token with the auth service.
// Also reloads centroid cache if centroids changed.
func (c *EnterpriseHierarchicalClient) UpdateCredentials(creds *auth.ClientCredentials) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	encryptor, err := encrypt.NewAESGCM(creds.AESKey)
	if err != nil {
		return fmt.Errorf("failed to create encryptor: %w", err)
	}

	c.credentials = creds
	c.encryptor = encryptor

	// Reload centroid caches if centroids changed.
	if len(creds.Centroids) > 0 {
		level := c.heProvider.GetParams().MaxLevel()
		if c.centroidCache != nil && c.centroidCache.NeedsRefresh(creds.Centroids) {
			if err := c.centroidCache.LoadCentroids(creds.Centroids, level); err != nil {
				return fmt.Errorf("failed to reload centroid cache: %w", err)
			}
		}
		if c.batchCentroidCache != nil {
			if err := c.batchCentroidCache.LoadCentroids(creds.Centroids, level); err != nil {
				return fmt.Errorf("failed to reload batch centroid cache: %w", err)
			}
		}
	}

	return nil
}

// IsTokenExpired checks if the token needs refresh.
func (c *EnterpriseHierarchicalClient) IsTokenExpired() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.credentials.IsExpired()
}

// TimeUntilTokenExpiry returns the duration until the token expires.
func (c *EnterpriseHierarchicalClient) TimeUntilTokenExpiry() time.Duration {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.credentials.TimeUntilExpiry()
}

// GetEnterpriseID returns the enterprise ID from credentials.
func (c *EnterpriseHierarchicalClient) GetEnterpriseID() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.credentials.EnterpriseID
}

// helper functions (reused from hierarchical.go)

func normalizeVectorCopyEnt(v []float64) []float64 {
	result := make([]float64, len(v))
	copy(result, v)

	var norm float64
	for _, val := range result {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for i := range result {
			result[i] /= norm
		}
	}

	return result
}

func dotProductVecEnt(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// extractOriginalID extracts the original vector ID from a blob ID.
// Blob IDs may have a "_dupN" suffix for redundant cluster assignment.
// Example: "doc-123_dup1" -> "doc-123", "doc-123" -> "doc-123"
func extractOriginalID(blobID string) string {
	if idx := strings.Index(blobID, "_dup"); idx > 0 {
		return blobID[:idx]
	}
	return blobID
}
