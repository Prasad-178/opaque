package client

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"sort"
	"sync"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/cache"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/hierarchical"
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
type EnterpriseHierarchicalClient struct {
	config      hierarchical.Config
	credentials *auth.ClientCredentials

	// Derived from credentials
	encryptor *encrypt.AESGCM
	heEngine  *crypto.Engine

	// Centroid cache for faster HE operations
	// Pre-encodes centroids as HE plaintexts
	centroidCache *cache.CentroidCache

	// Storage backend
	store blob.Store

	mu sync.RWMutex
}

// NewEnterpriseHierarchicalClient creates a client from authenticated credentials.
// The credentials contain:
//   - AES key for vector decryption
//   - Centroids for HE scoring
func NewEnterpriseHierarchicalClient(
	cfg hierarchical.Config,
	credentials *auth.ClientCredentials,
	store blob.Store,
) (*EnterpriseHierarchicalClient, error) {
	if credentials == nil {
		return nil, fmt.Errorf("credentials are required")
	}
	if store == nil {
		return nil, fmt.Errorf("store is required")
	}

	// Create encryptor from credentials
	encryptor, err := encrypt.NewAESGCM(credentials.AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create encryptor: %w", err)
	}

	// Create HE engine for query encryption
	heEngine, err := crypto.NewClientEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create HE engine: %w", err)
	}

	// Override config from credentials if not set
	if cfg.Dimension <= 0 {
		cfg.Dimension = credentials.Dimension
	}
	if cfg.NumSuperBuckets <= 0 {
		cfg.NumSuperBuckets = credentials.NumSuperBuckets
	}

	// Create centroid cache and pre-load centroids
	centroidCache := cache.NewCentroidCache(heEngine.GetParams(), heEngine.GetEncoder())
	if len(credentials.Centroids) > 0 {
		// Pre-encode all centroids (expensive but done once)
		if err := centroidCache.LoadCentroids(credentials.Centroids, heEngine.GetParams().MaxLevel()); err != nil {
			return nil, fmt.Errorf("failed to load centroids into cache: %w", err)
		}
	}

	return &EnterpriseHierarchicalClient{
		config:        cfg,
		credentials:   credentials,
		encryptor:     encryptor,
		heEngine:      heEngine,
		centroidCache: centroidCache,
		store:         store,
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
	encQuery, err := c.heEngine.EncryptVector(normalizedQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt query: %w", err)
	}
	result.Timing.HEEncryptQuery = time.Since(startEncrypt)

	// Step 1b: Compute HE scores for ALL centroids (parallel)
	// Uses cached pre-encoded plaintexts for faster computation
	// In a real client-server setup, this would be done on the server
	startHE := time.Now()
	centroids := c.credentials.Centroids
	encScores := make([]*rlwe.Ciphertext, len(centroids))
	scoreErrs := make([]error, len(centroids))

	// Get cached plaintexts (pre-encoded centroids)
	cachedPlaintexts := c.centroidCache.GetAll(len(centroids))
	useCached := cachedPlaintexts != nil && len(cachedPlaintexts) == len(centroids) && cachedPlaintexts[0] != nil

	var wg sync.WaitGroup
	numWorkers := 4
	workChan := make(chan int, len(centroids))

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range workChan {
				if useCached {
					// Use cached pre-encoded plaintext (faster)
					encScores[i], scoreErrs[i] = c.heEngine.HomomorphicDotProductCached(encQuery, cachedPlaintexts[i])
				} else {
					// Fallback to encoding on-the-fly
					encScores[i], scoreErrs[i] = c.heEngine.HomomorphicDotProduct(encQuery, centroids[i])
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
	startDecrypt := time.Now()
	scores := make([]float64, len(encScores))

	for i, encScore := range encScores {
		score, err := c.heEngine.DecryptScalar(encScore)
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt score %d: %w", i, err)
		}
		scores[i] = score
	}
	result.Timing.HEDecryptScores = time.Since(startDecrypt)

	// Step 1d: Select top super-buckets (SERVER NEVER SEES THIS!)
	topSupers := selectTopKIndices(scores, c.config.TopSuperBuckets)
	result.Stats.SuperBucketsSelected = len(topSupers)

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

	// Decrypt all vectors
	startAES := time.Now()
	type decryptedVec struct {
		id     string
		vector []float64
	}
	decrypted := make([]decryptedVec, 0, len(blobs))

	for _, b := range blobs {
		vec, err := c.encryptor.DecryptVectorWithID(b.Ciphertext, b.ID)
		if err != nil {
			continue // Skip failed decryptions (likely decoy or corrupted)
		}
		decrypted = append(decrypted, decryptedVec{id: b.ID, vector: vec})
	}
	result.Timing.AESDecrypt = time.Since(startAES)

	// Score locally
	startScore := time.Now()
	type scored struct {
		id    string
		score float64
	}
	scoredResults := make([]scored, len(decrypted))

	for i, d := range decrypted {
		normalizedVec := normalizeVectorCopy(d.vector)
		score := dotProductVec(normalizedQuery, normalizedVec)
		scoredResults[i] = scored{id: d.id, score: score}
	}

	// Sort by score descending
	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].score > scoredResults[j].score
	})

	result.Timing.LocalScoring = time.Since(startScore)
	result.Stats.VectorsScored = len(scoredResults)

	// Return top-K
	n := topK
	if n > len(scoredResults) {
		n = len(scoredResults)
	}
	result.Results = make([]hierarchical.Result, n)
	for i := 0; i < n; i++ {
		result.Results[i] = hierarchical.Result{
			ID:    scoredResults[i].id,
			Score: scoredResults[i].score,
		}
	}

	result.Timing.Total = time.Since(startTotal)
	return result, nil
}

// generateDecoySupers generates decoy super-bucket IDs from non-selected super-buckets.
func (c *EnterpriseHierarchicalClient) generateDecoySupers(selectedSupers []int, numDecoys int) []int {
	if numDecoys <= 0 {
		return nil
	}

	// Create set of selected super-buckets
	selected := make(map[int]bool)
	for _, s := range selectedSupers {
		selected[s] = true
	}

	// Collect all non-selected super-buckets
	var nonSelected []int
	for i := 0; i < c.config.NumSuperBuckets; i++ {
		if !selected[i] {
			nonSelected = append(nonSelected, i)
		}
	}

	if len(nonSelected) == 0 {
		return nil
	}

	// Generate random decoy super-buckets
	decoys := make([]int, 0, numDecoys)
	for i := 0; i < numDecoys && i < len(nonSelected); i++ {
		// Pick random non-selected super-bucket
		idx, _ := rand.Int(rand.Reader, big.NewInt(int64(len(nonSelected))))
		superID := nonSelected[idx.Int64()]

		// Remove from pool to avoid duplicates
		nonSelected = append(nonSelected[:idx.Int64()], nonSelected[idx.Int64()+1:]...)
		decoys = append(decoys, superID)
	}

	return decoys
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

	// Reload centroid cache if centroids changed
	if c.centroidCache != nil && len(creds.Centroids) > 0 {
		if c.centroidCache.NeedsRefresh(creds.Centroids) {
			if err := c.centroidCache.LoadCentroids(creds.Centroids, c.heEngine.GetParams().MaxLevel()); err != nil {
				return fmt.Errorf("failed to reload centroid cache: %w", err)
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
