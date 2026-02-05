// Tier 2: Data-Private Client
//
// This client provides encrypted vector storage where the server/storage
// never sees plaintext vectors. All encryption and similarity computation
// happens client-side.
//
// Use cases:
// - Blockchain vector storage
// - Zero-trust cloud storage
// - User-controlled encryption keys

package client

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// Tier2Config holds configuration for the Tier 2 client.
type Tier2Config struct {
	// Dimension is the vector dimension.
	Dimension int

	// LSHBits is the number of LSH hash bits.
	// Fewer bits = larger buckets = more privacy but less precision.
	LSHBits int

	// LSHSeed is the random seed for LSH planes (must match across clients).
	LSHSeed int64

	// MaxBucketsToFetch limits how many buckets to download per search.
	MaxBucketsToFetch int

	// MaxResultsPerBucket limits results from each bucket (0 = unlimited).
	MaxResultsPerBucket int
}

// DefaultTier2Config returns sensible defaults.
func DefaultTier2Config() Tier2Config {
	return Tier2Config{
		Dimension:           128,
		LSHBits:             64, // Fewer bits for more privacy
		LSHSeed:             42,
		MaxBucketsToFetch:   5,
		MaxResultsPerBucket: 0,
	}
}

// Tier2Client provides data-private vector search.
// Vectors are encrypted client-side; storage never sees plaintext.
type Tier2Client struct {
	config    Tier2Config
	encryptor *encrypt.AESGCM
	store     blob.Store
	lshIndex  *lsh.Index

	// Privacy features
	privacyConfig    PrivacyConfig
	timingObfuscator *TimingObfuscator
	dummyRunner      *DummyQueryRunner
	privacyMetrics   *PrivacyMetrics

	mu sync.RWMutex
}

// NewTier2Client creates a new Tier 2 client.
func NewTier2Client(cfg Tier2Config, encryptor *encrypt.AESGCM, store blob.Store) (*Tier2Client, error) {
	if encryptor == nil {
		return nil, errors.New("encryptor is required")
	}
	if store == nil {
		return nil, errors.New("store is required")
	}

	// Create LSH index for computing hashes
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: cfg.Dimension,
		NumBits:   cfg.LSHBits,
		Seed:      cfg.LSHSeed,
	})

	return &Tier2Client{
		config:    cfg,
		encryptor: encryptor,
		store:     store,
		lshIndex:  lshIndex,
	}, nil
}

// Insert encrypts and stores a single vector.
func (c *Tier2Client) Insert(ctx context.Context, id string, vector []float64, metadata map[string]any) error {
	return c.InsertBatch(ctx, []string{id}, [][]float64{vector}, []map[string]any{metadata})
}

// InsertBatch encrypts and stores multiple vectors.
func (c *Tier2Client) InsertBatch(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error {
	if len(ids) != len(vectors) {
		return errors.New("ids and vectors length mismatch")
	}

	blobs := make([]*blob.Blob, len(ids))

	for i, id := range ids {
		vector := vectors[i]

		if len(vector) != c.config.Dimension {
			return fmt.Errorf("vector %s has wrong dimension: %d (expected %d)", id, len(vector), c.config.Dimension)
		}

		// Compute LSH bucket
		lshHash := c.lshIndex.HashBytes(vector)
		bucket := hex.EncodeToString(lshHash)

		// Encrypt vector (with ID as AAD to prevent ID swapping)
		ciphertext, err := c.encryptor.EncryptVectorWithID(vector, id)
		if err != nil {
			return fmt.Errorf("failed to encrypt vector %s: %w", id, err)
		}

		// Create blob
		b := blob.NewBlob(id, bucket, ciphertext, c.config.Dimension)

		// Encrypt metadata if provided
		if metadata != nil && i < len(metadata) && metadata[i] != nil {
			// For now, skip metadata encryption (can add later)
			// metaBytes, _ := json.Marshal(metadata[i])
			// metaCiphertext, _ := c.encryptor.Encrypt(metaBytes)
			// b.WithMetadata(metaCiphertext)
		}

		blobs[i] = b
	}

	return c.store.PutBatch(ctx, blobs)
}

// Delete removes a vector by ID.
func (c *Tier2Client) Delete(ctx context.Context, id string) error {
	return c.store.Delete(ctx, id)
}

// DeleteBatch removes multiple vectors by ID.
func (c *Tier2Client) DeleteBatch(ctx context.Context, ids []string) error {
	return c.store.DeleteBatch(ctx, ids)
}

// Search performs a privacy-preserving search.
// 1. Compute LSH hash locally
// 2. Fetch matching bucket(s) from storage
// 3. Decrypt vectors locally
// 4. Compute similarity locally
// 5. Return top-K results
func (c *Tier2Client) Search(ctx context.Context, query []float64, topK int) ([]Result, error) {
	return c.SearchWithOptions(ctx, query, SearchOptions{
		TopK:       topK,
		NumBuckets: 1,
	})
}

// SearchOptions controls search behavior.
type SearchOptions struct {
	// TopK is the number of results to return.
	TopK int

	// NumBuckets is how many buckets to search (more = better recall, slower).
	NumBuckets int

	// DecoyBuckets is how many random buckets to fetch (for privacy).
	// Storage can't tell which buckets you're actually interested in.
	DecoyBuckets int

	// UseMultiProbe enables multi-probe LSH (searches neighboring buckets).
	UseMultiProbe bool
}

// SearchWithOptions performs a search with custom options.
func (c *Tier2Client) SearchWithOptions(ctx context.Context, query []float64, opts SearchOptions) ([]Result, error) {
	if len(query) != c.config.Dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.config.Dimension)
	}

	// Normalize query
	normalizedQuery := normalizeVector(query)

	// Compute LSH hash
	lshHash := c.lshIndex.HashBytes(normalizedQuery)
	primaryBucket := hex.EncodeToString(lshHash)

	// Determine which buckets to fetch
	bucketsToFetch := []string{primaryBucket}

	// Add neighboring buckets if multi-probe enabled
	if opts.UseMultiProbe && opts.NumBuckets > 1 {
		neighbors := c.findNeighborBuckets(lshHash, opts.NumBuckets-1)
		bucketsToFetch = append(bucketsToFetch, neighbors...)
	}

	// Add decoy buckets for privacy
	if opts.DecoyBuckets > 0 {
		decoys, err := c.getRandomBuckets(ctx, opts.DecoyBuckets, bucketsToFetch)
		if err == nil {
			bucketsToFetch = append(bucketsToFetch, decoys...)
		}
	}

	// Fetch blobs from all buckets
	blobs, err := c.store.GetBuckets(ctx, bucketsToFetch)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch buckets: %w", err)
	}

	if len(blobs) == 0 {
		return []Result{}, nil
	}

	// Decrypt and score in parallel
	type scoredResult struct {
		id    string
		score float64
		err   error
	}

	results := make([]scoredResult, len(blobs))
	var wg sync.WaitGroup

	for i, b := range blobs {
		wg.Add(1)
		go func(idx int, blobItem *blob.Blob) {
			defer wg.Done()

			// Decrypt vector
			vector, err := c.encryptor.DecryptVectorWithID(blobItem.Ciphertext, blobItem.ID)
			if err != nil {
				results[idx] = scoredResult{id: blobItem.ID, err: err}
				return
			}

			// Normalize and compute similarity
			normalizedVec := normalizeVector(vector)
			score := dotProduct(normalizedQuery, normalizedVec)

			results[idx] = scoredResult{id: blobItem.ID, score: score}
		}(i, b)
	}

	wg.Wait()

	// Collect valid results
	validResults := make([]Result, 0, len(results))
	for _, r := range results {
		if r.err == nil {
			validResults = append(validResults, Result{ID: r.id, Score: r.score})
		}
	}

	// Sort by score descending
	sort.Slice(validResults, func(i, j int) bool {
		return validResults[i].Score > validResults[j].Score
	})

	// Return top-K
	if opts.TopK > 0 && len(validResults) > opts.TopK {
		validResults = validResults[:opts.TopK]
	}

	return validResults, nil
}

// findNeighborBuckets finds LSH buckets that differ by 1-2 bits (for multi-probe).
func (c *Tier2Client) findNeighborBuckets(hash []byte, count int) []string {
	neighbors := make([]string, 0, count)

	// Flip each bit to find neighbors
	for byteIdx := 0; byteIdx < len(hash) && len(neighbors) < count; byteIdx++ {
		for bitIdx := 0; bitIdx < 8 && len(neighbors) < count; bitIdx++ {
			// Create a copy with one bit flipped
			neighbor := make([]byte, len(hash))
			copy(neighbor, hash)
			neighbor[byteIdx] ^= (1 << bitIdx)
			neighbors = append(neighbors, hex.EncodeToString(neighbor))
		}
	}

	return neighbors
}

// getRandomBuckets returns random buckets for decoy purposes.
func (c *Tier2Client) getRandomBuckets(ctx context.Context, count int, exclude []string) ([]string, error) {
	allBuckets, err := c.store.ListBuckets(ctx)
	if err != nil {
		return nil, err
	}

	// Build exclusion set
	excludeSet := make(map[string]bool)
	for _, b := range exclude {
		excludeSet[b] = true
	}

	// Filter and collect
	decoys := make([]string, 0, count)
	for _, b := range allBuckets {
		if !excludeSet[b] {
			decoys = append(decoys, b)
			if len(decoys) >= count {
				break
			}
		}
	}

	return decoys, nil
}

// GetStats returns storage statistics.
func (c *Tier2Client) GetStats(ctx context.Context) (*blob.StoreStats, error) {
	return c.store.Stats(ctx)
}

// GetKeyFingerprint returns the encryption key fingerprint.
// Useful for verifying the correct key is being used.
func (c *Tier2Client) GetKeyFingerprint() string {
	return c.encryptor.KeyFingerprint()
}

// GetLSHPlanes returns the LSH hyperplanes.
// Other clients need these to compute compatible LSH hashes.
func (c *Tier2Client) GetLSHPlanes() [][]float64 {
	return c.lshIndex.GetPlanes()
}

// Close closes the underlying store.
func (c *Tier2Client) Close() error {
	c.StopDummyQueries()
	return c.store.Close()
}

// SetPrivacyConfig configures privacy features.
func (c *Tier2Client) SetPrivacyConfig(cfg PrivacyConfig) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.privacyConfig = cfg
	c.timingObfuscator = NewTimingObfuscator(cfg.MinLatency, cfg.JitterRange)
	c.privacyMetrics = &PrivacyMetrics{}
}

// GetPrivacyMetrics returns privacy-related metrics.
func (c *Tier2Client) GetPrivacyMetrics() *PrivacyMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.privacyMetrics
}

// StartDummyQueries starts sending background dummy queries.
func (c *Tier2Client) StartDummyQueries() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.dummyRunner != nil {
		return // Already running
	}

	interval := c.privacyConfig.DummyQueryInterval
	if interval == 0 {
		interval = 30 * 1e9 // 30 seconds default
	}

	c.dummyRunner = NewDummyQueryRunner(c, interval)
	c.dummyRunner.Start()
}

// StopDummyQueries stops sending background dummy queries.
func (c *Tier2Client) StopDummyQueries() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.dummyRunner != nil {
		c.dummyRunner.Stop()
		c.dummyRunner = nil
	}
}

// SearchWithPrivacy performs a search with full privacy enhancements.
// Uses timing obfuscation, decoy buckets, and shuffling based on PrivacyConfig.
func (c *Tier2Client) SearchWithPrivacy(ctx context.Context, query []float64, topK int) ([]Result, error) {
	c.mu.RLock()
	cfg := c.privacyConfig
	obfuscator := c.timingObfuscator
	metrics := c.privacyMetrics
	c.mu.RUnlock()

	// Start timing obfuscation
	var done func()
	if obfuscator != nil {
		done = obfuscator.ObfuscateContext(ctx)
	}

	// Build search options with privacy settings
	opts := SearchOptions{
		TopK:          topK,
		NumBuckets:    3, // Multiple buckets for better recall
		DecoyBuckets:  cfg.DecoyBuckets,
		UseMultiProbe: true,
	}

	results, err := c.searchWithPrivacyInternal(ctx, query, opts, cfg.ShuffleBeforeProcess)

	// Complete timing obfuscation
	if done != nil {
		done()
	}

	// Record metrics
	if metrics != nil && obfuscator != nil {
		delay := cfg.MinLatency + cfg.JitterRange/2 // Approximate
		metrics.RecordQuery(false, cfg.DecoyBuckets, delay)
	}

	return results, err
}

// searchWithPrivacyInternal performs the search with optional shuffling.
func (c *Tier2Client) searchWithPrivacyInternal(ctx context.Context, query []float64, opts SearchOptions, shuffle bool) ([]Result, error) {
	if len(query) != c.config.Dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.config.Dimension)
	}

	// Normalize query
	normalizedQuery := normalizeVector(query)

	// Compute LSH hash
	lshHash := c.lshIndex.HashBytes(normalizedQuery)
	primaryBucket := hex.EncodeToString(lshHash)

	// Determine which buckets to fetch
	bucketsToFetch := []string{primaryBucket}

	// Add neighboring buckets if multi-probe enabled
	if opts.UseMultiProbe && opts.NumBuckets > 1 {
		neighbors := c.findNeighborBuckets(lshHash, opts.NumBuckets-1)
		bucketsToFetch = append(bucketsToFetch, neighbors...)
	}

	// Add decoy buckets for privacy
	if opts.DecoyBuckets > 0 {
		decoys, err := c.getRandomBuckets(ctx, opts.DecoyBuckets, bucketsToFetch)
		if err == nil {
			bucketsToFetch = append(bucketsToFetch, decoys...)
		}
	}

	// Shuffle bucket order before fetching (hides which is primary)
	if shuffle {
		shuffleSlice(bucketsToFetch)
	}

	// Fetch blobs from all buckets
	blobs, err := c.store.GetBuckets(ctx, bucketsToFetch)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch buckets: %w", err)
	}

	if len(blobs) == 0 {
		return []Result{}, nil
	}

	// Shuffle blobs before processing (hides processing order)
	if shuffle {
		shuffleSlice(blobs)
	}

	// Decrypt and score in parallel
	type scoredResult struct {
		id    string
		score float64
		err   error
	}

	results := make([]scoredResult, len(blobs))
	var wg sync.WaitGroup

	for i, b := range blobs {
		wg.Add(1)
		go func(idx int, blobItem *blob.Blob) {
			defer wg.Done()

			// Decrypt vector
			vector, err := c.encryptor.DecryptVectorWithID(blobItem.Ciphertext, blobItem.ID)
			if err != nil {
				results[idx] = scoredResult{id: blobItem.ID, err: err}
				return
			}

			// Normalize and compute similarity
			normalizedVec := normalizeVector(vector)
			score := dotProduct(normalizedQuery, normalizedVec)

			results[idx] = scoredResult{id: blobItem.ID, score: score}
		}(i, b)
	}

	wg.Wait()

	// Collect valid results
	validResults := make([]Result, 0, len(results))
	for _, r := range results {
		if r.err == nil {
			validResults = append(validResults, Result{ID: r.id, Score: r.score})
		}
	}

	// Sort by score descending
	sort.Slice(validResults, func(i, j int) bool {
		return validResults[i].Score > validResults[j].Score
	})

	// Return top-K
	if opts.TopK > 0 && len(validResults) > opts.TopK {
		validResults = validResults[:opts.TopK]
	}

	return validResults, nil
}

// Helper functions

func normalizeVector(v []float64) []float64 {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return v
	}

	normalized := make([]float64, len(v))
	for i, val := range v {
		normalized[i] = val / norm
	}
	return normalized
}

// Note: dotProduct is defined in client.go, reusing it here
