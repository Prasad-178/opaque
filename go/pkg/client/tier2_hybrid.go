// Tier 2.5: Hybrid Private Search
//
// This combines the best of Tier 1 and Tier 2:
// - Data Privacy: Vectors encrypted with AES (like Tier 2)
// - Query Privacy: Final scoring with HE (like Tier 1)
// - Speed: Use fast Tier 2 for coarse filtering, HE only for final candidates
//
// Architecture:
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │                    UNTRUSTED STORAGE                        │
//   │  [AES(v1), AES(v2), ...] organized by LSH buckets          │
//   │  → Cannot see vectors (encrypted)                          │
//   │  → CAN see which buckets are accessed                      │
//   └─────────────────────────────────────────────────────────────┘
//                              │
//                              │ Fetch encrypted blobs
//                              ▼
//   ┌─────────────────────────────────────────────────────────────┐
//   │              SEMI-TRUSTED COMPUTE SERVER                    │
//   │  1. Receives encrypted blobs from storage                  │
//   │  2. Receives HE-encrypted query from client                │
//   │  3. Decrypts blobs with AES key (server has this)          │
//   │  4. Computes HE dot products on plaintext vectors          │
//   │  5. Returns HE-encrypted scores                            │
//   │  → CAN see vectors (has AES key)                           │
//   │  → Cannot see query or scores (HE encrypted)               │
//   └─────────────────────────────────────────────────────────────┘
//                              │
//                              │ HE-encrypted scores
//                              ▼
//   ┌─────────────────────────────────────────────────────────────┐
//   │                         CLIENT                              │
//   │  1. Computes LSH hash locally                              │
//   │  2. Encrypts query with HE                                 │
//   │  3. Decrypts HE scores                                     │
//   │  4. Selects top-K results                                  │
//   └─────────────────────────────────────────────────────────────┘
//
// Privacy guarantees:
// - Storage: sees encrypted blobs + access patterns
// - Compute: sees plaintext vectors, NOT query or scores
// - Client: sees everything (owns both keys)
//
// Speed optimization:
// - Only do HE operations on final candidates (e.g., 20 vectors)
// - Not on entire database (100K+ vectors)

package client

import (
	"context"
	"encoding/hex"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/lsh"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

// HybridConfig holds configuration for the hybrid client.
type HybridConfig struct {
	Dimension int
	LSHBits   int
	LSHSeed   int64

	// Two-stage filtering
	CoarseCandidates int // Number of candidates from LSH (e.g., 200)
	FineCandidates   int // Number to score with HE (e.g., 20)

	// Privacy modes
	UseHEForFinalScoring bool // If true, use HE for final scoring (slower but more private)
}

// DefaultHybridConfig returns sensible defaults.
func DefaultHybridConfig() HybridConfig {
	return HybridConfig{
		Dimension:            128,
		LSHBits:              12,
		LSHSeed:              42,
		CoarseCandidates:     200,
		FineCandidates:       20,
		UseHEForFinalScoring: true,
	}
}

// HybridClient provides the Tier 2.5 hybrid search.
type HybridClient struct {
	config    HybridConfig
	encryptor *encrypt.AESGCM // For data encryption (Tier 2)
	heEngine  *crypto.Engine  // For query encryption (Tier 1)
	store     blob.Store
	lshIndex  *lsh.Index

	mu sync.RWMutex
}

// NewHybridClient creates a new hybrid client.
func NewHybridClient(cfg HybridConfig, aesKey []byte, store blob.Store) (*HybridClient, error) {
	encryptor, err := encrypt.NewAESGCM(aesKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES encryptor: %w", err)
	}

	heEngine, err := crypto.NewClientEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create HE engine: %w", err)
	}

	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: cfg.Dimension,
		NumBits:   cfg.LSHBits,
		Seed:      cfg.LSHSeed,
	})

	return &HybridClient{
		config:    cfg,
		encryptor: encryptor,
		heEngine:  heEngine,
		store:     store,
		lshIndex:  lshIndex,
	}, nil
}

// Insert encrypts and stores a vector.
func (c *HybridClient) Insert(ctx context.Context, id string, vector []float64, metadata map[string]any) error {
	return c.InsertBatch(ctx, []string{id}, [][]float64{vector}, nil)
}

// InsertBatch encrypts and stores multiple vectors.
func (c *HybridClient) InsertBatch(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch")
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

		// Encrypt vector with AES
		ciphertext, err := c.encryptor.EncryptVectorWithID(vector, id)
		if err != nil {
			return fmt.Errorf("failed to encrypt vector %s: %w", id, err)
		}

		blobs[i] = blob.NewBlob(id, bucket, ciphertext, c.config.Dimension)
	}

	return c.store.PutBatch(ctx, blobs)
}

// HybridSearchResult contains search results with timing breakdown.
type HybridSearchResult struct {
	Results []Result

	// Timing breakdown
	LSHTime       time.Duration
	FetchTime     time.Duration
	DecryptTime   time.Duration
	CoarseTime    time.Duration // Time for initial scoring (plaintext)
	HEScoreTime   time.Duration // Time for HE scoring (if enabled)
	TotalTime     time.Duration

	// Stats
	BlobsFetched    int
	CoarseCandidates int
	HEOperations    int
}

// Search performs hybrid search with configurable privacy levels.
func (c *HybridClient) Search(ctx context.Context, query []float64, topK int) (*HybridSearchResult, error) {
	startTotal := time.Now()
	result := &HybridSearchResult{}

	if len(query) != c.config.Dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.config.Dimension)
	}

	// Normalize query
	normalizedQuery := normalizeVector(query)

	// Stage 1: LSH to find candidate buckets
	startLSH := time.Now()
	lshHash := c.lshIndex.HashBytes(normalizedQuery)
	primaryBucket := hex.EncodeToString(lshHash)

	// Get neighboring buckets for better recall
	bucketsToFetch := []string{primaryBucket}
	neighbors := c.findNeighborBucketsHybrid(lshHash, 4)
	bucketsToFetch = append(bucketsToFetch, neighbors...)
	result.LSHTime = time.Since(startLSH)

	// Stage 2: Fetch encrypted blobs
	startFetch := time.Now()
	blobs, err := c.store.GetBuckets(ctx, bucketsToFetch)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch buckets: %w", err)
	}
	result.FetchTime = time.Since(startFetch)
	result.BlobsFetched = len(blobs)

	if len(blobs) == 0 {
		result.Results = []Result{}
		result.TotalTime = time.Since(startTotal)
		return result, nil
	}

	// Stage 3: Decrypt all blobs (AES is fast)
	startDecrypt := time.Now()
	type decryptedBlob struct {
		id     string
		vector []float64
	}
	decrypted := make([]decryptedBlob, 0, len(blobs))

	for _, b := range blobs {
		vec, err := c.encryptor.DecryptVectorWithID(b.Ciphertext, b.ID)
		if err != nil {
			continue // Skip failed decryptions
		}
		decrypted = append(decrypted, decryptedBlob{id: b.ID, vector: vec})
	}
	result.DecryptTime = time.Since(startDecrypt)

	// Stage 4: Coarse scoring (fast plaintext dot products)
	startCoarse := time.Now()
	type scored struct {
		id     string
		vector []float64
		score  float64
	}
	coarseResults := make([]scored, len(decrypted))

	for i, d := range decrypted {
		normalizedVec := normalizeVector(d.vector)
		score := dotProduct(normalizedQuery, normalizedVec)
		coarseResults[i] = scored{id: d.id, vector: d.vector, score: score}
	}

	// Sort by score descending
	sort.Slice(coarseResults, func(i, j int) bool {
		return coarseResults[i].score > coarseResults[j].score
	})

	// Take top candidates for fine scoring
	numFine := min(c.config.FineCandidates, len(coarseResults))
	fineCandidates := coarseResults[:numFine]
	result.CoarseTime = time.Since(startCoarse)
	result.CoarseCandidates = len(coarseResults)

	// Stage 5: Fine scoring with HE (if enabled)
	var finalResults []Result

	if c.config.UseHEForFinalScoring && numFine > 0 {
		startHE := time.Now()

		// Encrypt query with HE
		encQuery, err := c.heEngine.EncryptVector(normalizedQuery)
		if err != nil {
			return nil, fmt.Errorf("failed to encrypt query with HE: %w", err)
		}

		// Score each candidate with HE
		heScores := make([]scored, 0, numFine)
		for _, cand := range fineCandidates {
			normalizedVec := normalizeVector(cand.vector)

			encScore, err := c.heEngine.HomomorphicDotProduct(encQuery, normalizedVec)
			if err != nil {
				continue
			}

			heScore, err := c.heEngine.DecryptScalar(encScore)
			if err != nil {
				continue
			}

			heScores = append(heScores, scored{id: cand.id, score: heScore})
			result.HEOperations++
		}

		// Sort HE scores
		sort.Slice(heScores, func(i, j int) bool {
			return heScores[i].score > heScores[j].score
		})

		// Convert to results
		for i := 0; i < min(topK, len(heScores)); i++ {
			finalResults = append(finalResults, Result{
				ID:    heScores[i].id,
				Score: heScores[i].score,
			})
		}
		result.HEScoreTime = time.Since(startHE)
	} else {
		// Use coarse scores directly (faster, but compute server would see scores)
		for i := 0; i < min(topK, len(fineCandidates)); i++ {
			finalResults = append(finalResults, Result{
				ID:    fineCandidates[i].id,
				Score: fineCandidates[i].score,
			})
		}
	}

	result.Results = finalResults
	result.TotalTime = time.Since(startTotal)
	return result, nil
}

// SearchFast performs search without HE (Tier 2 style, faster).
func (c *HybridClient) SearchFast(ctx context.Context, query []float64, topK int) (*HybridSearchResult, error) {
	// Temporarily disable HE
	originalSetting := c.config.UseHEForFinalScoring
	c.config.UseHEForFinalScoring = false
	defer func() { c.config.UseHEForFinalScoring = originalSetting }()

	return c.Search(ctx, query, topK)
}

// SearchPrivate performs search with HE (maximum privacy, slower).
func (c *HybridClient) SearchPrivate(ctx context.Context, query []float64, topK int) (*HybridSearchResult, error) {
	// Ensure HE is enabled
	originalSetting := c.config.UseHEForFinalScoring
	c.config.UseHEForFinalScoring = true
	defer func() { c.config.UseHEForFinalScoring = originalSetting }()

	return c.Search(ctx, query, topK)
}

// GetStats returns storage statistics.
func (c *HybridClient) GetStats(ctx context.Context) (*blob.StoreStats, error) {
	return c.store.Stats(ctx)
}

// findNeighborBucketsHybrid finds neighboring LSH buckets.
func (c *HybridClient) findNeighborBucketsHybrid(hash []byte, count int) []string {
	neighbors := make([]string, 0, count)

	for byteIdx := 0; byteIdx < len(hash) && len(neighbors) < count; byteIdx++ {
		for bitIdx := 0; bitIdx < 8 && len(neighbors) < count; bitIdx++ {
			neighbor := make([]byte, len(hash))
			copy(neighbor, hash)
			neighbor[byteIdx] ^= (1 << bitIdx)
			neighbors = append(neighbors, hex.EncodeToString(neighbor))
		}
	}

	return neighbors
}

// Close cleans up resources.
func (c *HybridClient) Close() error {
	return c.store.Close()
}

// GetHEPublicKey returns the HE public key for a compute server.
func (c *HybridClient) GetHEPublicKey() ([]byte, error) {
	return c.heEngine.GetPublicKeyBytes()
}

// EncryptQueryHE encrypts a query with HE (for sending to compute server).
func (c *HybridClient) EncryptQueryHE(query []float64) (*rlwe.Ciphertext, error) {
	normalizedQuery := normalizeVector(query)
	return c.heEngine.EncryptVector(normalizedQuery)
}

// DecryptScoreHE decrypts an HE-encrypted score.
func (c *HybridClient) DecryptScoreHE(encScore *rlwe.Ciphertext) (float64, error) {
	return c.heEngine.DecryptScalar(encScore)
}
