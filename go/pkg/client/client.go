// Package client provides the Opaque SDK for privacy-preserving search.
package client

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// Config holds client configuration.
type Config struct {
	// Vector dimension
	Dimension int

	// Number of LSH bits
	LSHBits int

	// Maximum candidates to retrieve from LSH (Stage 1)
	MaxCandidates int

	// Number to score with HE after Hamming ranking (Stage 2)
	HECandidates int

	// Number of final results to return
	TopK int

	// Enable hash masking for privacy
	EnableHashMasking bool
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		Dimension:         128,
		LSHBits:           128,
		MaxCandidates:     200, // Stage 1: Get many candidates (cheap)
		HECandidates:      10,  // Stage 2: Score fewer with HE (expensive)
		TopK:              10,
		EnableHashMasking: true,
	}
}

// Result represents a search result.
type Result struct {
	ID    string
	Score float64
}

// Client is the main SDK entry point for privacy-preserving search.
type Client struct {
	config Config

	// Cryptography
	engine *crypto.Engine

	// LSH
	lshPlanes  [][]float64
	lshBits    int
	sessionKey []byte // For hash masking

	mu sync.RWMutex
}

// NewClient creates a new Opaque client.
func NewClient(cfg Config) (*Client, error) {
	// Create crypto engine
	engine, err := crypto.NewClientEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create crypto engine: %w", err)
	}

	client := &Client{
		config: cfg,
		engine: engine,
	}

	// Generate session key for hash masking
	if cfg.EnableHashMasking {
		client.sessionKey = lsh.GenerateSessionKey(16) // 128-bit key
	}

	return client, nil
}

// RotateSessionKey generates a new session key for hash masking.
// Call this periodically to prevent query correlation.
func (c *Client) RotateSessionKey() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sessionKey = lsh.GenerateSessionKey(16)
}

// GetSessionKey returns the current session key (for server coordination).
func (c *Client) GetSessionKey() []byte {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.sessionKey
}

// GetPublicKey returns the serialized public key for registration.
func (c *Client) GetPublicKey() ([]byte, error) {
	return c.engine.GetPublicKeyBytes()
}

// SetLSHPlanes sets the LSH hyperplanes from the server.
func (c *Client) SetLSHPlanes(planes []float64, numPlanes, dimension int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(planes) != numPlanes*dimension {
		return fmt.Errorf("planes length mismatch: got %d, expected %d", len(planes), numPlanes*dimension)
	}

	c.lshPlanes = make([][]float64, numPlanes)
	for i := 0; i < numPlanes; i++ {
		c.lshPlanes[i] = planes[i*dimension : (i+1)*dimension]
	}
	c.lshBits = numPlanes

	return nil
}

// ComputeLSHHash computes the LSH hash for a query vector.
func (c *Client) ComputeLSHHash(vector []float64) ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.lshPlanes == nil {
		return nil, errors.New("LSH planes not set - call SetLSHPlanes first")
	}

	numBytes := (c.lshBits + 7) / 8
	hash := make([]byte, numBytes)

	for i, plane := range c.lshPlanes {
		dot := dotProduct(vector, plane)
		if dot > 0 {
			hash[i/8] |= (1 << (i % 8))
		}
	}

	return hash, nil
}

// ComputeMaskedLSHHash computes the LSH hash with session key masking.
// The masked hash hides the exact bucket from the server.
func (c *Client) ComputeMaskedLSHHash(vector []float64) ([]byte, error) {
	hash, err := c.ComputeLSHHash(vector)
	if err != nil {
		return nil, err
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.sessionKey != nil && c.config.EnableHashMasking {
		return lsh.MaskHash(hash, c.sessionKey), nil
	}

	return hash, nil
}

// EncryptQuery encrypts a query vector.
func (c *Client) EncryptQuery(vector []float64) ([]byte, error) {
	// Normalize the vector
	normalized := crypto.NormalizeVector(vector)

	// Encrypt
	ct, err := c.engine.EncryptVector(normalized)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt: %w", err)
	}

	// Serialize
	return c.engine.SerializeCiphertext(ct)
}

// DecryptScores decrypts a list of encrypted scores.
func (c *Client) DecryptScores(encryptedScores [][]byte) ([]float64, error) {
	scores := make([]float64, len(encryptedScores))
	errs := make([]error, len(encryptedScores))

	var wg sync.WaitGroup

	for i, encBytes := range encryptedScores {
		wg.Add(1)
		go func(idx int, data []byte) {
			defer wg.Done()

			ct, err := c.engine.DeserializeCiphertext(data)
			if err != nil {
				errs[idx] = err
				return
			}

			score, err := c.engine.DecryptScalar(ct)
			if err != nil {
				errs[idx] = err
				return
			}

			scores[idx] = score
		}(i, encBytes)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return scores, nil
}

// DecryptScoresSequential decrypts scores sequentially (for debugging).
func (c *Client) DecryptScoresSequential(encryptedScores [][]byte) ([]float64, error) {
	scores := make([]float64, len(encryptedScores))

	for i, encBytes := range encryptedScores {
		ct, err := c.engine.DeserializeCiphertext(encBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize score %d: %w", i, err)
		}

		score, err := c.engine.DecryptScalar(ct)
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt score %d: %w", i, err)
		}

		scores[i] = score
	}

	return scores, nil
}

// TopKResults sorts results by score and returns top-k.
func (c *Client) TopKResults(ids []string, scores []float64, k int) []Result {
	if len(ids) != len(scores) {
		return nil
	}

	results := make([]Result, len(ids))
	for i := range ids {
		results[i] = Result{ID: ids[i], Score: scores[i]}
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if k > len(results) {
		k = len(results)
	}

	return results[:k]
}

// SearchLocal performs a complete local search (for testing without server).
// This simulates the full protocol locally.
func (c *Client) SearchLocal(ctx context.Context, query []float64, index *lsh.Index, vectors map[string][]float64, topK int) ([]Result, error) {
	// 1. Compute LSH hash
	queryHash := index.HashBytes(query)

	// 2. Get candidates
	candidates, err := index.Search(queryHash, c.config.MaxCandidates)
	if err != nil {
		return nil, fmt.Errorf("LSH search failed: %w", err)
	}

	// 3. Encrypt query
	normalizedQuery := crypto.NormalizeVector(query)
	encQuery, err := c.engine.EncryptVector(normalizedQuery)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: %w", err)
	}

	// 4. Compute encrypted scores (simulating server)
	// In real scenario, this happens on server
	scores := make([]float64, len(candidates))
	for i, cand := range candidates {
		vec, ok := vectors[cand.ID]
		if !ok {
			continue
		}

		// Normally server does this homomorphically
		// For local testing, we compute plaintext score
		normalizedVec := crypto.NormalizeVector(vec)
		scores[i] = dotProduct(normalizedQuery, normalizedVec)
	}

	// 5. Build results
	results := make([]Result, len(candidates))
	for i, cand := range candidates {
		results[i] = Result{ID: cand.ID, Score: scores[i]}
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// For reference, print that encryption was tested
	_ = encQuery

	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

// TwoStageSearchLocal performs optimized two-stage search locally.
// Stage 1: LSH retrieves many candidates (cheap, ~0.1ms)
// Stage 2: Rank by Hamming distance, HE score only top N (expensive but fewer)
// This provides better accuracy with faster execution.
func (c *Client) TwoStageSearchLocal(ctx context.Context, query []float64, index *lsh.Index, vectors map[string][]float64) ([]Result, error) {
	normalizedQuery := crypto.NormalizeVector(query)

	// Stage 1: Get many candidates from LSH (cheap)
	candidates, err := index.TwoStageSearch(normalizedQuery, c.config.MaxCandidates, c.config.HECandidates)
	if err != nil {
		return nil, fmt.Errorf("LSH search failed: %w", err)
	}

	if len(candidates) == 0 {
		return []Result{}, nil
	}

	// Encrypt query once
	encQuery, err := c.engine.EncryptVector(normalizedQuery)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: %w", err)
	}

	// Stage 2: HE dot products only on top candidates (parallel)
	type scoredResult struct {
		id    string
		score float64
		err   error
	}

	results := make([]scoredResult, len(candidates))
	var wg sync.WaitGroup

	for i, cand := range candidates {
		wg.Add(1)
		go func(idx int, candidate lsh.Candidate) {
			defer wg.Done()

			vec, ok := vectors[candidate.ID]
			if !ok {
				results[idx] = scoredResult{id: candidate.ID, err: fmt.Errorf("vector not found: %s", candidate.ID)}
				return
			}

			normalizedVec := crypto.NormalizeVector(vec)
			encScore, err := c.engine.HomomorphicDotProduct(encQuery, normalizedVec)
			if err != nil {
				results[idx] = scoredResult{id: candidate.ID, err: err}
				return
			}

			score, err := c.engine.DecryptScalar(encScore)
			if err != nil {
				results[idx] = scoredResult{id: candidate.ID, err: err}
				return
			}

			results[idx] = scoredResult{id: candidate.ID, score: score}
		}(i, cand)
	}

	wg.Wait()

	// Collect results, skip errors
	finalResults := make([]Result, 0, len(results))
	for _, r := range results {
		if r.err == nil {
			finalResults = append(finalResults, Result{ID: r.id, Score: r.score})
		}
	}

	// Client-side final ranking (server doesn't see this order)
	sort.Slice(finalResults, func(i, j int) bool {
		return finalResults[i].Score > finalResults[j].Score
	})

	if c.config.TopK > 0 && len(finalResults) > c.config.TopK {
		finalResults = finalResults[:c.config.TopK]
	}

	return finalResults, nil
}

// EncryptAndComputeLocal tests the full homomorphic pipeline locally.
func (c *Client) EncryptAndComputeLocal(query, vector []float64) (float64, float64, error) {
	// Normalize
	normalizedQuery := crypto.NormalizeVector(query)
	normalizedVector := crypto.NormalizeVector(vector)

	// Encrypt query
	encQuery, err := c.engine.EncryptVector(normalizedQuery)
	if err != nil {
		return 0, 0, fmt.Errorf("encryption failed: %w", err)
	}

	// Compute homomorphic dot product
	encScore, err := c.engine.HomomorphicDotProduct(encQuery, normalizedVector)
	if err != nil {
		return 0, 0, fmt.Errorf("homomorphic dot product failed: %w", err)
	}

	// Decrypt
	decryptedScore, err := c.engine.DecryptScalar(encScore)
	if err != nil {
		return 0, 0, fmt.Errorf("decryption failed: %w", err)
	}

	// Compute plaintext score for comparison
	plaintextScore := dotProduct(normalizedQuery, normalizedVector)

	return decryptedScore, plaintextScore, nil
}

// GetEngine returns the crypto engine (for testing).
func (c *Client) GetEngine() *crypto.Engine {
	return c.engine
}

// dotProduct computes the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	var sum float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// cosineSimilarity computes cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	dot := dotProduct(a, b)
	normA := math.Sqrt(dotProduct(a, a))
	normB := math.Sqrt(dotProduct(b, b))
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (normA * normB)
}
