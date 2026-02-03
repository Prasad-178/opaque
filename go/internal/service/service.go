// Package service implements the Opaque search service.
package service

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/schemes/bfv"

	"github.com/opaque/opaque/go/internal/session"
	"github.com/opaque/opaque/go/internal/store"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// Config holds service configuration.
type Config struct {
	// LSH configuration
	LSHNumBits   int
	LSHDimension int
	LSHSeed      int64

	// Session configuration
	MaxSessionTTL time.Duration

	// Concurrency limits
	MaxConcurrentScores int
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		LSHNumBits:          128,
		LSHDimension:        128,
		LSHSeed:             42,
		MaxSessionTTL:       24 * time.Hour,
		MaxConcurrentScores: 16,
	}
}

// SearchService implements the Opaque search service.
type SearchService struct {
	config   Config
	lshIndex *lsh.Index
	store    store.VectorStore
	sessions *session.Manager

	// BFV parameters (shared, no secret key!)
	params  bfv.Parameters
	encoder *bfv.Encoder

	mu sync.RWMutex
}

// NewSearchService creates a new search service.
func NewSearchService(cfg Config, vectorStore store.VectorStore) (*SearchService, error) {
	// Initialize BFV parameters (must match client)
	params, err := bfv.NewParametersFromLiteral(bfv.ExampleParameters128BitLogN14LogQP438)
	if err != nil {
		return nil, fmt.Errorf("failed to create BFV parameters: %w", err)
	}

	// Create LSH index
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: cfg.LSHDimension,
		NumBits:   cfg.LSHNumBits,
		Seed:      cfg.LSHSeed,
	})

	return &SearchService{
		config:   cfg,
		lshIndex: lshIndex,
		store:    vectorStore,
		sessions: session.NewManager(cfg.MaxSessionTTL),
		params:   params,
		encoder:  bfv.NewEncoder(params),
	}, nil
}

// RegisterKey registers a client's public key and creates a session.
func (s *SearchService) RegisterKey(ctx context.Context, publicKey []byte, ttlSeconds int32) (string, int32, error) {
	ttl := time.Duration(ttlSeconds) * time.Second
	if ttl > s.config.MaxSessionTTL {
		ttl = s.config.MaxSessionTTL
	}

	sess, err := s.sessions.Create(publicKey, ttl)
	if err != nil {
		return "", 0, fmt.Errorf("failed to create session: %w", err)
	}

	actualTTL := int32(sess.ExpiresAt.Sub(sess.CreatedAt).Seconds())
	return sess.ID, actualTTL, nil
}

// GetPlanes returns the LSH hyperplanes for client-side hashing.
func (s *SearchService) GetPlanes(ctx context.Context, sessionID string) ([]float64, int32, int32, error) {
	// Validate session
	if _, err := s.sessions.Get(sessionID); err != nil {
		return nil, 0, 0, fmt.Errorf("invalid session: %w", err)
	}

	planes := s.lshIndex.GetPlanes()

	// Flatten planes
	flat := make([]float64, 0, len(planes)*len(planes[0]))
	for _, plane := range planes {
		flat = append(flat, plane...)
	}

	return flat, int32(len(planes)), int32(s.config.LSHDimension), nil
}

// GetCandidates performs LSH lookup to find candidate vectors.
func (s *SearchService) GetCandidates(ctx context.Context, sessionID string, lshHash []byte, numCandidates int32, multiProbe bool, numProbes int32) ([]string, []int32, error) {
	// Validate session
	if _, err := s.sessions.Get(sessionID); err != nil {
		return nil, nil, fmt.Errorf("invalid session: %w", err)
	}

	var candidates []lsh.Candidate
	var err error

	if multiProbe && numProbes > 1 {
		candidates, err = s.lshIndex.MultiProbeSearch(lshHash, int(numCandidates), int(numProbes))
	} else {
		candidates, err = s.lshIndex.Search(lshHash, int(numCandidates))
	}

	if err != nil {
		return nil, nil, fmt.Errorf("LSH search failed: %w", err)
	}

	ids := make([]string, len(candidates))
	distances := make([]int32, len(candidates))
	for i, c := range candidates {
		ids[i] = c.ID
		distances[i] = int32(c.Distance)
	}

	return ids, distances, nil
}

// ComputeScores computes encrypted similarity scores for candidates.
func (s *SearchService) ComputeScores(ctx context.Context, sessionID string, encryptedQuery []byte, candidateIDs []string) ([][]byte, []string, error) {
	// Validate session and get public key
	sess, err := s.sessions.Get(sessionID)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid session: %w", err)
	}

	// Deserialize encrypted query
	encQuery := rlwe.NewCiphertext(s.params, 1, s.params.MaxLevel())
	if _, err := encQuery.ReadFrom(bytes.NewReader(encryptedQuery)); err != nil {
		return nil, nil, fmt.Errorf("failed to deserialize encrypted query: %w", err)
	}

	// Create evaluator with rotation keys from public key
	// Note: For proper rotation support, we'd need evaluation keys from the client
	// For now, use a basic evaluator
	evaluator := bfv.NewEvaluator(s.params, nil)

	// Fetch vectors from store
	vectors, err := s.store.GetByIDs(ctx, candidateIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to fetch vectors: %w", err)
	}

	// Compute scores in parallel
	encScores := make([][]byte, len(vectors))
	errs := make([]error, len(vectors))

	sem := make(chan struct{}, s.config.MaxConcurrentScores)
	var wg sync.WaitGroup

	for i, vec := range vectors {
		wg.Add(1)
		sem <- struct{}{}

		go func(idx int, vector []float64) {
			defer wg.Done()
			defer func() { <-sem }()

			// Compute homomorphic dot product
			score, err := s.homomorphicDotProduct(evaluator, encQuery, vector)
			if err != nil {
				errs[idx] = err
				return
			}

			// Serialize result
			buf := new(bytes.Buffer)
			if _, err := score.WriteTo(buf); err != nil {
				errs[idx] = err
				return
			}
			encScores[idx] = buf.Bytes()
		}(i, vec)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errs {
		if err != nil {
			return nil, nil, err
		}
	}

	// Needed to avoid unused variable error for sess
	_ = sess

	return encScores, candidateIDs, nil
}

// homomorphicDotProduct computes E(q · v) from E(q) and plaintext v.
func (s *SearchService) homomorphicDotProduct(evaluator *bfv.Evaluator, encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error) {
	// Scale vector to fixed-point representation
	// Must match the constants in pkg/crypto/crypto.go
	const scaleFactor = 30000.0
	const offset = 1.0

	scaled := make([]uint64, len(vector))
	for i, v := range vector {
		// Add offset and scale (matching client encoding)
		scaled[i] = uint64((v + offset) * scaleFactor)
	}

	// Pad to power of 2 if needed
	n := 1
	for n < len(scaled) {
		n *= 2
	}
	if len(scaled) < n {
		padded := make([]uint64, n)
		copy(padded, scaled)
		scaled = padded
	}

	// Encode as plaintext
	pt := bfv.NewPlaintext(s.params, encQuery.Level())
	if err := s.encoder.Encode(scaled, pt); err != nil {
		return nil, fmt.Errorf("failed to encode vector: %w", err)
	}

	// Multiply: E(q) * v = E(q * v) component-wise
	result, err := evaluator.MulNew(encQuery, pt)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}

	// Note: Summation via rotations requires evaluation keys
	// Without them, we return the component-wise product
	// The client will need to sum after decryption, or we need
	// to implement a key exchange for evaluation keys

	return result, nil
}

// Search performs the complete search flow.
func (s *SearchService) Search(ctx context.Context, sessionID string, lshHash []byte, encryptedQuery []byte, maxCandidates, topN int32) ([][]byte, []string, []int32, error) {
	// Get candidates
	candidateIDs, distances, err := s.GetCandidates(ctx, sessionID, lshHash, maxCandidates, true, 5)
	if err != nil {
		return nil, nil, nil, err
	}

	// Limit to top N
	if int(topN) < len(candidateIDs) {
		candidateIDs = candidateIDs[:topN]
		distances = distances[:topN]
	}

	// Compute scores
	encScores, ids, err := s.ComputeScores(ctx, sessionID, encryptedQuery, candidateIDs)
	if err != nil {
		return nil, nil, nil, err
	}

	return encScores, ids, distances, nil
}

// AddVectors adds vectors to the index and store.
func (s *SearchService) AddVectors(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error {
	// Add to store
	if err := s.store.Add(ctx, ids, vectors, metadata); err != nil {
		return fmt.Errorf("failed to add to store: %w", err)
	}

	// Add to LSH index
	if err := s.lshIndex.Add(ids, vectors); err != nil {
		return fmt.Errorf("failed to add to LSH index: %w", err)
	}

	return nil
}

// RemoveVectors removes vectors from the index and store.
func (s *SearchService) RemoveVectors(ctx context.Context, ids []string) error {
	// Remove from store
	if err := s.store.Delete(ctx, ids); err != nil {
		return fmt.Errorf("failed to delete from store: %w", err)
	}

	// Remove from LSH index
	s.lshIndex.Remove(ids)

	return nil
}

// GetVectorCount returns the number of vectors in the index.
func (s *SearchService) GetVectorCount(ctx context.Context) (int64, error) {
	return s.store.Count(ctx)
}

// GetSessionCount returns the number of active sessions.
func (s *SearchService) GetSessionCount() int {
	return s.sessions.Count()
}

// HealthCheck returns service health status.
func (s *SearchService) HealthCheck(ctx context.Context) (bool, string, int64, int64) {
	count, err := s.store.Count(ctx)
	if err != nil {
		return false, fmt.Sprintf("store error: %v", err), 0, 0
	}

	return true, "healthy", int64(s.sessions.Count()), count
}

// ValidateSession checks if a session is valid.
func (s *SearchService) ValidateSession(sessionID string) error {
	_, err := s.sessions.Get(sessionID)
	return err
}

// GetLSHIndex returns the LSH index (for testing).
func (s *SearchService) GetLSHIndex() *lsh.Index {
	return s.lshIndex
}

// Errors
var (
	ErrInvalidSession = errors.New("invalid session")
	ErrInvalidQuery   = errors.New("invalid encrypted query")
)
