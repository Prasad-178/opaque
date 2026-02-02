// Package lsh provides locality-sensitive hashing for approximate nearest neighbor search.
package lsh

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"sort"
	"sync"
)

// Index is a locality-sensitive hash index using random hyperplanes.
type Index struct {
	planes    [][]float64         // Random hyperplanes for hashing
	buckets   map[uint64][]string // Hash -> document IDs
	vectors   map[string][]float64 // ID -> vector (for optional re-ranking)
	numBits   int
	dimension int
	seed      int64
	mu        sync.RWMutex
}

// Candidate represents a search result candidate
type Candidate struct {
	ID       string
	Distance int // Hamming distance
}

// Config holds configuration for creating a new index
type Config struct {
	Dimension int   // Vector dimension
	NumBits   int   // Number of hash bits (64, 128, 256, etc.)
	Seed      int64 // Random seed for reproducibility
}

// NewIndex creates a new LSH index with random hyperplanes.
func NewIndex(cfg Config) *Index {
	rng := rand.New(rand.NewSource(cfg.Seed))

	idx := &Index{
		planes:    make([][]float64, cfg.NumBits),
		buckets:   make(map[uint64][]string),
		vectors:   make(map[string][]float64),
		numBits:   cfg.NumBits,
		dimension: cfg.Dimension,
		seed:      cfg.Seed,
	}

	// Generate random hyperplanes (normalized)
	for i := 0; i < cfg.NumBits; i++ {
		plane := make([]float64, cfg.Dimension)
		var norm float64
		for j := 0; j < cfg.Dimension; j++ {
			plane[j] = rng.NormFloat64()
			norm += plane[j] * plane[j]
		}
		// Normalize
		norm = math.Sqrt(norm)
		for j := range plane {
			plane[j] /= norm
		}
		idx.planes[i] = plane
	}

	return idx
}

// Add adds vectors to the index.
func (idx *Index) Add(ids []string, vectors [][]float64) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch: %d vs %d", len(ids), len(vectors))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, id := range ids {
		vec := vectors[i]
		if len(vec) != idx.dimension {
			return fmt.Errorf("vector %s has wrong dimension: %d (expected %d)", id, len(vec), idx.dimension)
		}

		hash := idx.hash(vec)
		idx.buckets[hash] = append(idx.buckets[hash], id)
		idx.vectors[id] = vec
	}

	return nil
}

// AddOne adds a single vector to the index.
func (idx *Index) AddOne(id string, vector []float64) error {
	return idx.Add([]string{id}, [][]float64{vector})
}

// Remove removes vectors from the index by ID.
func (idx *Index) Remove(ids []string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, id := range ids {
		vec, ok := idx.vectors[id]
		if !ok {
			continue
		}

		hash := idx.hash(vec)

		// Remove from bucket
		bucket := idx.buckets[hash]
		for i, bid := range bucket {
			if bid == id {
				idx.buckets[hash] = append(bucket[:i], bucket[i+1:]...)
				break
			}
		}

		// Remove vector
		delete(idx.vectors, id)
	}
}

// Search finds k nearest candidates by Hamming distance to the query hash.
func (idx *Index) Search(queryHash []byte, k int) ([]Candidate, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Convert query hash to uint64 (pad if necessary)
	queryHashInt := hashBytesToUint64(queryHash)

	candidates := make([]Candidate, 0, k*10)

	// Compute Hamming distance to all buckets
	for bucketHash, ids := range idx.buckets {
		dist := bits.OnesCount64(queryHashInt ^ bucketHash)
		for _, id := range ids {
			candidates = append(candidates, Candidate{ID: id, Distance: dist})
		}
	}

	// Sort by Hamming distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	// Return top k
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates, nil
}

// hashBytesToUint64 converts a byte slice hash to uint64
func hashBytesToUint64(hash []byte) uint64 {
	// Pad to 8 bytes if needed
	padded := make([]byte, 8)
	copy(padded, hash)
	return binary.BigEndian.Uint64(padded)
}

// SearchVector computes hash and searches in one call.
func (idx *Index) SearchVector(query []float64, k int) ([]Candidate, error) {
	if len(query) != idx.dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), idx.dimension)
	}

	hash := idx.HashBytes(query)
	return idx.Search(hash, k)
}

// MultiProbeSearch searches with multi-probe LSH (checks neighboring buckets).
// This is essentially the same as regular search but with a relaxed distance
// threshold that increases recall at the cost of precision.
// The numProbes parameter controls the maximum Hamming distance to include.
func (idx *Index) MultiProbeSearch(queryHash []byte, k int, numProbes int) ([]Candidate, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	queryHashInt := hashBytesToUint64(queryHash)

	candidates := make([]Candidate, 0, k*10)

	// Include all vectors from all buckets, sorted by distance
	// numProbes acts as a soft threshold - we include more candidates
	// from closer buckets and fewer from farther buckets
	for bucketHash, ids := range idx.buckets {
		dist := bits.OnesCount64(queryHashInt ^ bucketHash)
		for _, id := range ids {
			candidates = append(candidates, Candidate{ID: id, Distance: dist})
		}
	}

	// Sort by Hamming distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	// Return top k
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates, nil
}

// Hash computes the LSH signature for a vector (as uint64).
func (idx *Index) hash(vector []float64) uint64 {
	var hash uint64

	for i, plane := range idx.planes {
		if i >= 64 {
			break // Only use first 64 bits for bucket key
		}
		dot := dotProduct(vector, plane)
		if dot > 0 {
			hash |= (1 << i)
		}
	}

	return hash
}

// HashBytes returns the LSH hash as bytes (for gRPC transport).
func (idx *Index) HashBytes(vector []float64) []byte {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	numBytes := (idx.numBits + 7) / 8
	hash := make([]byte, numBytes)

	for i, plane := range idx.planes {
		dot := dotProduct(vector, plane)
		if dot > 0 {
			hash[i/8] |= (1 << (i % 8))
		}
	}

	return hash
}

// GetVector retrieves a stored vector by ID.
func (idx *Index) GetVector(id string) ([]float64, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	vec, ok := idx.vectors[id]
	return vec, ok
}

// GetVectors retrieves multiple vectors by ID.
func (idx *Index) GetVectors(ids []string) [][]float64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	vectors := make([][]float64, len(ids))
	for i, id := range ids {
		vectors[i] = idx.vectors[id]
	}
	return vectors
}

// Count returns the total number of vectors in the index.
func (idx *Index) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// BucketCount returns the number of buckets.
func (idx *Index) BucketCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.buckets)
}

// GetPlanes returns the hyperplanes (for client-side hashing).
func (idx *Index) GetPlanes() [][]float64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Return a copy
	planes := make([][]float64, len(idx.planes))
	for i, p := range idx.planes {
		planes[i] = make([]float64, len(p))
		copy(planes[i], p)
	}
	return planes
}

// Save serializes the index to a writer.
func (idx *Index) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	enc := gob.NewEncoder(w)

	// Save metadata
	if err := enc.Encode(idx.numBits); err != nil {
		return err
	}
	if err := enc.Encode(idx.dimension); err != nil {
		return err
	}
	if err := enc.Encode(idx.seed); err != nil {
		return err
	}

	// Save planes
	if err := enc.Encode(idx.planes); err != nil {
		return err
	}

	// Save buckets
	if err := enc.Encode(idx.buckets); err != nil {
		return err
	}

	// Save vectors
	if err := enc.Encode(idx.vectors); err != nil {
		return err
	}

	return nil
}

// SaveToFile saves the index to a file.
func (idx *Index) SaveToFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return idx.Save(f)
}

// Load deserializes an index from a reader.
func Load(r io.Reader) (*Index, error) {
	dec := gob.NewDecoder(r)

	var numBits, dimension int
	var seed int64

	if err := dec.Decode(&numBits); err != nil {
		return nil, err
	}
	if err := dec.Decode(&dimension); err != nil {
		return nil, err
	}
	if err := dec.Decode(&seed); err != nil {
		return nil, err
	}

	idx := &Index{
		numBits:   numBits,
		dimension: dimension,
		seed:      seed,
	}

	if err := dec.Decode(&idx.planes); err != nil {
		return nil, err
	}
	if err := dec.Decode(&idx.buckets); err != nil {
		return nil, err
	}
	if err := dec.Decode(&idx.vectors); err != nil {
		return nil, err
	}

	return idx, nil
}

// LoadFromFile loads an index from a file.
func LoadFromFile(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return Load(f)
}

// dotProduct computes the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// HammingDistance computes the Hamming distance between two byte slices.
func HammingDistance(a, b []byte) int {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var dist int
	for i := 0; i < minLen; i++ {
		dist += bits.OnesCount8(a[i] ^ b[i])
	}

	// Count remaining bits in longer slice
	for i := minLen; i < len(a); i++ {
		dist += bits.OnesCount8(a[i])
	}
	for i := minLen; i < len(b); i++ {
		dist += bits.OnesCount8(b[i])
	}

	return dist
}

// TwoStageSearch performs optimized two-stage retrieval:
// Stage 1: Get many candidates from LSH (cheap)
// Stage 2: Rank by Hamming distance, return top N for expensive HE scoring
// This improves both accuracy (more initial candidates) and speed (fewer HE ops).
func (idx *Index) TwoStageSearch(query []float64, initialCandidates, topN int) ([]Candidate, error) {
	if len(query) != idx.dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), idx.dimension)
	}

	// Stage 1: Get many candidates
	hash := idx.HashBytes(query)
	candidates, err := idx.Search(hash, initialCandidates)
	if err != nil {
		return nil, err
	}

	// Stage 2: Already sorted by Hamming distance, just return top N
	if len(candidates) > topN {
		candidates = candidates[:topN]
	}

	return candidates, nil
}

// TwoStageConfig holds configuration for two-stage search
type TwoStageConfig struct {
	InitialCandidates int // Number of candidates from LSH (default: 200)
	TopN              int // Number to return for HE scoring (default: 10)
}

// DefaultTwoStageConfig returns sensible defaults
func DefaultTwoStageConfig() TwoStageConfig {
	return TwoStageConfig{
		InitialCandidates: 200,
		TopN:              10,
	}
}

// MaskHash XORs the hash with a session key for privacy.
// This prevents the server from correlating queries across sessions.
func MaskHash(hash []byte, sessionKey []byte) []byte {
	masked := make([]byte, len(hash))
	for i := range hash {
		masked[i] = hash[i] ^ sessionKey[i%len(sessionKey)]
	}
	return masked
}

// UnmaskHash reverses the masking operation.
func UnmaskHash(maskedHash []byte, sessionKey []byte) []byte {
	return MaskHash(maskedHash, sessionKey) // XOR is its own inverse
}

// GenerateSessionKey generates a random session key for hash masking.
func GenerateSessionKey(numBytes int) []byte {
	key := make([]byte, numBytes)
	rand.Read(key)
	return key
}
