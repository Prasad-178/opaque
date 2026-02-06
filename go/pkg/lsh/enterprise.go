package lsh

import (
	"math"
	"math/rand"
)

// EnterpriseHasher provides LSH hashing using pre-distributed hyperplanes.
// This is used by clients who receive hyperplanes from the auth service
// instead of generating them from a seed.
//
// Key security property: The auth service distributes pre-computed hyperplanes
// instead of seeds. This provides an additional layer of security - even if
// hyperplanes leak, the original seed remains secret.
type EnterpriseHasher struct {
	planes    [][]float64
	dimension int
}

// NewEnterpriseHasher creates a hasher from pre-generated hyperplanes.
func NewEnterpriseHasher(planes [][]float64) *EnterpriseHasher {
	dimension := 0
	if len(planes) > 0 {
		dimension = len(planes[0])
	}
	return &EnterpriseHasher{
		planes:    planes,
		dimension: dimension,
	}
}

// Hash computes the LSH hash for a vector.
func (h *EnterpriseHasher) Hash(vector []float64) []byte {
	numBytes := (len(h.planes) + 7) / 8
	hash := make([]byte, numBytes)

	for i, plane := range h.planes {
		dot := dotProduct(vector, plane)
		if dot > 0 {
			hash[i/8] |= (1 << (i % 8))
		}
	}

	return hash
}

// HashToIndex computes a bucket index in range [0, numBuckets).
func (h *EnterpriseHasher) HashToIndex(vector []float64, numBuckets int) int {
	hash := h.Hash(vector)
	return HashToIndex(hash, numBuckets)
}

// GetPlanes returns the hyperplanes (for serialization).
func (h *EnterpriseHasher) GetPlanes() [][]float64 {
	// Return a copy to prevent mutation
	planes := make([][]float64, len(h.planes))
	for i, p := range h.planes {
		planes[i] = make([]float64, len(p))
		copy(planes[i], p)
	}
	return planes
}

// Dimension returns the expected vector dimension.
func (h *EnterpriseHasher) Dimension() int {
	return h.dimension
}

// NumBits returns the number of hash bits (hyperplanes).
func (h *EnterpriseHasher) NumBits() int {
	return len(h.planes)
}

// GenerateHyperplanes generates random normalized hyperplanes from a seed.
// This should only be called by the enterprise admin during setup,
// NOT by clients during search.
//
// The hyperplanes are normalized to unit length to ensure consistent
// hashing behavior.
func GenerateHyperplanes(seed int64, numBits, dimension int) [][]float64 {
	rng := rand.New(rand.NewSource(seed))

	planes := make([][]float64, numBits)
	for i := 0; i < numBits; i++ {
		plane := make([]float64, dimension)
		var norm float64
		for j := 0; j < dimension; j++ {
			plane[j] = rng.NormFloat64()
			norm += plane[j] * plane[j]
		}
		// Normalize to unit length
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range plane {
				plane[j] /= norm
			}
		}
		planes[i] = plane
	}
	return planes
}

// GenerateHyperplanesFromBytes generates hyperplanes from a byte seed.
// Useful when the seed comes from a cryptographically random source.
func GenerateHyperplanesFromBytes(seed []byte, numBits, dimension int) [][]float64 {
	// Convert first 8 bytes to int64
	var seedInt int64
	for i := 0; i < 8 && i < len(seed); i++ {
		seedInt |= int64(seed[i]) << (uint(i) * 8)
	}
	return GenerateHyperplanes(seedInt, numBits, dimension)
}
