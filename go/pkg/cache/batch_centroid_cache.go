// Package cache provides caching for expensive HE operations.
package cache

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// BatchCentroidCache packs multiple centroids into single plaintexts for SIMD operations.
// This dramatically reduces the number of HE operations by leveraging CKKS slot packing.
//
// Example with 128-dim vectors and 16384 slots:
//   - Pack 128 centroids per plaintext (16384/128 = 128)
//   - For 64 centroids: single plaintext, single HE operation!
//   - Speedup: 64x fewer HE multiplications
//
// Layout in slots: [c0[0..dim], c1[0..dim], c2[0..dim], ...]
// Query packing:   [q[0..dim],  q[0..dim],  q[0..dim],  ...]
// After multiply:  [c0*q[0..dim], c1*q[0..dim], ...]
// After sum:       [dot(c0,q), ?, ?, ..., dot(c1,q), ?, ?, ...]
// Extract at:      positions [0, dim, 2*dim, ...]
type BatchCentroidCache struct {
	// Packed plaintexts - each contains multiple centroids
	packedPlaintexts []*rlwe.Plaintext

	// Configuration
	dimension        int // Vector dimension (e.g., 128)
	centroidsPerPack int // How many centroids fit in one plaintext
	numCentroids     int // Total number of centroids

	// HE parameters
	params  hefloat.Parameters
	encoder *hefloat.Encoder

	mu sync.RWMutex
}

// NewBatchCentroidCache creates a new batch centroid cache.
func NewBatchCentroidCache(params hefloat.Parameters, encoder *hefloat.Encoder, dimension int) *BatchCentroidCache {
	maxSlots := params.MaxSlots()
	centroidsPerPack := maxSlots / dimension

	return &BatchCentroidCache{
		dimension:        dimension,
		centroidsPerPack: centroidsPerPack,
		params:           params,
		encoder:          encoder,
	}
}

// LoadCentroids packs all centroids into batch plaintexts.
// Each plaintext contains centroidsPerPack centroids laid out sequentially.
func (c *BatchCentroidCache) LoadCentroids(centroids [][]float64, level int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(centroids) == 0 {
		return nil
	}

	c.numCentroids = len(centroids)
	numPacks := (len(centroids) + c.centroidsPerPack - 1) / c.centroidsPerPack
	c.packedPlaintexts = make([]*rlwe.Plaintext, numPacks)

	maxSlots := c.params.MaxSlots()

	for packIdx := 0; packIdx < numPacks; packIdx++ {
		// Create packed vector: [c0[0..dim], c1[0..dim], c2[0..dim], ...]
		packedVec := make([]float64, maxSlots)

		for localIdx := 0; localIdx < c.centroidsPerPack; localIdx++ {
			globalIdx := packIdx*c.centroidsPerPack + localIdx
			if globalIdx >= len(centroids) {
				break
			}

			slotOffset := localIdx * c.dimension
			centroid := centroids[globalIdx]

			// Copy centroid to appropriate slot positions
			for j := 0; j < len(centroid) && j < c.dimension; j++ {
				packedVec[slotOffset+j] = centroid[j]
			}
		}

		// Encode as plaintext
		pt := hefloat.NewPlaintext(c.params, level)
		if err := c.encoder.Encode(packedVec, pt); err != nil {
			return fmt.Errorf("failed to encode packed centroids (pack %d): %w", packIdx, err)
		}
		c.packedPlaintexts[packIdx] = pt
	}

	return nil
}

// GetPackedPlaintexts returns all packed plaintexts.
func (c *BatchCentroidCache) GetPackedPlaintexts() []*rlwe.Plaintext {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.packedPlaintexts
}

// GetDimension returns the vector dimension.
func (c *BatchCentroidCache) GetDimension() int {
	return c.dimension
}

// GetCentroidsPerPack returns how many centroids are packed per plaintext.
func (c *BatchCentroidCache) GetCentroidsPerPack() int {
	return c.centroidsPerPack
}

// GetNumCentroids returns the total number of centroids.
func (c *BatchCentroidCache) GetNumCentroids() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.numCentroids
}

// GetNumPacks returns the number of packed plaintexts.
func (c *BatchCentroidCache) GetNumPacks() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.packedPlaintexts)
}

// PackQuery creates a packed query vector matching the centroid layout.
// The query is replicated centroidsPerPack times to enable SIMD multiply.
func (c *BatchCentroidCache) PackQuery(query []float64) []float64 {
	maxSlots := c.params.MaxSlots()
	packed := make([]float64, maxSlots)

	// Replicate query across all centroid positions
	for i := 0; i < c.centroidsPerPack; i++ {
		slotOffset := i * c.dimension
		for j := 0; j < len(query) && j < c.dimension; j++ {
			packed[slotOffset+j] = query[j]
		}
	}

	return packed
}

// Size returns the number of packed plaintexts.
func (c *BatchCentroidCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.packedPlaintexts)
}

// Clear removes all packed plaintexts.
func (c *BatchCentroidCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.packedPlaintexts = nil
	c.numCentroids = 0
}
