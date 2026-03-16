// Package cache provides caching for expensive HE operations.
package cache

import (
	"fmt"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// CentroidCache caches pre-encoded centroids for faster HE operations.
// Encoding centroids as HE plaintexts is expensive, so we cache them.
//
// Cache invalidation:
//   - Centroids rarely change (only when data is added/removed)
//   - Track last update time to detect staleness
//   - Caller can force refresh when needed
type CentroidCache struct {
	// Pre-encoded centroids as HE plaintexts
	// Key: centroid index, Value: encoded plaintext
	encoded map[int]*rlwe.Plaintext

	// Original centroid vectors for staleness check
	originals map[int][]float64

	// Metadata
	lastUpdate time.Time
	version    int64 // Incremented on each update

	// HE encoder (borrowed from Engine)
	params  hefloat.Parameters
	encoder *hefloat.Encoder

	mu sync.RWMutex
}

// CentroidCacheConfig contains configuration for the cache.
type CentroidCacheConfig struct {
	// MaxStaleDuration is how long cached centroids are considered valid.
	// Default: 1 hour
	MaxStaleDuration time.Duration
}

// DefaultCentroidCacheConfig returns sensible defaults.
func DefaultCentroidCacheConfig() CentroidCacheConfig {
	return CentroidCacheConfig{
		MaxStaleDuration: 1 * time.Hour,
	}
}

// NewCentroidCache creates a new centroid cache.
// The params and encoder should be from the HE Engine.
func NewCentroidCache(params hefloat.Parameters, encoder *hefloat.Encoder) *CentroidCache {
	return &CentroidCache{
		encoded:    make(map[int]*rlwe.Plaintext),
		originals:  make(map[int][]float64),
		lastUpdate: time.Now(),
		version:    0,
		params:     params,
		encoder:    encoder,
	}
}

// LoadCentroids pre-encodes all centroids into the cache.
// This should be called once when the client starts or when centroids change.
func (c *CentroidCache) LoadCentroids(centroids [][]float64, level int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	maxSlots := c.params.MaxSlots()

	// Clear existing cache
	c.encoded = make(map[int]*rlwe.Plaintext)
	c.originals = make(map[int][]float64)

	for i, centroid := range centroids {
		// Pad vector to max slots with 0
		paddedVector := make([]float64, maxSlots)
		copy(paddedVector, centroid)

		// Encode as plaintext
		pt := hefloat.NewPlaintext(c.params, level)
		if err := c.encoder.Encode(paddedVector, pt); err != nil {
			return fmt.Errorf("failed to encode centroid %d: %w", i, err)
		}

		c.encoded[i] = pt

		// Store original for staleness check
		origCopy := make([]float64, len(centroid))
		copy(origCopy, centroid)
		c.originals[i] = origCopy
	}

	c.lastUpdate = time.Now()
	c.version++

	return nil
}

// Get returns a pre-encoded centroid, or nil if not cached.
func (c *CentroidCache) Get(index int) *rlwe.Plaintext {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.encoded[index]
}

// GetAll returns all cached centroids as a slice.
// Returns nil entries for missing indices.
func (c *CentroidCache) GetAll(count int) []*rlwe.Plaintext {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]*rlwe.Plaintext, count)
	for i := 0; i < count; i++ {
		result[i] = c.encoded[i]
	}
	return result
}

// IsStale checks if the cache is older than maxAge.
func (c *CentroidCache) IsStale(maxAge time.Duration) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return time.Since(c.lastUpdate) > maxAge
}

// NeedsRefresh checks if any centroid has changed from the cached original.
func (c *CentroidCache) NeedsRefresh(centroids [][]float64) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(centroids) != len(c.originals) {
		return true
	}

	for i, centroid := range centroids {
		orig, exists := c.originals[i]
		if !exists {
			return true
		}
		if len(centroid) != len(orig) {
			return true
		}
		// Check if any value differs significantly
		for j := range centroid {
			if abs(centroid[j]-orig[j]) > 1e-9 {
				return true
			}
		}
	}

	return false
}

// Version returns the current cache version (incremented on each load).
func (c *CentroidCache) Version() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.version
}

// LastUpdate returns when the cache was last updated.
func (c *CentroidCache) LastUpdate() time.Time {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lastUpdate
}

// Size returns the number of cached centroids.
func (c *CentroidCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.encoded)
}

// Clear removes all cached centroids.
func (c *CentroidCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.encoded = make(map[int]*rlwe.Plaintext)
	c.originals = make(map[int][]float64)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
