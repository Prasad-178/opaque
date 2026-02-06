package hierarchical

import (
	"context"
	"fmt"
	"math"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// Builder constructs a hierarchical index from a batch of vectors.
type Builder struct {
	config    Config
	lshSuper  *lsh.Index // For super-bucket assignment
	lshSub    *lsh.Index // For sub-bucket assignment
	encryptor *encrypt.AESGCM

	// Building state
	superBuckets []*SuperBucket
	vectorLocs   map[string]*VectorLocation
	subBucketCounts map[string]int // bucketKey -> vector count
}

// NewBuilder creates a new index builder.
func NewBuilder(cfg Config, aesKey []byte) (*Builder, error) {
	if cfg.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if cfg.NumSuperBuckets <= 0 {
		cfg.NumSuperBuckets = 64
	}
	if cfg.NumSubBuckets <= 0 {
		cfg.NumSubBuckets = 64
	}

	encryptor, err := encrypt.NewAESGCM(aesKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create encryptor: %w", err)
	}

	// Calculate number of bits needed for super-bucket LSH
	// We need enough bits to have good distribution across NumSuperBuckets
	superBits := int(math.Ceil(math.Log2(float64(cfg.NumSuperBuckets)))) + 2 // Extra bits for better distribution
	if superBits < 6 {
		superBits = 6
	}

	subBits := int(math.Ceil(math.Log2(float64(cfg.NumSubBuckets)))) + 2
	if subBits < 6 {
		subBits = 6
	}

	lshSuper := lsh.NewIndex(lsh.Config{
		Dimension: cfg.Dimension,
		NumBits:   superBits,
		Seed:      cfg.LSHSuperSeed,
	})

	lshSub := lsh.NewIndex(lsh.Config{
		Dimension: cfg.Dimension,
		NumBits:   subBits,
		Seed:      cfg.LSHSubSeed,
	})

	// Initialize super-buckets
	superBuckets := make([]*SuperBucket, cfg.NumSuperBuckets)
	for i := 0; i < cfg.NumSuperBuckets; i++ {
		superBuckets[i] = &SuperBucket{
			ID:          i,
			Centroid:    make([]float64, cfg.Dimension),
			VectorCount: 0,
			sum:         make([]float64, cfg.Dimension),
		}
	}

	return &Builder{
		config:          cfg,
		lshSuper:        lshSuper,
		lshSub:          lshSub,
		encryptor:       encryptor,
		superBuckets:    superBuckets,
		vectorLocs:      make(map[string]*VectorLocation),
		subBucketCounts: make(map[string]int),
	}, nil
}

// Build constructs the hierarchical index from vectors and stores encrypted blobs.
func (b *Builder) Build(ctx context.Context, ids []string, vectors [][]float64, store blob.Store) (*Index, error) {
	if len(ids) != len(vectors) {
		return nil, fmt.Errorf("ids and vectors length mismatch")
	}

	// Phase 1: Assign vectors to super-buckets and sub-buckets
	blobs := make([]*blob.Blob, len(ids))
	for i, id := range ids {
		vec := vectors[i]

		if len(vec) != b.config.Dimension {
			return nil, fmt.Errorf("vector %s has wrong dimension: %d (expected %d)", id, len(vec), b.config.Dimension)
		}

		// Determine super-bucket
		superID := b.lshSuper.HashToIndexFromVector(vec, b.config.NumSuperBuckets)

		// Determine sub-bucket within super
		subID := b.lshSub.HashToIndexFromVector(vec, b.config.NumSubBuckets)

		// Create bucket key
		bucketKey := formatBucketKey(superID, subID)

		// Update super-bucket centroid (running sum)
		super := b.superBuckets[superID]
		super.VectorCount++
		for j, v := range vec {
			super.sum[j] += v
		}

		// Track sub-bucket count
		b.subBucketCounts[bucketKey]++

		// Track vector location
		b.vectorLocs[id] = &VectorLocation{
			ID:        id,
			SuperID:   superID,
			SubID:     subID,
			BucketKey: bucketKey,
		}

		// Encrypt vector
		ciphertext, err := b.encryptor.EncryptVectorWithID(vec, id)
		if err != nil {
			return nil, fmt.Errorf("failed to encrypt vector %s: %w", id, err)
		}

		blobs[i] = blob.NewBlob(id, bucketKey, ciphertext, b.config.Dimension)
	}

	// Phase 2: Compute centroids (average of vectors in each super-bucket)
	for _, super := range b.superBuckets {
		if super.VectorCount > 0 {
			for j := range super.Centroid {
				super.Centroid[j] = super.sum[j] / float64(super.VectorCount)
			}
			// Normalize centroid for cosine similarity
			normalizeVector(super.Centroid)
		}
	}

	// Phase 3: Store all blobs
	if err := store.PutBatch(ctx, blobs); err != nil {
		return nil, fmt.Errorf("failed to store blobs: %w", err)
	}

	// Build index structure
	idx := &Index{
		Config:          b.config,
		SuperBuckets:    b.superBuckets,
		LSHSuper:        b.lshSuper,
		LSHSub:          b.lshSub,
		Store:           store,
		Encryptor:       b.encryptor,
		VectorLocations: b.vectorLocs,
		SubBucketCounts: b.subBucketCounts,
	}

	// Precompute centroids slice for efficient HE ops
	idx.Centroids = make([][]float64, b.config.NumSuperBuckets)
	for i, super := range b.superBuckets {
		idx.Centroids[i] = super.Centroid
	}

	return idx, nil
}

// formatBucketKey creates a bucket key from super and sub IDs.
// Format: "{super:02d}_{sub:02d}" e.g., "07_23"
func formatBucketKey(superID, subID int) string {
	return fmt.Sprintf("%02d_%02d", superID, subID)
}

// parseBucketKey parses a bucket key into super and sub IDs.
func parseBucketKey(key string) (superID, subID int, err error) {
	_, err = fmt.Sscanf(key, "%d_%d", &superID, &subID)
	return
}

// normalizeVector normalizes a vector in place to unit length.
func normalizeVector(v []float64) {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

// dotProduct computes the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
