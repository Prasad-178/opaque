package hierarchical

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// EnterpriseBuilder constructs a hierarchical index using enterprise configuration.
// Unlike the standard Builder which uses explicit AES keys and LSH seeds,
// this builder derives everything from the enterprise config, ensuring
// per-enterprise isolation and secret management.
type EnterpriseBuilder struct {
	config        Config
	enterpriseCfg *enterprise.Config
	lshSuper      *lsh.Index
	lshSub        *lsh.Index
	encryptor     *encrypt.AESGCM

	// Building state
	superBuckets    []*SuperBucket
	vectorLocs      map[string]*VectorLocation
	subBucketCounts map[string]int
}

// NewEnterpriseBuilder creates a new builder using enterprise configuration.
// The enterprise config provides:
//   - AES key for vector encryption
//   - LSH seed for bucket assignment (kept secret, per-enterprise)
//   - Dimension and bucket configuration
func NewEnterpriseBuilder(cfg Config, enterpriseCfg *enterprise.Config) (*EnterpriseBuilder, error) {
	if err := enterpriseCfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid enterprise config: %w", err)
	}

	// Override config from enterprise config if not set
	if cfg.Dimension <= 0 {
		cfg.Dimension = enterpriseCfg.Dimension
	}
	if cfg.NumSuperBuckets <= 0 {
		cfg.NumSuperBuckets = enterpriseCfg.NumSuperBuckets
	}
	// Use enterprise config's NumSubBuckets to ensure consistency
	numSubBuckets := enterpriseCfg.NumSubBuckets
	if numSubBuckets <= 0 {
		numSubBuckets = 64 // Default
	}
	cfg.NumSubBuckets = numSubBuckets

	encryptor, err := encrypt.NewAESGCM(enterpriseCfg.AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create encryptor: %w", err)
	}

	// Calculate number of bits needed for LSH
	superBits := int(math.Ceil(math.Log2(float64(cfg.NumSuperBuckets)))) + 2
	if superBits < 6 {
		superBits = 6
	}
	subBits := int(math.Ceil(math.Log2(float64(cfg.NumSubBuckets)))) + 2
	if subBits < 6 {
		subBits = 6
	}

	// Use enterprise secret seeds instead of public seeds
	superSeed := enterpriseCfg.GetLSHSeedAsInt64()
	subSeed := enterpriseCfg.GetSubLSHSeedAsInt64()

	lshSuper := lsh.NewIndex(lsh.Config{
		Dimension: cfg.Dimension,
		NumBits:   superBits,
		Seed:      superSeed,
	})

	lshSub := lsh.NewIndex(lsh.Config{
		Dimension: cfg.Dimension,
		NumBits:   subBits,
		Seed:      subSeed,
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

	return &EnterpriseBuilder{
		config:          cfg,
		enterpriseCfg:   enterpriseCfg,
		lshSuper:        lshSuper,
		lshSub:          lshSub,
		encryptor:       encryptor,
		superBuckets:    superBuckets,
		vectorLocs:      make(map[string]*VectorLocation),
		subBucketCounts: make(map[string]int),
	}, nil
}

// Build constructs the hierarchical index and updates the enterprise config with centroids.
// After building, call GetEnterpriseConfig() to get the updated config with computed centroids.
func (b *EnterpriseBuilder) Build(ctx context.Context, ids []string, vectors [][]float64, store blob.Store) (*Index, error) {
	if len(ids) != len(vectors) {
		return nil, fmt.Errorf("ids and vectors length mismatch: %d vs %d", len(ids), len(vectors))
	}

	// Phase 1: Assign vectors to buckets and encrypt
	blobs := make([]*blob.Blob, len(ids))
	for i, id := range ids {
		vec := vectors[i]

		if len(vec) != b.config.Dimension {
			return nil, fmt.Errorf("vector %s has wrong dimension: %d (expected %d)",
				id, len(vec), b.config.Dimension)
		}

		// Determine bucket assignment using enterprise LSH
		superID := b.lshSuper.HashToIndexFromVector(vec, b.config.NumSuperBuckets)
		subID := b.lshSub.HashToIndexFromVector(vec, b.config.NumSubBuckets)
		bucketKey := formatBucketKey(superID, subID)

		// Update centroid running sum
		super := b.superBuckets[superID]
		super.VectorCount++
		for j, v := range vec {
			super.sum[j] += v
		}

		// Track counts and locations
		b.subBucketCounts[bucketKey]++
		b.vectorLocs[id] = &VectorLocation{
			ID:        id,
			SuperID:   superID,
			SubID:     subID,
			BucketKey: bucketKey,
		}

		// Encrypt with enterprise AES key
		ciphertext, err := b.encryptor.EncryptVectorWithID(vec, id)
		if err != nil {
			return nil, fmt.Errorf("failed to encrypt vector %s: %w", id, err)
		}

		blobs[i] = blob.NewBlob(id, bucketKey, ciphertext, b.config.Dimension)
	}

	// Phase 2: Compute and normalize centroids
	for _, super := range b.superBuckets {
		if super.VectorCount > 0 {
			for j := range super.Centroid {
				super.Centroid[j] = super.sum[j] / float64(super.VectorCount)
			}
			normalizeVector(super.Centroid)
		}
	}

	// Extract centroids for enterprise config and index
	centroids := make([][]float64, b.config.NumSuperBuckets)
	for i, super := range b.superBuckets {
		centroids[i] = make([]float64, len(super.Centroid))
		copy(centroids[i], super.Centroid)
	}

	// Update enterprise config with computed centroids
	b.enterpriseCfg.SetCentroids(centroids)

	// Phase 3: Store all encrypted blobs
	if err := store.PutBatch(ctx, blobs); err != nil {
		return nil, fmt.Errorf("failed to store blobs: %w", err)
	}

	// Build index structure
	idx := &Index{
		Config:          b.config,
		SuperBuckets:    b.superBuckets,
		Centroids:       centroids,
		LSHSuper:        b.lshSuper,
		LSHSub:          b.lshSub,
		Store:           store,
		Encryptor:       b.encryptor,
		VectorLocations: b.vectorLocs,
		SubBucketCounts: b.subBucketCounts,
	}

	return idx, nil
}

// GetEnterpriseConfig returns the enterprise configuration with updated centroids.
// Call this after Build() to get the config that should be stored and distributed.
func (b *EnterpriseBuilder) GetEnterpriseConfig() *enterprise.Config {
	return b.enterpriseCfg
}

// ConfigFromEnterprise creates a hierarchical Config from enterprise settings.
// This is a helper for creating Config with enterprise defaults.
func ConfigFromEnterprise(enterpriseCfg *enterprise.Config) Config {
	numSubBuckets := enterpriseCfg.NumSubBuckets
	if numSubBuckets <= 0 {
		numSubBuckets = 64 // Default
	}
	return Config{
		Dimension:          enterpriseCfg.Dimension,
		NumSuperBuckets:    enterpriseCfg.NumSuperBuckets,
		NumSubBuckets:      numSubBuckets,
		TopSuperBuckets:    8, // Default to top 8
		SubBucketsPerSuper: 2,
		NumDecoys:          8,
		LSHSuperSeed:       enterpriseCfg.GetLSHSeedAsInt64(),
		LSHSubSeed:         enterpriseCfg.GetSubLSHSeedAsInt64(),
	}
}

// BuildEnterpriseIndex is a convenience function that creates an enterprise config,
// builds the index, and returns both the index and the config for storage.
func BuildEnterpriseIndex(
	ctx context.Context,
	enterpriseID string,
	dimension int,
	numSuperBuckets int,
	ids []string,
	vectors [][]float64,
	store blob.Store,
) (*Index, *enterprise.Config, error) {
	// Create enterprise config with fresh secrets
	enterpriseCfg, err := enterprise.NewConfig(enterpriseID, dimension, numSuperBuckets)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create enterprise config: %w", err)
	}

	// Create builder
	cfg := ConfigFromEnterprise(enterpriseCfg)
	builder, err := NewEnterpriseBuilder(cfg, enterpriseCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create builder: %w", err)
	}

	// Build index
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build index: %w", err)
	}

	// Get updated config with centroids
	updatedCfg := builder.GetEnterpriseConfig()
	updatedCfg.UpdatedAt = time.Now()

	return idx, updatedCfg, nil
}
