package hierarchical

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/cluster"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// KMeansBuilder constructs a hierarchical index using k-means clustering
// for super-bucket assignment instead of LSH.
//
// Key advantages over LSH-based assignment:
//   - Centroids are optimal representatives (cluster centers by definition)
//   - Vectors in same cluster are actually similar
//   - HE scoring on centroids correlates directly with vector proximity
//   - Much better recall with the same number of buckets
//
// Privacy is maintained: vectors are still AES encrypted, queries HE encrypted,
// and decoy buckets still hide access patterns.
type KMeansBuilder struct {
	config        Config
	enterpriseCfg *enterprise.Config
	kmeans        *cluster.KMeans
	lshSub        *lsh.Index // LSH still used for sub-buckets (for privacy)
	encryptor     *encrypt.AESGCM

	// Building state
	superBuckets    []*SuperBucket
	vectorLocs      map[string]*VectorLocation
	subBucketCounts map[string]int
}

// NewKMeansBuilder creates a new builder using k-means clustering.
func NewKMeansBuilder(cfg Config, enterpriseCfg *enterprise.Config) (*KMeansBuilder, error) {
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
	numSubBuckets := enterpriseCfg.NumSubBuckets
	if numSubBuckets <= 0 {
		numSubBuckets = 64 // Default
	}
	cfg.NumSubBuckets = numSubBuckets

	encryptor, err := encrypt.NewAESGCM(enterpriseCfg.AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create encryptor: %w", err)
	}

	// LSH for sub-buckets (still needed for sub-bucket assignment)
	subBits := int(math.Ceil(math.Log2(float64(cfg.NumSubBuckets)))) + 2
	if subBits < 6 {
		subBits = 6
	}
	subSeed := enterpriseCfg.GetSubLSHSeedAsInt64()
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

	return &KMeansBuilder{
		config:          cfg,
		enterpriseCfg:   enterpriseCfg,
		lshSub:          lshSub,
		encryptor:       encryptor,
		superBuckets:    superBuckets,
		vectorLocs:      make(map[string]*VectorLocation),
		subBucketCounts: make(map[string]int),
	}, nil
}

// Build constructs the hierarchical index using k-means clustering.
func (b *KMeansBuilder) Build(ctx context.Context, ids []string, vectors [][]float64, store blob.Store) (*Index, error) {
	if len(ids) != len(vectors) {
		return nil, fmt.Errorf("ids and vectors length mismatch: %d vs %d", len(ids), len(vectors))
	}

	// Validate dimensions
	for i, vec := range vectors {
		if len(vec) != b.config.Dimension {
			return nil, fmt.Errorf("vector %s has wrong dimension: %d (expected %d)",
				ids[i], len(vec), b.config.Dimension)
		}
	}

	// Phase 1: Run k-means clustering to determine super-bucket assignments
	// Normalize vectors for better clustering (cosine similarity based)
	normalizedVecs := make([][]float64, len(vectors))
	for i, vec := range vectors {
		normalizedVecs[i] = cluster.NormalizeVector(vec)
	}

	// Run k-means with k = NumSuperBuckets
	kmCfg := cluster.Config{
		K:         b.config.NumSuperBuckets,
		MaxIter:   100,
		Tolerance: 1e-4,
		Seed:      b.enterpriseCfg.GetLSHSeedAsInt64(), // Use enterprise seed for reproducibility
	}
	b.kmeans = cluster.NewKMeans(kmCfg)
	if err := b.kmeans.Fit(normalizedVecs); err != nil {
		return nil, fmt.Errorf("k-means clustering failed: %w", err)
	}

	// Use k-means centroids (they're optimal by construction!)
	for i, centroid := range b.kmeans.Centroids {
		copy(b.superBuckets[i].Centroid, centroid)
	}

	// Phase 2: Assign vectors to buckets and encrypt
	blobs := make([]*blob.Blob, len(ids))
	for i, id := range ids {
		vec := vectors[i]

		// Super-bucket from k-means clustering
		superID := b.kmeans.Labels[i]

		// Sub-bucket still uses LSH (for privacy within cluster)
		subID := b.lshSub.HashToIndexFromVector(vec, b.config.NumSubBuckets)
		bucketKey := formatBucketKey(superID, subID)

		// Update bucket stats
		b.superBuckets[superID].VectorCount++
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

	// Extract centroids for enterprise config
	centroids := make([][]float64, b.config.NumSuperBuckets)
	for i, super := range b.superBuckets {
		centroids[i] = make([]float64, len(super.Centroid))
		copy(centroids[i], super.Centroid)
	}

	// Update enterprise config with k-means centroids
	b.enterpriseCfg.SetCentroids(centroids)

	// Phase 3: Store all encrypted blobs
	if err := store.PutBatch(ctx, blobs); err != nil {
		return nil, fmt.Errorf("failed to store blobs: %w", err)
	}

	// Build index structure
	// Note: LSHSuper is nil for k-means mode, we store kmeans instead
	idx := &Index{
		Config:          b.config,
		SuperBuckets:    b.superBuckets,
		Centroids:       centroids,
		LSHSuper:        nil, // Not used in k-means mode
		LSHSub:          b.lshSub,
		Store:           store,
		Encryptor:       b.encryptor,
		VectorLocations: b.vectorLocs,
		SubBucketCounts: b.subBucketCounts,
		KMeans:          b.kmeans, // Store k-means for query assignment
	}

	return idx, nil
}

// GetEnterpriseConfig returns the enterprise configuration with k-means centroids.
func (b *KMeansBuilder) GetEnterpriseConfig() *enterprise.Config {
	return b.enterpriseCfg
}

// GetClusterStats returns statistics about the k-means clustering.
func (b *KMeansBuilder) GetClusterStats() ClusterStats {
	if b.kmeans == nil {
		return ClusterStats{}
	}

	sizes := b.kmeans.GetClusterSizes()
	minSize, maxSize, totalSize := len(sizes), 0, 0
	for _, s := range sizes {
		if s < minSize && s > 0 {
			minSize = s
		}
		if s > maxSize {
			maxSize = s
		}
		totalSize += s
	}

	emptyClusters := 0
	for _, s := range sizes {
		if s == 0 {
			emptyClusters++
		}
	}

	return ClusterStats{
		NumClusters:   b.config.NumSuperBuckets,
		MinSize:       minSize,
		MaxSize:       maxSize,
		AvgSize:       float64(totalSize) / float64(len(sizes)),
		EmptyClusters: emptyClusters,
		Iterations:    b.kmeans.Iterations,
	}
}

// ClusterStats contains statistics about k-means clustering.
type ClusterStats struct {
	NumClusters   int
	MinSize       int
	MaxSize       int
	AvgSize       float64
	EmptyClusters int
	Iterations    int
}

// BuildKMeansIndex is a convenience function for building with k-means.
func BuildKMeansIndex(
	ctx context.Context,
	enterpriseID string,
	dimension int,
	numSuperBuckets int,
	ids []string,
	vectors [][]float64,
	store blob.Store,
) (*Index, *enterprise.Config, error) {
	// Create enterprise config
	enterpriseCfg, err := enterprise.NewConfig(enterpriseID, dimension, numSuperBuckets)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create enterprise config: %w", err)
	}

	// Create builder
	cfg := ConfigFromEnterprise(enterpriseCfg)
	builder, err := NewKMeansBuilder(cfg, enterpriseCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create builder: %w", err)
	}

	// Build index
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build index: %w", err)
	}

	// Get updated config
	updatedCfg := builder.GetEnterpriseConfig()
	updatedCfg.UpdatedAt = time.Now()

	return idx, updatedCfg, nil
}
