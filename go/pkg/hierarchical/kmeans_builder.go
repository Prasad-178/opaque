package hierarchical

import (
	"context"
	"fmt"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/cluster"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/enterprise"
)

// KMeansBuilder constructs an index using k-means clustering for bucket assignment.
//
// Key advantages over LSH-based assignment:
//   - Centroids are optimal representatives (cluster centers by definition)
//   - Vectors in same cluster are actually similar
//   - HE scoring on centroids correlates directly with vector proximity
//   - Much better recall with the same number of buckets (98% vs 83% GT hit rate)
//
// Privacy is maintained: vectors are still AES encrypted, queries HE encrypted,
// and decoy buckets still hide access patterns.
//
// Note: Sub-buckets have been removed as they provided no accuracy benefit
// (sub-bucket IDs don't correlate with vector similarity).
type KMeansBuilder struct {
	config        Config
	enterpriseCfg *enterprise.Config
	kmeans        *cluster.KMeans
	encryptor     *encrypt.AESGCM

	// Building state
	superBuckets []*SuperBucket
	vectorLocs   map[string]*VectorLocation
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

	encryptor, err := encrypt.NewAESGCM(enterpriseCfg.AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create encryptor: %w", err)
	}

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
		config:        cfg,
		enterpriseCfg: enterpriseCfg,
		encryptor:     encryptor,
		superBuckets:  superBuckets,
		vectorLocs:    make(map[string]*VectorLocation),
	}, nil
}

// Build constructs the index using k-means clustering.
// Vectors are stored by super-bucket only (no sub-bucket division).
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

	// Phase 1: Run k-means clustering to determine bucket assignments
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

	// Phase 2: Assign vectors to super-buckets and encrypt
	// Support redundant assignment for improved recall on boundary queries
	numAssignments := b.config.RedundantAssignments
	if numAssignments <= 0 {
		numAssignments = 1
	}

	// Pre-allocate blobs slice (may grow if using redundant assignments)
	blobs := make([]*blob.Blob, 0, len(ids)*numAssignments)

	for i, id := range ids {
		vec := vectors[i]
		normalizedVec := normalizedVecs[i]

		// Get top-N cluster assignments for this vector
		var assignments []int
		if numAssignments == 1 {
			// Fast path: single assignment from k-means labels
			assignments = []int{b.kmeans.Labels[i]}
		} else {
			// Redundant assignment: find top-N nearest clusters
			assignments = b.kmeans.PredictTopK(normalizedVec, numAssignments)
		}

		for assignIdx, superID := range assignments {
			bucketKey := formatSuperBucketKey(superID)

			// Update bucket stats (only count for primary assignment)
			if assignIdx == 0 {
				b.superBuckets[superID].VectorCount++
			}

			// Create unique blob ID for redundant assignments
			blobID := id
			if assignIdx > 0 {
				blobID = fmt.Sprintf("%s_dup%d", id, assignIdx)
			}

			// Track vector location (only for primary assignment)
			if assignIdx == 0 {
				b.vectorLocs[id] = &VectorLocation{
					ID:        id,
					SuperID:   superID,
					BucketKey: bucketKey,
				}
			}

			// Encrypt with enterprise AES key
			// Note: We encrypt with original ID so decryption works with original ID
			ciphertext, err := b.encryptor.EncryptVectorWithID(vec, id)
			if err != nil {
				return nil, fmt.Errorf("failed to encrypt vector %s: %w", id, err)
			}

			blobs = append(blobs, blob.NewBlob(blobID, bucketKey, ciphertext, b.config.Dimension))
		}
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
	idx := &Index{
		Config:          b.config,
		SuperBuckets:    b.superBuckets,
		Centroids:       centroids,
		LSHSuper:        nil, // Not used in k-means mode
		LSHSub:          nil, // Sub-buckets removed
		Store:           store,
		Encryptor:       b.encryptor,
		VectorLocations: b.vectorLocs,
		SubBucketCounts: nil, // Sub-buckets removed
		KMeans:          b.kmeans,
	}

	return idx, nil
}

// formatSuperBucketKey formats a super-bucket ID as a bucket key.
func formatSuperBucketKey(superID int) string {
	return fmt.Sprintf("%02d", superID)
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
