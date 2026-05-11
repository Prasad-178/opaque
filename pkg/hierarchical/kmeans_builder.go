package hierarchical

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/Prasad-178/opaque/pkg/blob"
	"github.com/Prasad-178/opaque/pkg/cluster"
	"github.com/Prasad-178/opaque/pkg/encrypt"
	"github.com/Prasad-178/opaque/pkg/enterprise"
	"github.com/Prasad-178/opaque/pkg/pq"
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
	pqModel       *pq.PQ // Optional PQ model for compact code storage

	// Building state
	superBuckets []*SuperBucket
	vectorLocs   map[string]*VectorLocation
}

// SetPQ sets the product quantizer for PQ code generation during Build.
// When set, each blob will include encrypted PQ codes alongside the full vector.
func (b *KMeansBuilder) SetPQ(model *pq.PQ) {
	b.pqModel = model
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
//
// `vectors` is float32 — the storage tier for raw input vectors. At
// 1M × 1536-dim this is ~6 GB instead of ~12 GB at float64. Conversions
// to float64 happen at the per-vector boundaries where the AES encrypt /
// PQ encode / HE-related callees still expect float64.
func (b *KMeansBuilder) Build(ctx context.Context, ids []string, vectors [][]float32, store blob.Store) (*Index, error) {
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
	// Normalize vectors for better clustering (cosine similarity based).
	// normalizedVecs is float32 — saves 6 GB at 1M × 1536-dim vs prior float64.
	normalizedVecs := make([][]float32, len(vectors))
	for i, vec := range vectors {
		normalizedVecs[i] = cluster.NormalizeVector(vec)
	}

	// Run k-means with k = NumSuperBuckets. MaxIter=50 is plenty in practice —
	// the tolerance early-stop almost always kicks in by iter ~20-30 on real
	// SIFT/ada-002-class data; the prior 100 cap was just dead wall time.
	kmCfg := cluster.Config{
		K:         b.config.NumSuperBuckets,
		MaxIter:   50,
		Tolerance: 1e-4,
		Seed:      b.enterpriseCfg.GetLSHSeedAsInt64(), // Use enterprise seed for reproducibility
		NumInit:   b.config.NumKMeansInit,
	}
	b.kmeans = cluster.NewKMeans(kmCfg)
	if err := b.kmeans.Fit(normalizedVecs); err != nil {
		return nil, fmt.Errorf("k-means clustering failed: %w", err)
	}

	// Use k-means centroids (they're optimal by construction). Centroids are
	// stored as float64 in superBuckets / enterprise config / auth credentials
	// because they're tiny (≈ NumClusters × Dimension × 8 bytes ≪ MB) and
	// downstream HE encoding wants float64 anyway. Upcast on copy.
	for i, centroid := range b.kmeans.Centroids {
		dst := b.superBuckets[i].Centroid
		for j, v := range centroid {
			dst[j] = float64(v)
		}
	}

	// Generate logical→storage permutation. Blobs stored at storage_id =
	// perm[logical_id]; server cannot link a fetched storage ID back to its
	// centroid coordinates without the permutation, which lives only in
	// client credentials. See SECURITY_MODEL.md §5.
	perm, err := generateBlobIDPermutation(b.config.NumSuperBuckets)
	if err != nil {
		return nil, fmt.Errorf("failed to generate blob ID permutation: %w", err)
	}

	// Phase 2: Assign vectors to super-buckets and encrypt
	// Support redundant assignment for improved recall on boundary queries
	numAssignments := b.config.RedundantAssignments
	if numAssignments <= 0 {
		numAssignments = 1
	}

	// Pre-compute cluster assignments, VectorCounts, and vectorLocs (no encryption needed)
	type vectorAssignment struct {
		assignments []int
	}
	allAssignments := make([]vectorAssignment, len(ids))
	for i, id := range ids {
		var assignments []int
		if numAssignments == 1 {
			assignments = []int{b.kmeans.Labels[i]}
		} else {
			assignments = b.kmeans.PredictTopK(normalizedVecs[i], numAssignments)
		}
		allAssignments[i] = vectorAssignment{assignments: assignments}

		// Pre-compute metadata from primary assignment
		superID := assignments[0]
		b.superBuckets[superID].VectorCount++
		// vectorLocs tracks the storage bucketKey (matches what's stored on
		// disk). superID stays logical for HE-scoring metadata.
		bucketKey := formatSuperBucketKey(perm[superID])
		b.vectorLocs[id] = &VectorLocation{
			ID:        id,
			SuperID:   superID,
			BucketKey: bucketKey,
		}
	}

	// Free normalizedVecs before the parallel-encrypt phase. At 1M × 1536-dim
	// this slice peaks at ~12 GB; the workers below re-normalize per vector
	// on demand, which adds ~5-10 s of CPU but reclaims the buffer for Go's
	// GC during the AES-encrypt phase. Empirically this is the largest single
	// reclaimable working-set in the build path. See benchmarks/results
	// commit history for the m6i.4xlarge OOM at 1M × 1536-dim that motivated
	// this change.
	normalizedVecs = nil

	// Parallel encryption: split vectors across workers
	numWorkers := runtime.NumCPU()
	if numWorkers > len(ids) {
		numWorkers = len(ids)
	}
	chunkSize := (len(ids) + numWorkers - 1) / numWorkers

	type workerResult struct {
		blobs []*blob.Blob
		err   error
	}
	results := make([]workerResult, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(ids) {
			end = len(ids)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(w, start, end int) {
			defer wg.Done()
			localBlobs := make([]*blob.Blob, 0, (end-start)*numAssignments)

			for i := start; i < end; i++ {
				id := ids[i]
				vec := vectors[i]

				// Re-normalize per vector on demand (instead of holding all 1M
				// normalized copies in normalizedVecs above). Only allocate the
				// normalized buffer when we actually need it.
				var normalizedVec []float32
				if b.config.NormalizedStorage || b.pqModel != nil {
					normalizedVec = cluster.NormalizeVector(vec)
				}
				if b.config.NormalizedStorage {
					vec = normalizedVec
				}

				// PQ-encode once per vector (codes are the same for all assignments).
				// PQ.Encode still takes float64 — convert this single vector at the
				// boundary (~12 KB temp at 1536-dim, GC reclaims immediately).
				var pqCiphertext []byte
				if b.pqModel != nil {
					codes := b.pqModel.Encode(cluster.AsFloat64One(normalizedVec))
					ct, err := b.encryptor.EncryptWithAAD(codes, []byte(id))
					if err != nil {
						results[w] = workerResult{err: fmt.Errorf("failed to encrypt PQ codes for %s: %w", id, err)}
						return
					}
					pqCiphertext = ct
				}

				// Convert this single vector to float64 once for the AES-encrypt
				// boundary (encrypt.EncryptVectorWithID takes float64). Reused
				// across all redundant assignments below.
				vec64 := cluster.AsFloat64One(vec)

				for assignIdx, superID := range allAssignments[i].assignments {
					// Storage uses permuted ID; centroid scoring uses logical superID.
					bucketKey := formatSuperBucketKey(perm[superID])

					blobID := id
					if assignIdx > 0 {
						blobID = fmt.Sprintf("%s_dup%d", id, assignIdx)
					}

					// Encrypt with enterprise AES key (AESGCM.Seal is safe for concurrent use)
					ciphertext, err := b.encryptor.EncryptVectorWithID(vec64, id)
					if err != nil {
						results[w] = workerResult{err: fmt.Errorf("failed to encrypt vector %s: %w", id, err)}
						return
					}

					newBlob := blob.NewBlob(blobID, bucketKey, ciphertext, b.config.Dimension)
					if pqCiphertext != nil {
						newBlob.PQCiphertext = pqCiphertext
					}
					localBlobs = append(localBlobs, newBlob)
				}
			}
			results[w] = workerResult{blobs: localBlobs}
		}(w, start, end)
	}
	wg.Wait()

	// Merge results and check for errors
	blobs := make([]*blob.Blob, 0, len(ids)*numAssignments)
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
		blobs = append(blobs, r.blobs...)
	}

	// Extract centroids for enterprise config
	centroids := make([][]float64, b.config.NumSuperBuckets)
	for i, super := range b.superBuckets {
		centroids[i] = make([]float64, len(super.Centroid))
		copy(centroids[i], super.Centroid)
	}

	// Update enterprise config with k-means centroids and storage permutation.
	b.enterpriseCfg.SetCentroids(centroids)
	b.enterpriseCfg.SetBlobIDPermutation(perm)

	// Phase 3: Store the real encrypted blobs. We flush BEFORE generating
	// padding so the real-blob accumulator can be released and so the padding
	// streaming loop has the file store warmed up. Padding blobs share the
	// storage bucketKey of real blobs but contain random AES-GCM-shaped bytes;
	// client AES decryption fails GCM auth and the existing "skip failed
	// decryptions" path drops them silently.
	counts := make(map[string]int, b.config.NumSuperBuckets)
	for _, blb := range blobs {
		counts[blb.LSHBucket]++
	}
	if err := store.PutBatch(ctx, blobs); err != nil {
		return nil, fmt.Errorf("failed to store blobs: %w", err)
	}
	blobs = nil // free the real-blob accumulator before generating padding

	// Phase 4: Generate padding directly into the store, one blob at a time.
	// At 1M × 1536-dim with PaddingBucketed this avoids ~14 GB of in-memory
	// padding accumulator vs the prior pattern that appended all padding
	// into the same slice and called PutBatch once.
	if b.enterpriseCfg.PaddingMode != enterprise.PaddingNone {
		if err := streamPaddingBlobs(ctx, counts, b.enterpriseCfg.PaddingMode, b.config.Dimension, store); err != nil {
			return nil, fmt.Errorf("padding generation: %w", err)
		}
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
	return FormatBucketKey(superID)
}

// FormatBucketKey formats a super-bucket ID as a two-digit zero-padded bucket key.
func FormatBucketKey(superID int) string {
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
	minSize := math.MaxInt
	maxSize, totalSize, emptyClusters := 0, 0, 0
	for _, s := range sizes {
		if s == 0 {
			emptyClusters++
			continue
		}
		if s < minSize {
			minSize = s
		}
		if s > maxSize {
			maxSize = s
		}
		totalSize += s
	}
	if minSize == math.MaxInt {
		minSize = 0 // All clusters empty (shouldn't happen)
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

// GetClusterCounts returns the number of vectors in each cluster (primary assignments only).
func (b *KMeansBuilder) GetClusterCounts() []int {
	counts := make([]int, len(b.superBuckets))
	for i, sb := range b.superBuckets {
		counts[i] = sb.VectorCount
	}
	return counts
}

// BuildKMeansIndex is a convenience function for building with k-means.
// Accepts float64 input for backwards compatibility; converts to float32
// internally before passing to the builder. The float64 caller's slice is
// discarded after this conversion (best-effort: caller still holds the
// reference, but builder's working set is float32).
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

	// Build index — convert input vectors to float32 at this convenience-API
	// boundary. opaque.NewDB's path stores float32 directly and avoids the
	// double-allocation; this helper is mostly used by tests + small examples.
	idx, err := builder.Build(ctx, ids, cluster.AsFloat32(vectors), store)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build index: %w", err)
	}

	// Get updated config
	updatedCfg := builder.GetEnterpriseConfig()
	updatedCfg.UpdatedAt = time.Now()

	return idx, updatedCfg, nil
}
