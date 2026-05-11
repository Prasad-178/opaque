package hierarchical

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"time"

	"github.com/Prasad-178/opaque/pkg/blob"
	"github.com/Prasad-178/opaque/pkg/encrypt"
	"github.com/Prasad-178/opaque/pkg/enterprise"
	"github.com/Prasad-178/opaque/pkg/lsh"
)

// generateBlobIDPermutation returns a uniform random permutation of [0, n) using
// Fisher-Yates shuffle backed by crypto/rand. The mapping logical_id → π[logical_id]
// is used as the storage super-bucket ID, hiding the centroid-to-storage link from
// the server.
func generateBlobIDPermutation(n int) ([]int, error) {
	π := make([]int, n)
	for i := range π {
		π[i] = i
	}
	for i := n - 1; i > 0; i-- {
		j, err := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		if err != nil {
			return nil, fmt.Errorf("failed to generate permutation: %w", err)
		}
		k := int(j.Int64())
		π[i], π[k] = π[k], π[i]
	}
	return π, nil
}

// nextPow2 returns the smallest power of 2 ≥ n. nextPow2(0) = 1.
func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// computePaddingTargets returns the target blob count for each bucket under
// the chosen PaddingMode. counts maps bucketKey → real-blob-count.
//
//   - PaddingNone: returns nil (no padding).
//   - PaddingBucketed: each bucket independently padded to nextPow2(its own
//     count). Matches the public-API doc on opaque.PaddingBucketed
//     ("each cluster up to next power-of-2 vector count, ~6-12 % storage
//     waste"). Earlier versions of this function padded every bucket to
//     nextPow2(global_max_count) which at imbalanced clusters could ~2× the
//     total blob count and was the primary cause of the DBpedia 1M @ 1536-dim
//     OOM saga (see docs/DBPEDIA_OOM_ANALYSIS.md, May 2026).
//   - PaddingMaxFixed: every bucket padded to global maxCount.
func computePaddingTargets(counts map[string]int, mode enterprise.PaddingMode) map[string]int {
	if mode == enterprise.PaddingNone || len(counts) == 0 {
		return nil
	}
	targets := make(map[string]int, len(counts))
	if mode == enterprise.PaddingMaxFixed {
		maxCount := 0
		for _, c := range counts {
			if c > maxCount {
				maxCount = c
			}
		}
		for k := range counts {
			targets[k] = maxCount
		}
		return targets
	}
	// PaddingBucketed: per-cluster nextPow2.
	for k, c := range counts {
		targets[k] = nextPow2(c)
	}
	return targets
}

// generatePaddingBlobs returns dummy AES-GCM-shaped blobs that fill each
// bucket up to its per-bucket target count. Each padding blob has a uniformly
// random 32-char hex ID and `ciphertextLen` bytes of crypto/rand-sourced
// content. Real ciphertexts are AES-GCM with size = 4*dim + 28 bytes (12-byte
// nonce + 16-byte tag + dim×4-byte float32 plaintext, since commit 7a9a369);
// padding matches that exact shape so a server cannot distinguish padding
// from real on the wire.
//
// On the client, decryption of a padding blob fails GCM auth and the existing
// "skip failed decryptions" path silently drops it. No client-side awareness
// of padding is required.
//
// Backward-compat shim — production builders should call
// streamPaddingBlobs directly (writes to store as they're generated, avoiding
// the in-memory peak). Kept for tests that snapshot the padding set.
func generatePaddingBlobs(counts map[string]int, mode enterprise.PaddingMode, dimension int) ([]*blob.Blob, error) {
	targets := computePaddingTargets(counts, mode)
	if len(targets) == 0 {
		return nil, nil
	}
	ciphertextLen := dimension*4 + 28
	var pad []*blob.Blob
	for bucketKey, target := range targets {
		toAdd := target - counts[bucketKey]
		if toAdd <= 0 {
			continue
		}
		for i := 0; i < toAdd; i++ {
			b, err := newPaddingBlob(bucketKey, dimension, ciphertextLen)
			if err != nil {
				return nil, err
			}
			pad = append(pad, b)
		}
	}
	return pad, nil
}

// newPaddingBlob builds one random-content padding blob shaped exactly like a
// real AES-GCM-sealed float32-vector ciphertext.
func newPaddingBlob(bucketKey string, dimension, ciphertextLen int) (*blob.Blob, error) {
	id, err := randomHexID(32)
	if err != nil {
		return nil, fmt.Errorf("padding ID gen: %w", err)
	}
	ct := make([]byte, ciphertextLen)
	if _, err := rand.Read(ct); err != nil {
		return nil, fmt.Errorf("padding ciphertext gen: %w", err)
	}
	return blob.NewBlob(id, bucketKey, ct, dimension), nil
}

// streamPaddingBlobs generates padding blobs and writes them to `store` in
// chunks of paddingFlushChunk via PutBatch — avoiding both the in-memory
// accumulator peak (up to ~14 GB at 1M × 1536-dim with PaddingBucketed) AND
// the per-blob Put → per-blob index-Flush amplification that turned the
// padding phase into a multi-minute serial bottleneck on file-backed
// storage. PutBatch on FileStore is concurrent-safe and defers the index
// save to Close/Flush, so the padding phase now scales with raw disk
// bandwidth rather than syscall/lock overhead.
func streamPaddingBlobs(ctx context.Context, counts map[string]int, mode enterprise.PaddingMode, dimension int, store blob.Store) error {
	targets := computePaddingTargets(counts, mode)
	if len(targets) == 0 {
		return nil
	}
	const paddingFlushChunk = 1024
	ciphertextLen := dimension*4 + 28
	pending := make([]*blob.Blob, 0, paddingFlushChunk)
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		if err := store.PutBatch(ctx, pending); err != nil {
			return fmt.Errorf("store padding chunk: %w", err)
		}
		pending = pending[:0]
		return nil
	}
	for bucketKey, target := range targets {
		toAdd := target - counts[bucketKey]
		if toAdd <= 0 {
			continue
		}
		for i := 0; i < toAdd; i++ {
			b, err := newPaddingBlob(bucketKey, dimension, ciphertextLen)
			if err != nil {
				return err
			}
			pending = append(pending, b)
			if len(pending) >= paddingFlushChunk {
				if err := flush(); err != nil {
					return err
				}
			}
		}
	}
	return flush()
}

// randomHexID returns 2*nBytes hex characters from crypto/rand. Used to
// generate padding-blob IDs that are indistinguishable from random vector IDs.
func randomHexID(nBytes int) (string, error) {
	buf := make([]byte, nBytes)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	const hex = "0123456789abcdef"
	out := make([]byte, nBytes*2)
	for i, b := range buf {
		out[i*2] = hex[b>>4]
		out[i*2+1] = hex[b&0x0f]
	}
	return string(out), nil
}

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

	// Phase 0: Generate the logical→storage super-bucket ID permutation.
	// Blobs are stored at storage IDs π[superID]; the server therefore cannot
	// link a fetched storage ID back to its centroid coordinates without π,
	// which lives only in client credentials.
	perm, err := generateBlobIDPermutation(b.config.NumSuperBuckets)
	if err != nil {
		return nil, fmt.Errorf("failed to generate blob ID permutation: %w", err)
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
		// Storage uses permuted super-bucket ID; centroid scoring uses logical superID.
		storageBucketKey := formatBucketKey(perm[superID], subID)

		// Update centroid running sum (logical index — centroids[logical_id] used by HE)
		super := b.superBuckets[superID]
		super.VectorCount++
		for j, v := range vec {
			super.sum[j] += v
		}

		// Track counts and locations using the storage bucket key (matches blob storage).
		b.subBucketCounts[storageBucketKey]++
		b.vectorLocs[id] = &VectorLocation{
			ID:        id,
			SuperID:   superID,
			SubID:     subID,
			BucketKey: storageBucketKey,
		}

		// Encrypt with enterprise AES key
		ciphertext, err := b.encryptor.EncryptVectorWithID(vec, id)
		if err != nil {
			return nil, fmt.Errorf("failed to encrypt vector %s: %w", id, err)
		}

		blobs[i] = blob.NewBlob(id, storageBucketKey, ciphertext, b.config.Dimension)
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

	// Update enterprise config with computed centroids and storage permutation.
	b.enterpriseCfg.SetCentroids(centroids)
	b.enterpriseCfg.SetBlobIDPermutation(perm)

	// Phase 3: Store real blobs first (frees the real-blob accumulator before
	// padding generation).
	if err := store.PutBatch(ctx, blobs); err != nil {
		return nil, fmt.Errorf("failed to store blobs: %w", err)
	}
	blobs = nil

	// Phase 4: Generate padding blobs streaming directly into the store.
	// Avoids the in-memory padding-accumulator peak (~14 GB at 1M × 1536-dim
	// with PaddingBucketed). See enterprise_builder.go streamPaddingBlobs.
	if b.enterpriseCfg.PaddingMode != enterprise.PaddingNone {
		if err := streamPaddingBlobs(ctx, b.subBucketCounts, b.enterpriseCfg.PaddingMode, b.config.Dimension, store); err != nil {
			return nil, fmt.Errorf("padding generation: %w", err)
		}
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
