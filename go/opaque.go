// Package opaque provides privacy-preserving vector search using homomorphic encryption.
//
// Opaque encrypts vectors with AES-256-GCM, scores queries against cluster centroids
// using CKKS homomorphic encryption (so the server never sees the query), and hides
// access patterns with decoy bucket fetches.
//
// # Quick Start
//
//	db, err := opaque.NewDB(opaque.Config{
//	    Dimension:   128,
//	    NumClusters: 64,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer db.Close()
//
//	// Add vectors
//	db.Add(ctx, "doc-1", vector1)
//	db.Add(ctx, "doc-2", vector2)
//
//	// Build the index (runs k-means clustering + initializes HE engines)
//	if err := db.Build(ctx); err != nil {
//	    log.Fatal(err)
//	}
//
//	// Search
//	results, err := db.Search(ctx, queryVector, 10)
//
// # Lifecycle
//
// The DB follows a three-phase lifecycle:
//  1. Add vectors with [DB.Add] or [DB.AddBatch]
//  2. Build the index with [DB.Build] (expensive: k-means clustering + HE engine initialization)
//  3. Search with [DB.Search] (safe for concurrent use)
//
// K-means clustering requires all vectors upfront, so [DB.Build] must be called after
// all vectors are added. To add vectors after building, use [DB.Add] followed by [DB.Rebuild].
package opaque

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
	"github.com/opaque/opaque/go/pkg/pca"
)

// StorageBackend selects where encrypted vector blobs are stored.
type StorageBackend int

const (
	// Memory stores all data in RAM. Fast but not persistent across restarts.
	Memory StorageBackend = iota

	// File stores encrypted blobs on disk at the path specified by [Config.StoragePath].
	// Persistent across restarts, slower than memory for large datasets.
	File
)

// Config controls the behavior of a [DB] instance.
//
// Only [Config.Dimension] is required. All other fields have sensible defaults.
type Config struct {
	// Dimension is the length of each vector. Required.
	// All vectors added to the DB must have exactly this many elements.
	Dimension int

	// NumClusters is the number of k-means clusters used to partition vectors.
	// More clusters means faster search (fewer vectors per cluster) but weaker
	// privacy (smaller anonymity sets per cluster). Must be >= 2.
	// Default: 64.
	NumClusters int

	// TopClusters is the number of clusters probed during each search.
	// Higher values improve recall at the cost of more computation, bandwidth,
	// and weaker access pattern privacy (more clusters probed = easier to infer intent).
	// Must be <= NumClusters.
	// Default: max(NumClusters / 16, 4). For 64 clusters this is 4 (~6% of data).
	TopClusters int

	// NumDecoys is the number of extra clusters fetched per search to hide
	// which clusters are actually relevant. Higher values provide better
	// access pattern privacy at the cost of additional bandwidth.
	// Default: 8.
	NumDecoys int

	// WorkerPoolSize is the number of parallel CKKS homomorphic encryption engines.
	// Each engine consumes ~50MB of memory but enables parallel centroid scoring.
	// Set to 0 for automatic sizing (min(NumCPU, 8)).
	// Default: 0 (automatic).
	WorkerPoolSize int

	// Storage selects the backend for encrypted blob storage.
	// Default: Memory.
	Storage StorageBackend

	// StoragePath is the directory for file-backed storage.
	// Required when Storage is [File], ignored otherwise.
	StoragePath string

	// ProbeThreshold controls multi-probe cluster selection during search.
	// Clusters scoring within this fraction of the top cluster score are also probed,
	// beyond the TopClusters limit. For example, 0.95 means clusters within 5% of
	// the best score are included.
	// Set to 1.0 to disable multi-probe (strict top-K only).
	// Default: 0.95.
	ProbeThreshold float64

	// RedundantAssignments assigns each vector to multiple clusters during indexing.
	// Improves recall for vectors near cluster boundaries at the cost of increased storage.
	// A value of 2 means each vector is stored in its 2 nearest clusters.
	// Default: 1 (no redundancy).
	RedundantAssignments int

	// PCADimension enables optional PCA dimensionality reduction.
	// When set to a positive value, vectors are projected to this dimension
	// before clustering and encryption, reducing latency and bandwidth.
	// The PCA transform is applied client-side, so it has no privacy impact.
	// Must be less than Dimension. Set to 0 to disable (default).
	// Default: 0 (disabled).
	PCADimension int

	// NumKMeansInit is the number of k-means clustering initializations to run.
	// Multiple runs with different seeds are executed in parallel, and the result
	// with the lowest inertia (best cluster quality) is kept.
	// Higher values improve cluster quality at the cost of more CPU during Build.
	// Default: 1 (single initialization).
	NumKMeansInit int
}

// Result is a single search result containing the vector ID and its similarity score.
type Result struct {
	// ID is the identifier passed to [DB.Add] when the vector was indexed.
	ID string

	// Score is the cosine similarity between the query and this vector.
	// Higher is more similar. Range: [-1, 1] for normalized vectors.
	Score float64
}

// dbState tracks the lifecycle phase of a [DB].
type dbState int

const (
	stateEmpty    dbState = iota // No vectors added yet.
	stateBuffered                // Vectors added, index not built.
	stateReady                   // Index built, ready for search.
)

// DB is a privacy-preserving vector search database.
//
// It encrypts stored vectors with AES-256-GCM, scores queries against cluster centroids
// using CKKS homomorphic encryption, and fetches decoy clusters to hide access patterns.
//
// A DB must be built before searching. After [DB.Build] completes, [DB.Search] is safe
// for concurrent use from multiple goroutines.
type DB struct {
	cfg Config

	mu    sync.RWMutex
	state dbState

	// Buffered vectors (accumulated via Add/AddBatch, consumed by Build).
	pendingIDs     []string
	pendingVectors [][]float64

	// Built state (populated by Build, used by Search).
	blobStore    blob.Store
	searchClient *client.EnterpriseHierarchicalClient
	pcaModel     *pca.PCA // nil when PCA is disabled
	clusterStats hierarchical.ClusterStats
}

// NewDB creates a new vector search database with the given configuration.
//
// Only [Config.Dimension] is required; all other fields use sensible defaults if zero.
// No expensive initialization happens here — the heavy work is deferred to [DB.Build].
func NewDB(cfg Config) (*DB, error) {
	if cfg.Dimension <= 0 {
		return nil, fmt.Errorf("opaque: Dimension is required and must be positive, got %d", cfg.Dimension)
	}

	// Validate user-provided values before applying defaults.
	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}

	applyDefaults(&cfg)

	return &DB{
		cfg:   cfg,
		state: stateEmpty,
	}, nil
}

// Add buffers a single vector for indexing. The id must be unique within the DB.
//
// Add must be called before [DB.Build]. After Build, use [DB.Rebuild] to incorporate
// new vectors.
//
// The vector is copied internally, so the caller may modify the slice after Add returns.
func (db *DB) Add(ctx context.Context, id string, vector []float64) error {
	if len(vector) != db.cfg.Dimension {
		return fmt.Errorf("opaque: vector dimension %d does not match DB dimension %d", len(vector), db.cfg.Dimension)
	}
	if id == "" {
		return fmt.Errorf("opaque: vector ID must not be empty")
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if db.state == stateReady {
		return fmt.Errorf("opaque: cannot Add after Build; use Rebuild to add vectors to an existing index")
	}

	v := make([]float64, len(vector))
	copy(v, vector)

	db.pendingIDs = append(db.pendingIDs, id)
	db.pendingVectors = append(db.pendingVectors, v)
	db.state = stateBuffered
	return nil
}

// AddBatch buffers multiple vectors for indexing. The ids and vectors slices must
// have the same length. Each vector must have exactly [Config.Dimension] elements.
//
// This is equivalent to calling [DB.Add] for each vector, but acquires the lock once.
func (db *DB) AddBatch(ctx context.Context, ids []string, vectors [][]float64) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("opaque: ids length %d does not match vectors length %d", len(ids), len(vectors))
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if db.state == stateReady {
		return fmt.Errorf("opaque: cannot AddBatch after Build; use Rebuild to add vectors to an existing index")
	}

	for i, v := range vectors {
		if len(v) != db.cfg.Dimension {
			return fmt.Errorf("opaque: vector %d (id=%q) has dimension %d, expected %d", i, ids[i], len(v), db.cfg.Dimension)
		}
		if ids[i] == "" {
			return fmt.Errorf("opaque: vector %d has empty ID", i)
		}
	}

	for i, v := range vectors {
		vc := make([]float64, len(v))
		copy(vc, v)
		db.pendingIDs = append(db.pendingIDs, ids[i])
		db.pendingVectors = append(db.pendingVectors, vc)
	}
	db.state = stateBuffered
	return nil
}

// Build creates the search index from all buffered vectors.
//
// This is the most expensive operation in the lifecycle:
//   - Runs k-means clustering to partition vectors into clusters
//   - Encrypts each vector with AES-256-GCM
//   - Initializes the CKKS homomorphic encryption engine pool
//   - Pre-encodes cluster centroids as HE plaintexts
//
// After Build returns successfully, [DB.Search] is ready for use. Build must only
// be called once; use [DB.Rebuild] to re-index after adding new vectors.
func (db *DB) Build(ctx context.Context) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	return db.buildLocked(ctx)
}

// Rebuild re-indexes all vectors including any added since the last Build.
//
// This performs a full rebuild: the old index is discarded and a new one is created
// from all accumulated vectors. Use this after adding vectors to a built DB:
//
//	db.Build(ctx)           // initial build
//	// ... later ...
//	db.Rebuild(ctx)         // add pending vectors, rebuild from scratch
//
// Rebuild is not safe for concurrent use with Search.
func (db *DB) Rebuild(ctx context.Context) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(db.pendingIDs) == 0 {
		return fmt.Errorf("opaque: no vectors to rebuild")
	}

	// Tear down old state.
	db.closeStoreLocked()
	db.searchClient = nil
	db.state = stateBuffered

	return db.buildLocked(ctx)
}

// Search returns the topK most similar vectors to the query.
//
// Results are sorted by descending cosine similarity score. The query vector must
// have exactly [Config.Dimension] elements.
//
// Search uses SIMD-optimized batch HE operations internally for best performance.
// It is safe for concurrent use from multiple goroutines after [DB.Build] completes.
func (db *DB) Search(ctx context.Context, query []float64, topK int) ([]Result, error) {
	if len(query) != db.cfg.Dimension {
		return nil, fmt.Errorf("opaque: query dimension %d does not match DB dimension %d", len(query), db.cfg.Dimension)
	}
	if topK <= 0 {
		return nil, fmt.Errorf("opaque: topK must be positive, got %d", topK)
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	switch db.state {
	case stateEmpty:
		return nil, fmt.Errorf("opaque: no vectors indexed; call Add and Build first")
	case stateBuffered:
		return nil, fmt.Errorf("opaque: index not built; call Build before Search")
	}

	// Apply PCA transform to query if PCA is enabled.
	searchQuery := query
	if db.pcaModel != nil {
		reduced, err := db.pcaModel.Transform(query)
		if err != nil {
			return nil, fmt.Errorf("opaque: PCA transform failed: %w", err)
		}
		searchQuery = reduced
	}

	sr, err := db.searchClient.SearchBatch(ctx, searchQuery, topK)
	if err != nil {
		return nil, fmt.Errorf("opaque: search failed: %w", err)
	}

	results := make([]Result, len(sr.Results))
	for i, r := range sr.Results {
		results[i] = Result{ID: r.ID, Score: r.Score}
	}
	return results, nil
}

// Size returns the total number of vectors in the DB (both pending and indexed).
func (db *DB) Size() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.pendingIDs)
}

// IsReady reports whether the index has been built and the DB is ready for search.
func (db *DB) IsReady() bool {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.state == stateReady
}

// ClusterStats returns statistics about the k-means clustering from the most recent Build.
// Returns a zero value if the index has not been built yet.
func (db *DB) ClusterStats() ClusterStats {
	db.mu.RLock()
	defer db.mu.RUnlock()
	s := db.clusterStats
	return ClusterStats{
		NumClusters:   s.NumClusters,
		MinSize:       s.MinSize,
		MaxSize:       s.MaxSize,
		AvgSize:       s.AvgSize,
		EmptyClusters: s.EmptyClusters,
		Iterations:    s.Iterations,
	}
}

// ClusterStats contains statistics about k-means clustering quality.
type ClusterStats struct {
	NumClusters   int     // Number of clusters
	MinSize       int     // Smallest cluster size
	MaxSize       int     // Largest cluster size
	AvgSize       float64 // Average cluster size
	EmptyClusters int     // Number of empty clusters (should be 0)
	Iterations    int     // K-means iterations until convergence
}

// Close releases all resources held by the DB, including the blob store and
// HE engine pool. The DB must not be used after Close is called.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	db.searchClient = nil
	db.pcaModel = nil
	db.closeStoreLocked()
	db.state = stateEmpty
	return nil
}

// --- Internal helpers ---

// buildLocked performs the actual index build. Caller must hold db.mu write lock.
func (db *DB) buildLocked(ctx context.Context) error {
	if len(db.pendingIDs) == 0 {
		return fmt.Errorf("opaque: no vectors added; call Add or AddBatch before Build")
	}

	// Apply PCA dimensionality reduction if configured.
	indexVectors := db.pendingVectors
	indexDim := db.cfg.Dimension
	if db.cfg.PCADimension > 0 {
		model, err := pca.Fit(db.pendingVectors, db.cfg.PCADimension)
		if err != nil {
			return fmt.Errorf("opaque: PCA fitting failed: %w", err)
		}
		reduced, err := model.TransformBatch(db.pendingVectors)
		if err != nil {
			return fmt.Errorf("opaque: PCA transform failed: %w", err)
		}
		db.pcaModel = model
		indexVectors = reduced
		indexDim = db.cfg.PCADimension
	} else {
		db.pcaModel = nil
	}

	// Create blob store.
	store, err := db.createBlobStore()
	if err != nil {
		return fmt.Errorf("opaque: %w", err)
	}

	// Generate a unique enterprise ID for key namespacing.
	enterpriseID := db.cfg.enterpriseID()

	// Build the k-means index: clusters vectors, encrypts with AES-256-GCM, stores blobs.
	ecfgBuild, err := enterprise.NewConfig(enterpriseID, indexDim, db.cfg.NumClusters)
	if err != nil {
		store.Close()
		return fmt.Errorf("opaque: enterprise config: %w", err)
	}
	builderCfg := hierarchical.ConfigFromEnterprise(ecfgBuild)
	builderCfg.NumKMeansInit = db.cfg.NumKMeansInit
	builderCfg.RedundantAssignments = db.cfg.RedundantAssignments
	builder, err := hierarchical.NewKMeansBuilder(builderCfg, ecfgBuild)
	if err != nil {
		store.Close()
		return fmt.Errorf("opaque: builder init: %w", err)
	}
	_, err = builder.Build(ctx, db.pendingIDs, indexVectors, store)
	if err != nil {
		store.Close()
		return fmt.Errorf("opaque: index build failed: %w", err)
	}
	db.clusterStats = builder.GetClusterStats()
	enterpriseCfg := builder.GetEnterpriseConfig()
	enterpriseCfg.UpdatedAt = time.Now()

	// Construct credentials directly (bypass auth service for library usage).
	creds := makeCredentials(enterpriseCfg)

	// Map library config to internal search config (use effective dimension post-PCA).
	searchCfg := db.makeSearchConfig(enterpriseCfg, indexDim)

	// Create the search client with configurable pool size.
	searchClient, err := client.NewEnterpriseHierarchicalClientWithPoolSize(
		searchCfg, creds, store, db.cfg.WorkerPoolSize,
	)
	if err != nil {
		store.Close()
		return fmt.Errorf("opaque: failed to create search client: %w", err)
	}

	db.blobStore = store
	db.searchClient = searchClient
	db.state = stateReady
	return nil
}

// createBlobStore creates a new blob store based on the configured storage backend.
func (db *DB) createBlobStore() (blob.Store, error) {
	switch db.cfg.Storage {
	case Memory:
		return blob.NewMemoryStore(), nil
	case File:
		if db.cfg.StoragePath == "" {
			return nil, fmt.Errorf("StoragePath is required when Storage is File")
		}
		// Ensure the directory exists.
		if err := os.MkdirAll(db.cfg.StoragePath, 0o700); err != nil {
			return nil, fmt.Errorf("failed to create storage directory: %w", err)
		}
		return blob.NewFileStore(db.cfg.StoragePath)
	default:
		return nil, fmt.Errorf("unknown storage backend: %d", db.cfg.Storage)
	}
}

// makeCredentials constructs auth.ClientCredentials directly from an enterprise config.
// This bypasses the auth service, which is only needed for multi-user server deployments.
func makeCredentials(ecfg *enterprise.Config) *auth.ClientCredentials {
	return &auth.ClientCredentials{
		Token:           "opaque-library",
		TokenExpiry:     time.Now().Add(100 * 365 * 24 * time.Hour),
		AESKey:          ecfg.AESKey,
		Centroids:       ecfg.Centroids,
		EnterpriseID:    ecfg.EnterpriseID,
		Dimension:       ecfg.Dimension,
		NumSuperBuckets: ecfg.NumSuperBuckets,
		NumSubBuckets:   ecfg.NumSubBuckets,
	}
}

// makeSearchConfig maps the library Config to the internal hierarchical.Config
// used by the search client.
func (db *DB) makeSearchConfig(ecfg *enterprise.Config, effectiveDim int) hierarchical.Config {
	maxProbe := max(db.cfg.TopClusters*2, db.cfg.TopClusters)

	return hierarchical.Config{
		Dimension:            effectiveDim,
		NumSuperBuckets:      db.cfg.NumClusters,
		NumSubBuckets:        1,
		TopSuperBuckets:      db.cfg.TopClusters,
		SubBucketsPerSuper:   1,
		NumDecoys:            db.cfg.NumDecoys,
		ProbeThreshold:       db.cfg.ProbeThreshold,
		MaxProbeClusters:     maxProbe,
		RedundantAssignments: db.cfg.RedundantAssignments,
		LSHSuperSeed:         ecfg.GetLSHSeedAsInt64(),
		LSHSubSeed:           ecfg.GetSubLSHSeedAsInt64(),
	}
}

// closeStoreLocked closes the blob store if open. Caller must hold db.mu.
func (db *DB) closeStoreLocked() {
	if db.blobStore != nil {
		db.blobStore.Close()
		db.blobStore = nil
	}
}

// enterpriseID returns the configured enterprise ID or generates a random one.
func (cfg *Config) enterpriseID() string {
	if cfg.StoragePath != "" {
		// Use storage path as a stable identifier for file-backed DBs.
		return fmt.Sprintf("opaque-%s", cfg.StoragePath)
	}
	return generateID()
}

// applyDefaults fills zero-value fields with production-ready defaults.
func applyDefaults(cfg *Config) {
	if cfg.NumClusters <= 0 {
		cfg.NumClusters = 64
	}
	if cfg.TopClusters <= 0 {
		cfg.TopClusters = max(cfg.NumClusters/16, 4)
	}
	if cfg.NumDecoys <= 0 {
		cfg.NumDecoys = 8
	}
	if cfg.WorkerPoolSize <= 0 {
		cfg.WorkerPoolSize = min(runtime.NumCPU(), 8)
	}
	if cfg.ProbeThreshold <= 0 {
		cfg.ProbeThreshold = 0.95
	}
	if cfg.RedundantAssignments <= 0 {
		cfg.RedundantAssignments = 1
	}
}

// validateConfig checks user-provided config values before defaults are applied.
// Zero values are allowed since they will be replaced by defaults.
func validateConfig(cfg *Config) error {
	if cfg.NumClusters != 0 && cfg.NumClusters < 2 {
		return fmt.Errorf("opaque: NumClusters must be >= 2, got %d", cfg.NumClusters)
	}
	if cfg.NumClusters > 0 && cfg.TopClusters > cfg.NumClusters {
		return fmt.Errorf("opaque: TopClusters (%d) must be <= NumClusters (%d)", cfg.TopClusters, cfg.NumClusters)
	}
	if cfg.ProbeThreshold != 0 && (cfg.ProbeThreshold < 0 || cfg.ProbeThreshold > 1) {
		return fmt.Errorf("opaque: ProbeThreshold must be in [0, 1], got %f", cfg.ProbeThreshold)
	}
	if cfg.Storage == File && cfg.StoragePath == "" {
		return fmt.Errorf("opaque: StoragePath is required when Storage is File")
	}
	if cfg.PCADimension != 0 && cfg.PCADimension >= cfg.Dimension {
		return fmt.Errorf("opaque: PCADimension (%d) must be less than Dimension (%d)", cfg.PCADimension, cfg.Dimension)
	}
	return nil
}

// generateID creates a random 16-character hex identifier.
func generateID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand should never fail; fall back to timestamp.
		return fmt.Sprintf("opaque-%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("opaque-%x", b)
}
