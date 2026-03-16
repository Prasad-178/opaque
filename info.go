package opaque

import (
	"context"
	"fmt"
	"sort"

	"github.com/Prasad-178/opaque/pkg/encrypt"
)

// DBStats contains aggregate statistics about the database.
type DBStats struct {
	// TotalVectors is the total number of vectors (pending + indexed).
	TotalVectors int

	// IndexedVectors is the number of vectors in the built index.
	// Zero if the index has not been built.
	IndexedVectors int

	// PendingVectors is the number of vectors buffered but not yet indexed.
	PendingVectors int

	// ClusterStats contains k-means clustering statistics (zero if not built).
	ClusterStats ClusterStats

	// StorageBackend is the storage backend in use.
	StorageBackend StorageBackend

	// HasPCA is true if PCA dimensionality reduction is enabled.
	HasPCA bool

	// IsReady is true if the index is built and ready for search.
	IsReady bool
}

// Has reports whether a vector with the given ID exists in the DB.
//
// It checks both the built index (blob store) and pending vectors.
// Has is safe for concurrent use.
func (db *DB) Has(ctx context.Context, id string) bool {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Deleted vectors don't exist.
	if db.deletedIDs[id] {
		return false
	}

	// Check pending vectors.
	for _, pid := range db.pendingIDs {
		if pid == id {
			return true
		}
	}

	// Check blob store if built.
	if db.blobStore != nil {
		b, err := db.blobStore.Get(ctx, id)
		if err == nil && b != nil {
			return true
		}
	}

	return false
}

// Get retrieves a vector by ID, decrypting it from the blob store.
//
// Returns [ErrNotReady] if the index has not been built, or [ErrNotFound]
// if no vector with the given ID exists. Get is safe for concurrent use.
func (db *DB) Get(ctx context.Context, id string) ([]float64, error) {
	if id == "" {
		return nil, ErrEmptyID
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.state != stateReady {
		return nil, ErrNotReady
	}

	if db.deletedIDs[id] {
		return nil, ErrNotFound
	}

	b, err := db.blobStore.Get(ctx, id)
	if err != nil {
		return nil, ErrNotFound
	}

	enc, err := encrypt.NewAESGCM(db.enterpriseCfg.AESKey)
	if err != nil {
		return nil, fmt.Errorf("opaque: decryption setup failed: %w", err)
	}

	vec, err := enc.DecryptVectorWithID(b.Ciphertext, id)
	if err != nil {
		return nil, fmt.Errorf("opaque: decryption failed: %w", err)
	}

	return vec, nil
}

// Count returns the number of indexed vectors (in the built index only).
// Returns 0 if the index has not been built.
func (db *DB) Count(ctx context.Context) int {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.blobStore == nil {
		return 0
	}

	stats, err := db.blobStore.Stats(ctx)
	if err != nil {
		return 0
	}
	return int(stats.TotalBlobs)
}

// List returns a paginated slice of vector IDs from the built index.
//
// IDs are returned in sorted order. offset and limit control pagination.
// Returns [ErrNotReady] if the index has not been built.
func (db *DB) List(ctx context.Context, offset, limit int) ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.state != stateReady {
		return nil, ErrNotReady
	}

	buckets, err := db.blobStore.ListBuckets(ctx)
	if err != nil {
		return nil, fmt.Errorf("opaque: failed to list buckets: %w", err)
	}

	// Collect all IDs across buckets.
	var allIDs []string
	seen := make(map[string]bool)
	for _, bucket := range buckets {
		blobs, err := db.blobStore.GetBucket(ctx, bucket)
		if err != nil {
			continue
		}
		for _, b := range blobs {
			if !seen[b.ID] {
				seen[b.ID] = true
				allIDs = append(allIDs, b.ID)
			}
		}
	}

	sort.Strings(allIDs)

	// Apply pagination.
	if offset >= len(allIDs) {
		return []string{}, nil
	}
	end := offset + limit
	if end > len(allIDs) {
		end = len(allIDs)
	}
	return allIDs[offset:end], nil
}

// GetConfig returns a copy of the DB's current configuration.
func (db *DB) GetConfig() Config {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.cfg
}

// Stats returns aggregate statistics about the database.
func (db *DB) Stats(ctx context.Context) DBStats {
	db.mu.RLock()
	defer db.mu.RUnlock()

	indexed := 0
	if db.blobStore != nil {
		if s, err := db.blobStore.Stats(ctx); err == nil {
			indexed = int(s.TotalBlobs)
		}
	}

	s := db.clusterStats
	// pendingIDs includes all vectors ever added (they persist through Build),
	// so total = max(pending, indexed) rather than sum.
	total := len(db.pendingIDs)
	if indexed > total {
		total = indexed
	}

	return DBStats{
		TotalVectors:   total,
		IndexedVectors: indexed,
		PendingVectors: len(db.pendingIDs),
		ClusterStats: ClusterStats{
			NumClusters:   s.NumClusters,
			MinSize:       s.MinSize,
			MaxSize:       s.MaxSize,
			AvgSize:       s.AvgSize,
			EmptyClusters: s.EmptyClusters,
			Iterations:    s.Iterations,
		},
		StorageBackend: db.cfg.Storage,
		HasPCA:         db.pcaModel != nil,
		IsReady:        db.state == stateReady,
	}
}
